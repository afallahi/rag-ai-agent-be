import os
import logging
from concurrent.futures import ThreadPoolExecutor
from main.config import Config
from main.retrieval.vector_store import faiss_indexer
from main.utils.pdf_helper import list_pdf_files
from main.extractor.pdf_extractor_factory import create_pdf_extractor
from main.logger_config import log_duration
from main.pipeline.file_processor import process_file
from main.utils.s3_helper import download_pdf, hash_file, download_pdf_stream
from main.utils.manifest_helper import load_index_manifest, update_manifest_entry, prune_manifest


logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.path.join("faiss_index", "global.index")
os.makedirs(Config.DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)


def cleanup_if_ephemeral(path: str):
    if path.endswith("::ephemeral"):
        try:
            os.remove(path.replace("::ephemeral", ""))
        except Exception as e:
            logger.warning(f"Failed to delete ephemeral file: {e}")


def cleanup_stale_cache(pruned_keys: list[str]):
    for key in pruned_keys:
        cached_path = os.path.join(Config.CACHE_DIR, key)
        try:
            if os.path.exists(cached_path):
                os.remove(cached_path)
                logger.debug(f"Deleted stale cache file: {cached_path}")
        except Exception as e:
            logger.warning(f"Failed to delete stale cache file {cached_path}: {e}")


def get_keys_to_index(all_keys, manifest, force, cache_mode):
    keys_to_index = []
    for s3_key in all_keys:
        if cache_mode == "none":
            # Always reindex in streaming mode
            keys_to_index.append(s3_key)
            continue

        local_path = download_pdf(s3_key, cache_mode=cache_mode)
        actual_path = local_path.replace("::ephemeral", "")
        current_hash = hash_file(actual_path)
        manifest_entry = manifest.get(s3_key)

        if force or manifest_entry is None or manifest_entry.get("hash") != current_hash:
            keys_to_index.append(s3_key)
        else:
            logger.debug(f"Skipping unchanged file: {s3_key}")
            cleanup_if_ephemeral(local_path)
    return keys_to_index


def index_files(keys_to_index, extractor, cache_mode):
    all_chunks = []
    all_embeddings = []

    max_workers = getattr(Config, "MAX_WORKERS", os.cpu_count()) or 4
    logger.info(f"Indexing {len(keys_to_index)} files with {max_workers} workers.")

    def task(s3_key):
        if cache_mode == "none":
            pdf_bytes = download_pdf_stream(s3_key)
            return process_file(pdf_bytes, extractor)
        else:
            local_path = download_pdf(s3_key, cache_mode=cache_mode)
            actual_path = local_path.replace("::ephemeral", "")
            safe_name = os.path.basename(actual_path).replace(" ", "_").replace("\\", "_").replace("/", "_")
            result = process_file(actual_path, extractor, debug_name=safe_name)
            cleanup_if_ephemeral(local_path)
            return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, s3_key) for s3_key in keys_to_index]
        for s3_key, future in zip(keys_to_index, futures):
            chunks, embeddings = future.result()
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

            if cache_mode != "none":
                local_path = os.path.join(Config.CACHE_DIR, s3_key)
                actual_path = local_path.replace("::ephemeral", "")
                update_manifest_entry(
                    s3_key=s3_key,
                    hash_value=hash_file(actual_path),
                    chunk_count=len(chunks),
                    embedding_count=len(embeddings),
                    size=os.path.getsize(actual_path),
                    source="s3"
                )

    return all_chunks, all_embeddings


def finalize_index(all_chunks, all_embeddings, index_path):
    if not all_chunks or not all_embeddings:
        logger.warning("No data to build FAISS index.")
        return None

    logger.info("Indexed %d chunks, %d embeddings", len(all_chunks), len(all_embeddings))
    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, index_path)
    logger.debug("Global FAISS index saved to: %s", index_path)
    return index


@log_duration("Build Global FAISS Index")
def build_global_index(force: bool = False, cache_mode: str = None, index_path: str = FAISS_INDEX_PATH):
    cache_mode = cache_mode or Config.CACHE_MODE
    all_keys = list_pdf_files()
    if not all_keys:
        logger.warning("No PDF files found.")
        return None

    original_manifest = load_index_manifest()
    manifest, pruned_keys = prune_manifest(all_keys)
    was_pruned = len(pruned_keys) > 0

    cleanup_stale_cache(pruned_keys)

    keys_to_index = get_keys_to_index(all_keys, manifest, force, cache_mode)

    if not keys_to_index and not was_pruned:
        if os.path.exists(index_path):
            logger.info("All files up-to-date. Loading existing index.")
            return faiss_indexer.load_faiss_index(index_path)
        else:
            logger.warning("No new files to index and no existing index found.")
            return None

    if not keys_to_index and was_pruned:
        logger.info("No new files, but manifest was pruned. Rebuilding index to remove stale chunks.")

    extractor = create_pdf_extractor()
    chunks, embeddings = index_files(keys_to_index, extractor, cache_mode)
    return finalize_index(chunks, embeddings, index_path)


def rebuild_index(exclude_keys: list[str] = None, cache_mode: str = None, index_path: str = FAISS_INDEX_PATH):
    cache_mode = cache_mode or Config.CACHE_MODE
    exclude_keys = set(exclude_keys or [])
    all_keys = list_pdf_files()
    keys_to_index = [k for k in all_keys if k not in exclude_keys]

    if not keys_to_index:
        logger.warning("No files left to index after exclusions.")
        return None

    extractor = create_pdf_extractor()
    all_chunks = []
    all_embeddings = []

    max_workers = getattr(Config, "MAX_WORKERS", os.cpu_count()) or 4
    logger.info(f"Rebuilding index with {len(keys_to_index)} files (excluding {len(exclude_keys)}).")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for s3_key in keys_to_index:
            actual_path = download_pdf(s3_key, cache_mode=cache_mode).replace("::ephemeral", "")
            safe_name = os.path.basename(actual_path).replace(" ", "_").replace("\\", "_").replace("/", "_")
            futures.append(executor.submit(process_file, actual_path, extractor, safe_name))

        for s3_key, future in zip(keys_to_index, futures):
            chunks, embeddings = future.result()
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

            local_path = os.path.join(Config.CACHE_DIR, s3_key)
            actual_path = local_path.replace("::ephemeral", "")
            update_manifest_entry(
                s3_key=s3_key,
                hash_value=hash_file(actual_path),
                chunk_count=len(chunks),
                embedding_count=len(embeddings),
                size=os.path.getsize(actual_path),
                source="s3"
            )
            cleanup_if_ephemeral(local_path)

    if not all_chunks or not all_embeddings:
        logger.warning("No data to rebuild FAISS index.")
        return None

    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, index_path)

    return index
