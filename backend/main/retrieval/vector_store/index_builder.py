import os
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from main.config import Config
from main.retrieval.vector_store import faiss_indexer
from main.utils.pdf_helper import list_pdf_files
from main.extractor.pdf_extractor_factory import create_pdf_extractor
from main.logger_config import log_duration
from main.pipeline.file_processor import process_file
from main.utils.manifest_helper import load_index_manifest, save_index_manifest

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.path.join("faiss_index", "global.index")
INDEX_MANIFEST_PATH = os.path.join(Config.CACHE_DIR, "index_manifest.json")
os.makedirs(Config.DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)


@log_duration("Build Global FAISS Index")
def build_global_index(force: bool = False, index_path: str = FAISS_INDEX_PATH):
    all_files = list_pdf_files()
    if not all_files:
        logger.warning("No PDF files found.")
        return None

    if force:
        files_to_index = all_files
        logger.info("Force mode enabled — reindexing all files.")
    else:
        already_indexed = load_index_manifest()
        files_to_index = [f for f in all_files if f not in already_indexed]
        if not files_to_index:
            if os.path.exists(index_path):
                logger.info("All files already indexed. Loading existing index.")
                return faiss_indexer.load_faiss_index(index_path)
            else:
                logger.warning("No new files to index and no existing index found.")
                return None

    extractor = create_pdf_extractor()
    all_chunks = []
    all_embeddings = []

    max_workers = getattr(Config, "MAX_WORKERS", os.cpu_count()) or 4
    logger.info(f"Using {max_workers} parallel workers for indexing.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file, extractor) for file in files_to_index]
        for future in tqdm(futures, desc="Indexing PDFs"):
            chunks, embeddings = future.result()
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

    if not all_chunks or not all_embeddings:
        logger.warning("No data to build FAISS index.")
        return None

    logger.info("Indexed %d files, %d chunks, %d embeddings", len(files_to_index), len(all_chunks), len(all_embeddings))

    logger.debug("Building global FAISS index...")
    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, index_path)
    logger.debug("Global FAISS index saved to: %s", index_path)

    if not force:
        updated_manifest = load_index_manifest().union(set(files_to_index))
        save_index_manifest(updated_manifest)

    return index


def rebuild_index(exclude_files: list[str] = None, index_path: str = FAISS_INDEX_PATH):
    exclude_files = set(exclude_files or [])
    all_files = list_pdf_files()
    files_to_index = [f for f in all_files if f not in exclude_files]

    if not files_to_index:
        logger.warning("No files left to index after exclusions.")
        return None

    extractor = create_pdf_extractor()
    all_chunks = []
    all_embeddings = []

    max_workers = getattr(Config, "MAX_WORKERS", os.cpu_count()) or 4
    logger.info(f"Rebuilding index with {len(files_to_index)} files (excluding {len(exclude_files)}).")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file, extractor) for file in files_to_index]
        for future in tqdm(futures, desc="Rebuilding FAISS index"):
            chunks, embeddings = future.result()
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

    if not all_chunks or not all_embeddings:
        logger.warning("No data to rebuild FAISS index.")
        return None

    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, index_path)

    # Update manifest
    from main.utils.manifest_helper import save_index_manifest
    save_index_manifest(set(files_to_index))

    return index
