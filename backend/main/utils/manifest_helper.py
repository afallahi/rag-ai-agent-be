import os
import json
import logging
from datetime import datetime
from main.config import Config

logger = logging.getLogger(__name__)

INDEX_MANIFEST_PATH = os.path.join(Config.CACHE_DIR, "index_manifest.json")


def load_index_manifest() -> dict[str, dict]:
    """Load manifest as a dict of s3_key → metadata."""
    if os.path.exists(INDEX_MANIFEST_PATH):
        try:
            with open(INDEX_MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index manifest: {e}")
    return {}


def save_index_manifest(manifest: dict[str, dict]):
    """Save manifest with structured metadata."""
    try:
        with open(INDEX_MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save index manifest: {e}")


def update_manifest_entry(
    s3_key: str,
    hash_value: str,
    chunk_count: int,
    embedding_count: int,
    size: int | None = None,
    source: str = "s3"
):
    manifest = load_index_manifest()
    manifest[s3_key] = {
        "hash": hash_value,
        "indexed_at": datetime.utcnow().isoformat(),
        "chunk_count": chunk_count,
        "embedding_count": embedding_count,
        "size": size,
        "source": source
    }
    save_index_manifest(manifest)


def remove_from_manifest(files_to_remove: list[str]):
    """Remove entries by s3_key."""
    manifest = load_index_manifest()
    for key in files_to_remove:
        manifest.pop(key, None)
    save_index_manifest(manifest)


def prune_manifest(current_keys: list[str]) -> dict[str, dict]:
    """
    Remove manifest entries for files no longer present in S3.
    Returns the updated manifest.
    """
    manifest = load_index_manifest()
    stale_keys = [k for k in manifest if k not in current_keys]
    if stale_keys:
        logger.info(f"Pruning {len(stale_keys)} stale manifest entries.")
        for k in stale_keys:
            manifest.pop(k, None)
        save_index_manifest(manifest)
    return manifest, stale_keys
