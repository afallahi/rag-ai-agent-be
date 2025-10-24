import os
import json
import logging
from main.config import Config


logger = logging.getLogger(__name__)

INDEX_MANIFEST_PATH = os.path.join(Config.CACHE_DIR, "index_manifest.json")



def load_index_manifest() -> set[str]:
    if os.path.exists(INDEX_MANIFEST_PATH):
        try:
            with open(INDEX_MANIFEST_PATH, "r") as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load index manifest: {e}")
    return set()


def save_index_manifest(indexed_files: set[str]):
    try:
        with open(INDEX_MANIFEST_PATH, "w") as f:
            json.dump(sorted(indexed_files), f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save index manifest: {e}")


def remove_from_manifest(path: str, files_to_remove: list[str]):
    if not os.path.exists(path):
        return

    try:
        with open(path, "r") as f:
            current = set(json.load(f))
        updated = current - set(files_to_remove)
        with open(path, "w") as f:
            json.dump(sorted(updated), f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to update manifest: {e}")

