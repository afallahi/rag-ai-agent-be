import os
import logging
from main.config import Config
from main.utils import s3_helper

logger = logging.getLogger(__name__)


def list_pdf_files() -> list[str]:
    """Return a list of PDF files from local folder or S3 based on config."""
    if Config.USE_S3:
        return s3_helper.list_pdfs_in_bucket()
    else:
        return [
            f for f in os.listdir(Config.SAMPLE_DIR)
            if f.lower().endswith(".pdf")
        ]
    

def save_debug_outputs(filename: str, chunks: list[str], embeddings: list[list[float]]):
    """Save chunks and embeddings to debug files."""
    # Save chunks
    debug_path = os.path.join(Config.DEBUG_OUTPUT_DIR, f"{filename}.md")
    with open(debug_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n--- Chunk {i} ---\n{chunk}\n")
    logger.debug("Chunks saved to: %s", debug_path)

    # Save embeddings
    debug_embed_path = os.path.join(Config.DEBUG_OUTPUT_DIR, f"{filename}.embeddings.txt")
    with open(debug_embed_path, "w", encoding="utf-8") as f:
        for i, emb in enumerate(embeddings, start=1):
            f.write(f"Embedding {i}: {emb}\n")
    logger.debug("Embeddings saved to: %s", debug_embed_path)
