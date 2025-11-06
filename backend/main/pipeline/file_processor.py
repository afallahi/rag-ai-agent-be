import logging
import os
from typing import Union
from main.config import Config
from main.embedder import embedder
from main.chunker import text_chunker
from main.utils.pdf_helper import save_debug_outputs
from main.utils.text_preprocessor import preprocess_text

logger = logging.getLogger(__name__)


def process_file(source: Union[str, bytes], extractor, debug_name: str = None) -> tuple[list[str], list[list[float]]]:
    """
    Extracts text from a PDF (file path or raw bytes), preprocesses it, chunks it, and embeds the chunks.
    Returns: (chunks, embeddings)
    """
    try:
        logger.debug("Processing: %s", source if isinstance(source, str) else "<in-memory bytes>")
        text = extractor.extract_text(source)

        if not text.strip():
            logger.warning("No text extracted from %s", source if isinstance(source, str) else "<bytes>")
            return [], []

        cleaned_text = preprocess_text(text)
        chunks = text_chunker.chunk_text(cleaned_text)

        if not chunks:
            logger.warning("No chunks created for %s", source if isinstance(source, str) else "<bytes>")
            return [], []

        embeddings = embedder.embed_text_chunks(chunks)

        if not embeddings:
            logger.warning("No embeddings created for %s", source if isinstance(source, str) else "<bytes>")
            return [], []

        logger.debug("Created %d chunks and %d embeddings from %s", len(chunks), len(embeddings), source if isinstance(source, str) else "<bytes>")

        if Config.DEBUG and isinstance(source, str):
            debug_filename = debug_name or os.path.basename(source).replace(" ", "_").replace("\\", "_").replace("/", "_")
            save_debug_outputs(debug_filename, chunks, embeddings)

        return chunks, embeddings

    except Exception as e:
        logger.error(f"Failed to process {source if isinstance(source, str) else '<bytes>'}: {e}")
        return [], []
