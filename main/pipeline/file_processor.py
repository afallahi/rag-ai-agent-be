import logging
from main.config import Config
from main.embedder import embedder
from main.chunker import text_chunker
from main.utils.pdf_helper import save_debug_outputs



logger = logging.getLogger(__name__)


def process_file(file: str, extractor) -> tuple[list[str], list[list[float]]]:
    try:
        logger.debug("Processing: %s", file)
        text = extractor.extract_text(file)
        if not text.strip():
            logger.warning("No text extracted from %s", file)
            return [], []
        
        from main.utils.text_preprocessor import preprocess_text

        cleaned_text = preprocess_text(text)
        chunks = text_chunker.chunk_text(cleaned_text)
        if not chunks:
            logger.warning("No chunks created for %s", file)
            return [], []

        embeddings = embedder.embed_text_chunks(chunks)
        if not embeddings:
            logger.warning("No embeddings created for %s", file)
            return [], []

        logger.debug("Created %d chunks and %d embeddings from %s", len(chunks), len(embeddings), file)

        if Config.DEBUG:
            save_debug_outputs(file, chunks, embeddings)

        return chunks, embeddings

    except Exception as e:
        logger.error(f"Failed to process {file}: {e}")
        return [], []

