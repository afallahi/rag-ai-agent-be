import os
import logging
from main.config import Config
from main.embedder import embedder
from main.retrieval.vector_store import faiss_indexer
from main.chunker import text_chunker
from main.utils.pdf_helper import list_pdf_files, save_debug_outputs
from main.utils.s3_helper import download_pdf
from main.extractor.pdf_extractor_factory import create_pdf_extractor
from main.extractor.pdf_extractor_textract import TextractExtractor
from main.logger_config import log_duration


logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.path.join("faiss_index", "global.index")
os.makedirs(Config.DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)


@log_duration("Build Global FAISS Index")
def build_global_index(force: bool = False):
    """Extract, chunk, and embed all PDFs and return a global FAISS index."""
    if os.path.exists(FAISS_INDEX_PATH) and not force:
        logger.info("Global index already exists. Skipping reprocessing.")
        return faiss_indexer.load_faiss_index(FAISS_INDEX_PATH)

    all_chunks = []
    all_embeddings = []

    pdf_files = list_pdf_files()
    if not pdf_files:
        logger.warning("No PDF files found.")
        return None

    extractor = create_pdf_extractor()

    for file in pdf_files:
        if Config.USE_S3:
            if isinstance(extractor, TextractExtractor):
                file_path = file  # Textract uses S3 key directly
            else:
                file_path = download_pdf(file)
        else:
            file_path = os.path.join(Config.SAMPLE_DIR, file)

        logger.debug("Processing: %s", file)

        text = extractor.extract_text(file_path)
        if not text.strip():
            logger.warning("No text extracted from %s", file)
            continue

        chunks = text_chunker.chunk_text(text)
        if not chunks:
            logger.warning("No chunks created for %s", file)
            continue

        logger.debug("Created %d chunks from %s", len(chunks), file)

        embeddings = embedder.embed_text_chunks(chunks)
        if not embeddings:
            logger.warning("No embeddings created for %s", file)
            continue

        logger.debug("Generated %d embeddings from %s", len(embeddings), file)

        if Config.DEBUG:
            save_debug_outputs(file, chunks, embeddings)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    if not all_chunks or not all_embeddings:
        logger.warning("No data to build global FAISS index.")
        return None

    logger.debug("Building global FAISS index...")
    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, FAISS_INDEX_PATH)
    logger.debug("Global FAISS index saved to: %s", FAISS_INDEX_PATH)

    return index


@log_duration("FAISS Query + Rerank")
def retrieve_relevant_docs(index, query_text, embedding_model, reranker=None, top_k=10, top_n=4, score_threshold=0.2):
    """Retrieve top chunks from FAISS index and optionally rerank."""
    logger.debug("Embedding model: %s | Query: %s", type(embedding_model), query_text)
    top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedding_model, k=top_k)
    if not top_chunks:
        return []

    docs = [chunk for chunk, _ in top_chunks if chunk]

    if reranker:
        reranked = reranker.rerank(query_text, docs, top_n=top_n)
        return [doc for doc, _ in reranked if doc]

    max_score = max(score for _, score in top_chunks)
    if max_score >= score_threshold:
        return docs

    return []
