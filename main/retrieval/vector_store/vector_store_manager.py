import logging
from main.config import Config
from main.logger_config import log_duration
from main.retrieval.vector_store import faiss_indexer
from main.retrieval.rerankers.merge_utils import merge_adjacent_chunks

logger = logging.getLogger(__name__)


@log_duration("FAISS Query + Rerank")
def retrieve_relevant_docs(index, query_text, embedding_model, reranker=None, top_k=Config.TOP_K_FAISS, top_n=Config.TOP_N_RERANK, score_threshold=0.2):
    """Retrieve top chunks from FAISS index and optionally rerank, with neighbor merging."""
    logger.debug("Embedding model: %s | Query: %s", type(embedding_model), query_text)
    top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedding_model, k=top_k)
    if not top_chunks:
        return []

    all_chunks = [chunk for chunk, _ in top_chunks if chunk]
    docs = all_chunks.copy()

    if reranker:
        reranked = reranker.rerank(query_text, docs, top_n=top_n)
        merged_docs = merge_adjacent_chunks(reranked, docs, window_size=Config.MERGE_WINDOW_SIZE)
        logger.debug("Merged %d reranked sections for final context.", len(merged_docs))
        return merged_docs

    max_score = max(score for _, score in top_chunks)
    if max_score >= score_threshold:
        merged_docs = merge_adjacent_chunks(top_chunks, docs, window_size=Config.MERGE_WINDOW_SIZE)
        logger.debug("Merged %d sections from FAISS top results for context.", len(merged_docs))
        return merged_docs

    logger.debug("No results exceeded score threshold (%.2f).", score_threshold)
    return []
