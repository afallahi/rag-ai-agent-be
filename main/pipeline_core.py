# from main.vector_store.vector_store_manager import build_global_index, retrieve_relevant_docs
# from main.embedder import embedder
from main.intent_detector import IntentDetector
from main.config import Config
from main.embedder import embedder
from main.retrieval.retrievers.retriever_factory import get_retriever
import logging


logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, force_index: bool = False, retriever_type="faiss"):
        self.intent_detector = IntentDetector()
        self.retriever_type = retriever_type or Config.RETRIEVER_TYPE or "faiss"
        self.embedding_model = embedder.get_model()

        self.top_k_faiss = Config.TOP_K_FAISS
        self.top_n_rerank = Config.TOP_N_RERANK
        self.score_threshold = Config.FAISS_SCORE_THRESHOLD
        
        logger.info(f"Initializing RAGPipeline with retriever: {self.retriever_type.upper()}")
        self.retriever = get_retriever(retriever_type, force=force_index)


    def refresh_index(self):
        if hasattr(self.retriever, "index") and self.retriever.index is not None:
            logger.debug("FAISS index is ready.")
            return self.retriever.index
        else:
            logger.info("No FAISS index found or not applicable for retriever type '%s'.", self.retriever_type)
            return None
 

    def query_knowledge_base(self, query_text, reranker=None, top_k=None):
        top_k = top_k or self.top_k_faiss
        try:
            results = self.retriever.retrieve(
                query_text,
                top_k=top_k,
                embedding_model=self.embedding_model,
                reranker=reranker
            )
            if not results:
                logger.warning("No results retrieved for query: %s", query_text)
            return results
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            return []
