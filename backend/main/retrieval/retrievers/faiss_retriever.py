import logging
from main.retrieval.vector_store import index_builder as index_builder
from main.retrieval.vector_store import vector_store_manager as index_manager
from main.retrieval.retrievers.retriever_base import RetrieverBase
from main.config import Config


logger = logging.getLogger(__name__)

class FAISSRetriever(RetrieverBase):
    def __init__(self, force=False):
        self.index = index_builder.build_global_index(force=force)

    def retrieve(self, query_text: str, top_k: int = Config.TOP_K_FAISS, embedding_model=None, reranker=None):
        return index_manager.retrieve_relevant_docs(
            self.index, query_text, embedding_model, reranker, top_k=top_k
        )
