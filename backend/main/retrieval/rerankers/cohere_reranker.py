import cohere
from typing import List, Tuple
from .reranker_base import RerankerBase

class CohereReranker(RerankerBase):
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
        self.provider = "cohere"

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        results = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked_docs: List[Tuple[str, float]] = []
        for r in results.results:
            idx = r.index
            score = r.relevance_score
            doc_text = documents[idx]  # retrieve original text
            reranked_docs.append((doc_text, score))

        return reranked_docs
