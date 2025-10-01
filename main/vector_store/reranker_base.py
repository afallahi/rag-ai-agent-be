from abc import ABC, abstractmethod
from typing import List

class RerankerBase(ABC):
    """Abstract base class for rerankers."""

    provider: str = "base"

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[str]:
        """Given a query and candidate documents, return top_n reranked documents."""
        pass
