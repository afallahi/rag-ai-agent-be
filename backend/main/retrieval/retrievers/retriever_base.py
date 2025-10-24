from abc import ABC, abstractmethod

class RetrieverBase(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query_text: str, top_k: int = 5):
        """Return a list of relevant document chunks given a query."""
        pass
