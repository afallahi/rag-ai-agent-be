from abc import ABC, abstractmethod

class PDFExtractorBase(ABC):
    """Abstract base class for PDF extractors."""

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass
