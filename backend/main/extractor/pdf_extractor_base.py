from abc import ABC, abstractmethod
from typing import Union

class PDFExtractorBase(ABC):
    @abstractmethod
    def extract_text(self, source: Union[str, bytes]) -> str:
        """Extract text from a file path or raw PDF bytes"""
        pass
