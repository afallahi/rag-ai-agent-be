from .pdf_extractor_base import PDFExtractorBase
from .pdf_extractor_pymupdf import PyMuPDFExtractor
from .pdf_extractor_textract import TextractExtractor
from .pdf_extractor_hybrid import HybridPDFExtractor
from main.config import Config


def create_pdf_extractor(config: dict | None = None) -> PDFExtractorBase:
    """Factory to create PDF extractor instance from config dict or default Config."""
    if config is None:
        config = {}

    provider = config.get("provider", Config.PDF_EXTRACTOR_PROVIDER).lower()

    if provider == "pymupdf":
        return PyMuPDFExtractor()
    elif provider == "textract":
        return TextractExtractor(region=config.get("region", Config.AWS_REGION))
    elif provider == "hybrid":
        return HybridPDFExtractor()
    else:
        raise ValueError(f"Unsupported PDF extractor provider: {provider}")
