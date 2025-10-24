import fitz
from .pdf_extractor_base import PDFExtractorBase

class PyMuPDFExtractor(PDFExtractorBase):
    """Extract text using PyMuPDF."""

    def extract_text(self, file_path: str) -> str:
        text = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            raise RuntimeError(f"PyMuPDF failed to extract text: {e}") from e
