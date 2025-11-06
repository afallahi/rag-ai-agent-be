import fitz
from typing import Union
from .pdf_extractor_base import PDFExtractorBase

class PyMuPDFExtractor(PDFExtractorBase):
    def extract_text(self, source: Union[str, bytes]) -> str:
        text = []
        try:
            if isinstance(source, bytes):
                doc = fitz.open(stream=source, filetype="pdf")
            else:
                doc = fitz.open(source)

            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            raise RuntimeError(f"PyMuPDF failed to extract text: {e}") from e
