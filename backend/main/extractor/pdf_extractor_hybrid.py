import logging
import tempfile
from typing import Union
from .pdf_extractor_base import PDFExtractorBase
from .pdf_extractor_pymupdf import PyMuPDFExtractor
from .pdf_extractor_textract import TextractExtractor
from .chart_ocr_extractor import ChartOCRExtractor

logger = logging.getLogger(__name__)


class HybridPDFExtractor(PDFExtractorBase):
    def __init__(self, region="us-east-1"):
        self.text_extractor = PyMuPDFExtractor()
        self.table_extractor = TextractExtractor(region=region)
        self.chart_ocr_extractor = ChartOCRExtractor()

    def extract_text(self, source: Union[str, bytes]) -> str:
        if isinstance(source, bytes):
            text = self.text_extractor.extract_text(source)
            tables = self.table_extractor.extract_text(source)

            # Fallback: write bytes to temp file for chart OCR
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(source)
                tmp.flush()
                chart_labels = self.chart_ocr_extractor.extract_chart_labels(tmp.name)
        else:
            text = self.text_extractor.extract_text(source)
            tables = self.table_extractor.extract_text(source)
            chart_labels = self.chart_ocr_extractor.extract_chart_labels(source)

        return self._merge_content(text, tables, chart_labels)

    def _merge_content(self, text: str, tables: str, chart_labels: list[str]) -> str:
        sections = [
            "=== Native Text ===\n" + text.strip(),
            "=== Table Data ===\n" + tables.strip(),
            "=== Chart Labels ===\n" + "\n".join(chart_labels)
        ]
        return "\n\n".join([s for s in sections if s])
