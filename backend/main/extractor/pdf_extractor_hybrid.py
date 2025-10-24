import logging
import os
from .pdf_extractor_base import PDFExtractorBase
from .pdf_extractor_pymupdf import PyMuPDFExtractor
from .pdf_extractor_textract import TextractExtractor
from .chart_ocr_extractor import ChartOCRExtractor
from main.config import Config
from main.utils.s3_helper import download_pdf


logger = logging.getLogger(__name__)


class HybridPDFExtractor(PDFExtractorBase):
    def __init__(self, region="us-east-1"):
        self.text_extractor = PyMuPDFExtractor()
        self.table_extractor = TextractExtractor(region=region)
        self.chart_ocr_extractor = ChartOCRExtractor()

    def extract_text(self, file_path_or_key: str) -> str:
        if Config.USE_S3:
            s3_key = file_path_or_key
            local_path = download_pdf(s3_key)
        else:
            local_path = file_path_or_key
            s3_key = None
        
        text = self.text_extractor.extract_text(local_path)
        tables = self.table_extractor.extract_text(s3_key or local_path)
        chart_labels = self.chart_ocr_extractor.extract_chart_labels(local_path)

        return self._merge_content(text, tables, chart_labels)


    def _merge_content(self, text: str, tables: str, chart_labels: list[str]) -> str:
        sections = [
            "=== Native Text ===\n" + text.strip(),
            "=== Table Data ===\n" + tables.strip(),
            "=== Chart Labels ===\n" + "\n".join(chart_labels)
        ]
        return "\n\n".join([s for s in sections if s])
    
