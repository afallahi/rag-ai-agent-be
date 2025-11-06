import fitz
import io
import logging
from typing import Union
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class ChartOCRExtractor:
    """Extract embedded text from chart images using PyMuPDF + Tesseract."""

    def extract_chart_labels(self, source: Union[str, bytes]) -> list[str]:
        labels = []
        try:
            if isinstance(source, bytes):
                doc = fitz.open(stream=source, filetype="pdf")
            else:
                doc = fitz.open(source)

            for page_num, page in enumerate(doc, start=1):
                images = page.get_images(full=True)
                for img_index, img_info in enumerate(images):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    ocr_text = pytesseract.image_to_string(image)
                    cleaned = ocr_text.strip()
                    if cleaned:
                        labels.append(f"[Page {page_num} Image {img_index}] {cleaned}")
        except Exception as e:
            logger.warning(f"[ChartOCRExtractor] Failed to extract chart labels: {e}")

        return labels
