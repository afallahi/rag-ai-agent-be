import pytest
import io
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock
from main.extractor.chart_ocr_extractor import ChartOCRExtractor
from tests.test_constants import SAMPLE_PDF_PATH

def test_chart_ocr_from_bytes(monkeypatch):
    """Test chart OCR from in-memory PDF bytes with mocked Tesseract and fitz."""
    dummy_image = Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255)

    # Patch Tesseract
    monkeypatch.setattr("main.extractor.chart_ocr_extractor.pytesseract.image_to_string", lambda img: "Mocked Label")

    # Patch fitz.open to return a dummy document
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_images.return_value = [(42,)]
    mock_doc.__iter__.return_value = [mock_page]

    buffer = io.BytesIO()
    dummy_image.save(buffer, format="PNG")
    mock_doc.extract_image.return_value = {"image": buffer.getvalue()}

    monkeypatch.setattr("main.extractor.chart_ocr_extractor.fitz.open", lambda *args, **kwargs: mock_doc)

    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    extractor = ChartOCRExtractor()
    labels = extractor.extract_chart_labels(pdf_bytes)

    assert isinstance(labels, list)
    assert any("Mocked Label" in label for label in labels)


def test_chart_ocr_from_bytes(monkeypatch):
    """Test chart OCR from in-memory PDF bytes with mocked Tesseract and fitz."""
    dummy_image = Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255)

    # Patch Tesseract
    monkeypatch.setattr("main.extractor.chart_ocr_extractor.pytesseract.image_to_string", lambda img: "Mocked Label")

    # Patch fitz.open to return a dummy document
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_images.return_value = [(42,)]
    mock_doc.__iter__.return_value = [mock_page]

    buffer = io.BytesIO()
    dummy_image.save(buffer, format="PNG")
    mock_doc.extract_image.return_value = {"image": buffer.getvalue()}

    monkeypatch.setattr("main.extractor.chart_ocr_extractor.fitz.open", lambda *args, **kwargs: mock_doc)

    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    extractor = ChartOCRExtractor()
    labels = extractor.extract_chart_labels(pdf_bytes)

    assert isinstance(labels, list)
    assert any("Mocked Label" in label for label in labels)


def test_chart_ocr_empty(monkeypatch):
    """Test chart OCR returns empty list when no images or OCR fails."""
    monkeypatch.setattr("main.extractor.chart_ocr_extractor.pytesseract.image_to_string", lambda img: "")
    extractor = ChartOCRExtractor()
    labels = extractor.extract_chart_labels(SAMPLE_PDF_PATH)

    assert isinstance(labels, list)
    assert labels == [] or all(label.strip() == "" for label in labels)
