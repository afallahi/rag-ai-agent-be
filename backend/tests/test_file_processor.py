import pytest
from main.pipeline.file_processor import process_file
from main.extractor.pdf_extractor_pymupdf import PyMuPDFExtractor

from tests.test_constants import SAMPLE_PDF_PATH


def test_process_file_with_path():
    extractor = PyMuPDFExtractor()
    chunks, embeddings = process_file(SAMPLE_PDF_PATH, extractor)

    assert isinstance(chunks, list)
    assert isinstance(embeddings, list)
    assert len(chunks) == len(embeddings)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(isinstance(vec, list) for vec in embeddings)


def test_process_file_with_bytes():
    extractor = PyMuPDFExtractor()
    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    chunks, embeddings = process_file(pdf_bytes, extractor)

    assert isinstance(chunks, list)
    assert isinstance(embeddings, list)
    assert len(chunks) == len(embeddings)


def test_process_file_empty_text(monkeypatch):
    class DummyExtractor:
        def extract_text(self, source):
            return ""

    chunks, embeddings = process_file("irrelevant.pdf", DummyExtractor())
    assert chunks == []
    assert embeddings == []


def test_process_file_exception(monkeypatch):
    class FailingExtractor:
        def extract_text(self, source):
            raise RuntimeError("Extraction failed")

    chunks, embeddings = process_file("irrelevant.pdf", FailingExtractor())
    assert chunks == []
    assert embeddings == []
