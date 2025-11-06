import pytest
from main.extractor.pdf_extractor_factory import create_pdf_extractor
from main.extractor.pdf_extractor_pymupdf import PyMuPDFExtractor
from main.extractor.pdf_extractor_textract import TextractExtractor
from main.extractor.pdf_extractor_hybrid import HybridPDFExtractor


def test_create_pymupdf_extractor():
    config = {"provider": "pymupdf"}
    extractor = create_pdf_extractor(config)
    assert isinstance(extractor, PyMuPDFExtractor)


def test_create_textract_extractor():
    config = {"provider": "textract", "region": "us-east-1"}
    extractor = create_pdf_extractor(config)
    assert isinstance(extractor, TextractExtractor)


def test_create_hybrid_extractor():
    config = {"provider": "hybrid", "region": "us-west-2"}
    extractor = create_pdf_extractor(config)
    assert isinstance(extractor, HybridPDFExtractor)


def test_create_invalid_provider():
    config = {"provider": "unknown"}
    with pytest.raises(ValueError, match="Unsupported PDF extractor provider"):
        create_pdf_extractor(config)
