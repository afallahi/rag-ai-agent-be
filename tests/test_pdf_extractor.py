"""Test suite for PDF text extraction functionality."""
import os
import pytest
from main.extractor.pdf_extractor_factory import create_pdf_extractor
from main.config import Config


@pytest.mark.parametrize("provider", ["pymupdf", "textract"])
def test_extract_text_from_pdf(provider):
    """Test PDF text extraction for multiple providers."""
    sample_pdf = "sample_pdfs/DE-Fire-Pump.pdf"
    assert os.path.exists(sample_pdf), f"{sample_pdf} does not exist."

    extractor = create_pdf_extractor({"provider": provider, "region": Config.AWS_REGION})
    text = extractor.extract_text(sample_pdf)
    
    assert isinstance(text, str)
    assert len(text) > 10
