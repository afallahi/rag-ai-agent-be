import os
import pytest
from unittest.mock import patch
from tests.test_constants import SAMPLE_PDF_PATH
from main.extractor.pdf_extractor_factory import create_pdf_extractor
from main.config import Config

@pytest.mark.parametrize("provider", ["pymupdf", "hybrid"])
def test_extract_text_from_path(provider, monkeypatch):
    """Test PDF text extraction from file path for supported local providers."""
    assert os.path.exists(SAMPLE_PDF_PATH), f"{SAMPLE_PDF_PATH} does not exist."

    monkeypatch.setattr("main.extractor.pdf_extractor_textract.Config.USE_S3", False)
    monkeypatch.setattr(
        "main.extractor.pdf_extractor_textract.TextractExtractor.extract_text",
        lambda self, source: "Mocked Textract Output"
    )

    extractor = create_pdf_extractor({"provider": provider, "region": Config.AWS_REGION})
    text = extractor.extract_text(SAMPLE_PDF_PATH)

    assert isinstance(text, str)
    assert "Mocked" in text or len(text.strip()) > 0


@patch("main.extractor.pdf_extractor_textract.boto3.client")
def test_extract_text_textract_mocked(mock_boto_client):
    """Test Textract extractor with mocked AWS client."""
    mock_textract = mock_boto_client.return_value
    mock_textract.start_document_analysis.return_value = {
        "JobId": "fake-job-id"
    }
    mock_textract.get_document_analysis.return_value = {
        "JobStatus": "SUCCEEDED",
        "Blocks": [
            {"BlockType": "LINE", "Text": "Mocked chart label"},
            {"BlockType": "LINE", "Text": "Mocked axis label"}
        ]
    }

    extractor = create_pdf_extractor({"provider": "textract", "region": Config.AWS_REGION})
    text = extractor.extract_text(SAMPLE_PDF_PATH)
    assert "Mocked chart label" in text
    assert "Mocked axis label" in text
