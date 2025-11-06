import io
import os
import tempfile
import hashlib
import pytest
from unittest.mock import patch
from main.utils import s3_helper


def test_hash_file_consistency():
    """Test that hash_file returns consistent output for same content."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"sample content")
        tmp.flush()
        path = tmp.name

    try:
        h1 = s3_helper.hash_file(path)
        h2 = s3_helper.hash_file(path)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA-256
    finally:
        os.remove(path)


def test_download_pdf_stream_mocked():
    """Test that download_pdf_stream returns expected bytes."""
    mock_bytes = b"%PDF-1.4 fake content"

    class MockS3Client:
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(mock_bytes)}

    with patch("main.utils.s3_helper.boto3.client", return_value=MockS3Client()):
        result = s3_helper.download_pdf_stream("some/key.pdf")
        assert result == mock_bytes


def test_download_pdf_local_path_resolution(monkeypatch):
    """Test that download_pdf returns correct local path."""
    monkeypatch.setattr("main.utils.s3_helper.Config.USE_S3", False)
    monkeypatch.setattr("main.utils.s3_helper.Config.SAMPLE_DIR", "sample_pdfs")

    result = s3_helper.download_pdf("Circulator_E7_2_E7 2B.pdf")
    expected_path = os.path.join("sample_pdfs", "Circulator_E7_2_E7 2B.pdf")
    assert result.endswith(expected_path)
