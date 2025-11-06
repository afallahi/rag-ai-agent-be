import os
import tempfile
import json
import pytest
from main.utils.manifest_helper import load_index_manifest, update_manifest_entry


def test_load_manifest_creates_file_if_missing(monkeypatch):
    """Test that load_index_manifest creates an empty manifest if file is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "index_manifest.json")
        assert not os.path.exists(manifest_path)

        monkeypatch.setattr("main.utils.manifest_helper.INDEX_MANIFEST_PATH", manifest_path)

        manifest = load_index_manifest()
        assert isinstance(manifest, dict)
        assert manifest == {}


def test_update_manifest_entry_writes_correct_data(monkeypatch):
    """Test that update_manifest_entry correctly writes entry to manifest file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "index_manifest.json")
        monkeypatch.setattr("main.utils.manifest_helper.INDEX_MANIFEST_PATH", manifest_path)

        file_key = "doc1.pdf"
        update_manifest_entry(
            s3_key=file_key,
            hash_value="abc123",
            chunk_count=5,
            embedding_count=1,
            size=1234,
            source="s3"
        )

        with open(manifest_path, "r") as f:
            data = json.load(f)

        assert file_key in data
        assert data[file_key]["hash"] == "abc123"
        assert data[file_key]["chunk_count"] == 5
        assert data[file_key]["embedding_count"] == 1
        assert data[file_key]["size"] == 1234
        assert data[file_key]["source"] == "s3"


def test_update_manifest_entry_overwrites_existing(monkeypatch):
    """Test that update_manifest_entry overwrites existing entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "index_manifest.json")
        monkeypatch.setattr("main.utils.manifest_helper.INDEX_MANIFEST_PATH", manifest_path)

        file_key = "doc1.pdf"

        update_manifest_entry(file_key, "abc123", 3, 1, 1000, "s3")
        update_manifest_entry(file_key, "def456", 4, 2, 2000, "s3")

        manifest = load_index_manifest()
        assert manifest[file_key]["hash"] == "def456"
        assert manifest[file_key]["chunk_count"] == 4
        assert manifest[file_key]["embedding_count"] == 2
        assert manifest[file_key]["size"] == 2000
