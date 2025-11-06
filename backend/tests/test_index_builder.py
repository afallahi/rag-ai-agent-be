import pytest
from unittest.mock import patch, MagicMock
from main.retrieval.vector_store.index_builder import build_global_index


@pytest.mark.parametrize("cache_mode", ["none", "ephemeral", "full"])
def test_build_global_index_runs(cache_mode):
    """Test that build_global_index executes without error across cache modes."""
    with patch("main.retrieval.vector_store.index_builder.list_pdf_files", return_value=["doc1.pdf", "doc2.pdf"]), \
        patch("main.retrieval.vector_store.index_builder.download_pdf", return_value="doc1.pdf::ephemeral"), \
        patch("main.retrieval.vector_store.index_builder.download_pdf_stream", return_value=b"%PDF-1.4 fake bytes"), \
        patch("main.retrieval.vector_store.index_builder.hash_file", return_value="abc123"), \
        patch("main.retrieval.vector_store.index_builder.load_index_manifest", return_value={}), \
        patch("main.retrieval.vector_store.index_builder.update_manifest_entry"), \
        patch("main.retrieval.vector_store.index_builder.cleanup_if_ephemeral"), \
        patch("main.retrieval.vector_store.index_builder.create_pdf_extractor") as mock_extractor_factory, \
        patch("main.retrieval.vector_store.index_builder.process_file", return_value=(["chunk1"], [[0.1]*384])), \
        patch("main.retrieval.vector_store.index_builder.faiss_indexer.build_faiss_index") as mock_build, \
        patch("main.retrieval.vector_store.index_builder.faiss_indexer.save_faiss_index"), \
        patch("main.retrieval.vector_store.index_builder.os.path.getsize", return_value=12345), \
        patch("main.retrieval.vector_store.index_builder.os.path.exists", return_value=True):

        mock_extractor_factory.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        index = build_global_index(force=True, cache_mode=cache_mode)
        assert index is not None


def test_build_global_index_no_files():
    """Test that build_global_index returns None when no files are found."""
    with patch("main.retrieval.vector_store.index_builder.list_pdf_files", return_value=[]):
        index = build_global_index()
        assert index is None


def test_build_global_index_skips_unchanged(monkeypatch):
    """Test that unchanged files are skipped when force=False and no index exists."""
    manifest = {"doc1.pdf": {"hash": "abc123"}}
    monkeypatch.setattr("main.retrieval.vector_store.index_builder.list_pdf_files", lambda: ["doc1.pdf"])
    monkeypatch.setattr("main.retrieval.vector_store.index_builder.download_pdf", lambda s3_key, cache_mode: "doc1.pdf::ephemeral")
    monkeypatch.setattr("main.retrieval.vector_store.index_builder.hash_file", lambda path: "abc123")
    monkeypatch.setattr("main.retrieval.vector_store.index_builder.load_index_manifest", lambda: manifest)
    monkeypatch.setattr("main.retrieval.vector_store.index_builder.cleanup_if_ephemeral", lambda path: None)
    monkeypatch.setattr("main.retrieval.vector_store.index_builder.os.path.exists", lambda path: False)

    index = build_global_index(force=False)
    assert index is None
