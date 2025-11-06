"""Test suite for embedding generation from text chunks."""

import pytest
from main.embedder import embedder


def test_embed_text_chunks_basic():
    """Test that embeddings are generated correctly for a list of text chunks."""
    sample_chunks = [
        "This is the first test chunk.",
        "Another chunk of text to embed.",
        "Yet another meaningful text segment."
    ]

    embeddings = embedder.embed_text_chunks(sample_chunks)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_chunks)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(dim, float) for vec in embeddings for dim in vec)

    dim_lengths = {len(vec) for vec in embeddings}
    assert len(dim_lengths) == 1, f"Inconsistent embedding dimensions: {dim_lengths}"


def test_embed_text_chunks_empty_input():
    """Test that empty input returns an empty list."""
    embeddings = embedder.embed_text_chunks([])
    assert embeddings == []


@pytest.mark.parametrize("text", [
    "Short sentence.",
    "A longer paragraph that still fits within a single chunk and should embed cleanly.",
    " ".join(["word"] * 1000),  # long input
])
def test_embed_text_chunks_varied_lengths(text):
    """Test embedding generation for varied chunk lengths."""
    embeddings = embedder.embed_text_chunks([text])
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert all(isinstance(dim, float) for dim in embeddings[0])
