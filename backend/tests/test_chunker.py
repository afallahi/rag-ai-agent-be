"""Test suite for the text chunking functionality."""
import pytest
from main.chunker.text_chunker import chunk_text


def test_chunk_text_basic_overlap():
    """Test that long text is chunked with expected overlap and size constraints."""
    long_text = (
        "This is paragraph one. " * 10 +
        "This is paragraph two. " * 10 +
        "This is paragraph three. " * 10
    )

    chunk_size = 500
    chunk_overlap = 50

    chunks = chunk_text(long_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 1, "Expected multiple chunks for long input."
    assert all(len(chunk) <= chunk_size for chunk in chunks), "Chunk exceeds max size."

    # Check overlap between first and second chunk
    if len(chunks) > 1:
        first_tail = chunks[0][-chunk_overlap:]
        second_head = chunks[1][:chunk_overlap * 2]
        shared_words = set(first_tail.split()) & set(second_head.split())
        assert len(shared_words) > 3, f"Low overlap: {shared_words}"


def test_chunk_text_empty_input():
    """Test that empty input returns no chunks."""
    chunks = chunk_text("")
    assert chunks == []


def test_chunk_text_short_input():
    """Test that short input returns a single chunk."""
    short_text = "This is a short paragraph."
    chunks = chunk_text(short_text, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0] == short_text


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(100, 20), (300, 50), (1000, 100)])
def test_chunk_text_varied_sizes(chunk_size, chunk_overlap):
    """Test chunking behavior across different size and overlap settings."""
    text = "Lorem ipsum dolor sit amet. " * 100
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert all(len(chunk) <= chunk_size for chunk in chunks)
    assert len(chunks) > 1
