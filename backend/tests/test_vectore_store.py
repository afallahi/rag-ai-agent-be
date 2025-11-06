"""Test cases for FaissStore functionality."""

import os
import numpy as np
import tempfile
import shutil
import pytest
from main.retrieval.vector_store.faiss_indexer import FaissStore


@pytest.fixture
def temp_faiss_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_faiss_store_add_search_save_load(temp_faiss_dir):
    dim = 384
    store = FaissStore(dim)

    embeddings = np.random.rand(3, dim).astype("float32")
    documents = ["doc1", "doc2", "doc3"]

    store.add(embeddings, documents)
    assert store.index.ntotal == 3
    assert len(store.metadata) == 3

    # Search for first embedding
    results = store.search(embeddings[0], k=2)
    assert len(results) > 0
    assert any(doc == "doc1" for doc, _ in results)

    # Save and reload
    index_file = os.path.join(temp_faiss_dir, "test.index")
    metadata_file = os.path.join(temp_faiss_dir, "test_metadata.npy")
    store.save(index_file, metadata_file)

    assert os.path.exists(index_file)
    assert os.path.exists(metadata_file)

    new_store = FaissStore(dim)
    new_store.load(index_file, metadata_file)

    assert new_store.index.ntotal == 3
    assert len(new_store.metadata) == 3

    loaded_results = new_store.search(embeddings[1], k=2)
    assert len(loaded_results) > 0
    assert set(doc for doc, _ in loaded_results).issubset(set(documents))


def test_faiss_store_empty_search():
    store = FaissStore(dim=384)
    query = np.random.rand(384).astype("float32")
    results = store.search(query, k=3)
    assert results == []


def test_faiss_store_duplicate_adds():
    store = FaissStore(dim=384)
    emb = np.random.rand(1, 384).astype("float32")
    store.add(emb, ["doc1"])
    store.add(emb, ["doc1-again"])

    assert store.index.ntotal == 2
    assert store.metadata == ["doc1", "doc1-again"]
