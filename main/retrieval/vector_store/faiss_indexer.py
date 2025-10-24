import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Using IndexFlatIP for cosine similarity (normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, embeddings: np.ndarray, documents: list[str]):
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {embeddings.shape[1]}")

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(documents)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)
        results = []

        for dist_list, idx_list in zip(distances, indices):
            for dist, idx in zip(dist_list, idx_list):
                if idx == -1:
                    continue
                results.append((self.metadata[idx], float(dist)))

        return results

    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        np.save(metadata_path, np.array(self.metadata, dtype=object))

    def load(self, index_path: str, metadata_path: str):
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found.")

        self.index = faiss.read_index(index_path)
        self.metadata = np.load(metadata_path, allow_pickle=True).tolist()


def build_faiss_index(embeddings: list[list[float]], documents: list[str]) -> FaissStore:
    array = np.array(embeddings).astype("float32")
    dim = array.shape[1]
    store = FaissStore(dim)
    store.add(array, documents)
    return store


def save_faiss_index(store: FaissStore, index_path: str):
    base_path = os.path.splitext(index_path)[0]
    store.save(base_path + ".index", base_path + ".metadata.npy")


def load_faiss_index(index_path: str) -> FaissStore:
    base_path = os.path.splitext(index_path)[0]
    index = faiss.read_index(base_path + ".index")
    dim = index.d  # dim comes from embedding model. For all-MiniLM-L6-v2, it's 384.
    store = FaissStore(dim)
    store.index = index
    store.metadata = np.load(base_path + ".metadata.npy", allow_pickle=True).tolist()
    return store

def query_faiss_index(store: FaissStore, query_text: str, model: SentenceTransformer, k: int = 5) -> list[tuple[str, float]]:
    from main.utils.normalize_tokens import normalize_text  # local import to avoid circular dependency
    show_progress = os.getenv("DEBUG", "false").lower() == "true"
    normalized_query = normalize_text(query_text)
    query_embedding = model.encode([normalized_query], convert_to_numpy=True, show_progress_bar=show_progress)
    results = store.search(query_embedding, k)
    return results  # returns list of (chunk_text, score)
