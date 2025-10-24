"""Embedding Generator Module"""
from main.config import Config
from typing import List
from sentence_transformers import SentenceTransformer
from main.utils.normalize_tokens import normalize_text

# Load the model once (cached)
_model = SentenceTransformer(Config.EMBEDDING_MODEL)


def embed_text_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks.

    Args:
        chunks (List[str]): List of text strings.

    Returns:
        List[List[float]]: Corresponding list of embeddings.
    """
    if not chunks:
        return []
    
    normalized_chunks = [normalize_text(c) for c in chunks]
    return _model.encode(normalized_chunks, convert_to_numpy=True, show_progress_bar=Config.DEBUG).tolist()


def get_model() -> SentenceTransformer:
    """
    Expose the internal model (used for query embedding).

    Returns:
        SentenceTransformer: Preloaded embedding model.
    """
    return _model
