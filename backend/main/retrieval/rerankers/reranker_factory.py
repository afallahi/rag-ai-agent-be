# reranker_factory.py
import os
from .cohere_reranker import CohereReranker
from .bedrock_cohere_reranker import BedrockCohereReranker
from .reranker_base import RerankerBase
from main.config import Config


def create_reranker(config: dict) -> RerankerBase | None:
    """Factory to create reranker instance from config dict."""
    if not config or not config.get("provider"):
        return None

    provider = config["provider"]

    if provider == "cohere":
        return CohereReranker(
            api_key=config.get("api_key") or os.getenv("COHERE_API_KEY"),
            model=config.get("model", "rerank-english-v3.0"),
        )

    elif provider == "bedrock-cohere":
        return BedrockCohereReranker(
            model_id=config.get("model_id",  Config.COHERE_BEDROCK_RERANK_MODEL_ID),
            region=config.get("region", "us-east-1"),
        )

    else:
        raise ValueError(f"Unsupported reranker provider: {provider}")
