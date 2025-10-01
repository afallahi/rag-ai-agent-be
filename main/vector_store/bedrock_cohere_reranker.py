import boto3
import json
import logging
from typing import List, Tuple
from .reranker_base import RerankerBase
from main.config import Config

logger = logging.getLogger(__name__)


class BedrockCohereReranker(RerankerBase):
    def __init__(self, model_id: str = Config.COHERE_BEDROCK_RERANK_MODEL_ID, region: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.provider = "bedrock-cohere"

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[str]:

        payload = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_version": 2
            }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(payload),
            )
            result = json.loads(response["body"].read())

            logger.debug("=== Bedrock Cohere rerank raw results ===")
            for i, r in enumerate(result.get("results", [])):
                logger.debug(f"Rank {i+1}: {r}")


            reranked_docs: List[Tuple[str, float]] = []
            for r in result.get("results", []):
                idx = r.get("index")
                score = r.get("relevance_score", 0.0)
                if idx is not None and 0 <= idx < len(documents):
                    reranked_docs.append((documents[idx], score))

            # Sort by score descending
            reranked_docs.sort(key=lambda x: x[1], reverse=True)
            return reranked_docs
        except Exception as e:
            logger.exception("[BedrockCohereReranker] Unexpected error: %s", e)
            return []