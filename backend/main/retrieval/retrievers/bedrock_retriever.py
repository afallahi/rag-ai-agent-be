import boto3
import logging
from main.config import Config
from main.retrieval.retrievers.retriever_base import RetrieverBase

logger = logging.getLogger(__name__)

class BedrockRetriever(RetrieverBase):
    def __init__(self):
        self.client = boto3.client("bedrock-agent-runtime", region_name=Config.AWS_REGION)
        self.kb_id = Config.BEDROCK_KNOWLEDGE_BASE_ID

    def retrieve(self, query_text: str, top_k: int = 5):
        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query_text},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
            )
            results = response.get("retrievalResults", [])
            logger.debug("Retrieved %d docs from Bedrock KB", len(results))
            return [r["content"]["text"] for r in results if "content" in r]
        except Exception as e:
            logger.error("Bedrock retrieval failed: %s", e)
            return []
