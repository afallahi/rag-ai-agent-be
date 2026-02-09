import datetime
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from main.retrieval.vector_store.index_builder import build_global_index
from main.pipeline_core import RAGPipeline, generate_response, get_reranker, get_llm
from main.config import Config

router = APIRouter()
logger = logging.getLogger(__name__)
rag = RAGPipeline()


# Request schema
class QueryRequest(BaseModel):
    query: str
    history: list[list[str]] = []

# Routes

@router.post("/query")
async def query_endpoint(payload: QueryRequest):
    try:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        cfg = Config.get_all()
        config_llm = cfg['llm_provider']
        logger.info(f"Received query: {payload.query}, LLM={config_llm}")
        llm = get_llm(config_llm)
        reranker = get_reranker(cfg["rerank_provider"])
        response = generate_response(rag, payload.query, llm, payload.history, reranker)
        return {"results": [response], "timestamp": datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        logger.exception("Error during query processing")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/refresh-index")
def refresh_index():
    try:
        logger.info("Received request to refresh FAISS index.")
        index = build_global_index(force=False)
        if index is None:
            logger.warning("Index refresh completed: no changes detected.")
            return {"status": "no_update", "message": "Index is already up-to-date."}
        logger.info("Index refresh completed successfully.")
        return {"status": "updated", "message": "Index rebuilt successfully."}
    except Exception as e:
        logger.exception("Error during index refresh")
        raise HTTPException(status_code=500, detail=f"Index refresh failed: {str(e)}")
