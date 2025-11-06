import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import for CORS
from pydantic import BaseModel
from main.pipeline_core import RAGPipeline, generate_response, get_reranker, get_llm
from main.retrieval.vector_store.index_builder import build_global_index


app = FastAPI()
rag = RAGPipeline()
llm = get_llm()
reranker = get_reranker()

logger = logging.getLogger(__name__)

history = []

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies/authorization headers
    allow_methods=["*"],            # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],            # Allow all headers
)

# Request schema
class QueryRequest(BaseModel):
    query: str
    history: list[list[str]] = []

# Routes
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "1.0.0"}


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    try:
        logger.info(f"Received query: {payload.query}")
        response = generate_response(rag, payload.query, llm, payload.history, reranker)
        return {"results": [response]}
    except Exception as e:
        logger.exception("Error during query processing")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/refresh-index")
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
