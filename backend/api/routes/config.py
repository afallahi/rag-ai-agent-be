from fastapi import APIRouter
from pydantic import BaseModel
from main.config import Config as conf

router = APIRouter()

class ConfigUpdate(BaseModel):
    llm_provider: str | None = None
    ollama_model: str | None = None
    bedrock_model_id: str | None = None
    chunk_size: int | None = None
    rerank_provider: str | None = None
    top_k_faiss: int | None = None

@router.get("/config")
def get_config():
    return conf.get_all()


@router.post("/config")
def update_config(update: ConfigUpdate):
    updates = {k.upper(): v for k, v in update.dict().items() if v is not None}
    return conf.update(updates)


@router.post("/config/reset")
def reset_config():
    return conf.reset()


@router.get("/config/options")
def get_config_options():
    return conf.get_options()
