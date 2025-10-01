import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    SAMPLE_DIR: str = os.getenv("SAMPLE_DIR", "sample_pdfs")
    DEBUG_OUTPUT_DIR: str = os.getenv("DEBUG_OUTPUT_DIR", "debug_chunks")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-text-lite-v1")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
    COHERE_BEDROCK_RERANK_MODEL_ID = os.getenv("COHERE_BEDROCK_RERANK_MODEL_ID", "cohere.rerank-v3-5:0")


    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "none").lower()
