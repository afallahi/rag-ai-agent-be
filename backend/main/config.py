import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    SAMPLE_DIR: str = os.getenv("SAMPLE_DIR", "sample_pdfs")
    DEBUG_OUTPUT_DIR: str = os.getenv("DEBUG_OUTPUT_DIR", "debug_chunks")
    CACHE_MODE = os.getenv("CACHE_MODE", "full").lower()

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-text-lite-v1")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
    COHERE_BEDROCK_RERANK_MODEL_ID = os.getenv("COHERE_BEDROCK_RERANK_MODEL_ID", "cohere.rerank-v3-5:0")


    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "none").lower()

    PDF_EXTRACTOR_PROVIDER = os.getenv("PDF_EXTRACTOR_PROVIDER", "pymupdf").lower()
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    
    USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
    S3_BUCKET = os.getenv("S3_BUCKET", "blcp-rag-pdf-files")
    S3_PREFIX = os.getenv("S3_PREFIX", "")
    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")



    TOP_K_FAISS = int(os.getenv("TOP_K_FAISS", "40"))
    TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "10"))
    FAISS_SCORE_THRESHOLD = float(os.getenv("FAISS_SCORE_THRESHOLD", "0.2"))

    RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "faiss")
    BEDROCK_KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")

    MERGE_WINDOW_SIZE = int(os.getenv("MERGE_WINDOW_SIZE", "1"))
    PROXIMITY_MERGE = os.getenv("PROXIMITY_MERGE", "false").lower() == "true"


    _overrides = {}

    OPTIONS = {
        "llm_provider": ["ollama", "bedrock"],
        "ollama_model": ["mistral", "gemma", "llama2"],
        "bedrock_model_id": [
            "amazon.titan-text-lite-v1",
            "anthropic.claude-v2",
            "ai21.j2-mid"
        ],
        "rerank_provider": ["none", "cohere-direct", "cohere-bedrock"],
        "retriever_type": ["faiss", "bedrock"],
        "pdf_extractor_provider": ["pymupdf", "aws-textract", "hybrid"],
        "cache_mode": ["full", "partial", "none"],
        "embedding_model": [
            "all-MiniLM-L6-v2",
            "multi-qa-MiniLM-L6-cos-v1",
            "text-embedding-ada-002"
        ],
        # For FAISS retriever tuning
        "top_k_faiss": list(range(5, 101, 5)),  # 5,10,...100
        "top_n_rerank": list(range(1, 21)),     # 1–20
        "faiss_score_threshold": [round(x * 0.1, 2) for x in range(0, 11)]  # 0.0–1.0
    }


    @classmethod
    def get(cls, key: str):
        """Return current value (override if set, else default)."""
        return cls._overrides.get(key, getattr(cls, key))
    
    
    @classmethod
    def get_all(cls):
        allowed = (str, int, float, bool, type(None), list, dict)
        base = {
            k.lower(): v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and isinstance(v, allowed)
        }
        base.update(cls._overrides)
        return base


    @classmethod
    def get_options(cls):
        return cls.OPTIONS


    @classmethod
    def update(cls, updates: dict):
        # normalize incoming keys to lowercase
        normalized = {k.lower(): v for k, v in updates.items()}
        for k, v in normalized.items():
            if k in cls.OPTIONS and v not in cls.OPTIONS[k]:
                raise ValueError(f"Invalid value {v} for {k}")
        cls._overrides.update(normalized)
        return cls.get_all()
    

    @classmethod
    def reset(cls):
        """Clear overrides."""
        cls._overrides.clear()
        return cls.get_all()