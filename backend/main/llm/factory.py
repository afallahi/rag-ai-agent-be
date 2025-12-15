from main.llm.bedrock_client import BedrockClient
from main.llm.ollama_client import OllamaClient


LLM_CLIENTS = {
    "bedrock": BedrockClient,
    "ollama": OllamaClient,
}

def get_llm_client(provider: str | None = None):
    """Factory function to get the appropriate LLM client based on configuration."""
    client_cls = LLM_CLIENTS.get(provider)
    if not client_cls:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return client_cls()
