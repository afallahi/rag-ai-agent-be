from main.config import Config

def get_llm_client(provider: str | None = None):
    """Factory function to get the appropriate LLM client based on configuration."""
    provider = provider or Config.LLM_PROVIDER
    if provider == "bedrock":
        from main.llm.bedrock_client import BedrockClient as LLMClient
    else:
        from main.llm.ollama_client import OllamaClient as LLMClient

    return LLMClient()
