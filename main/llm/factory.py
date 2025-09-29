from main.config import Config

if Config.LLM_PROVIDER == "bedrock":
    from main.llm.bedrock_client import BedrockClient as LLMClient
else:
    from main.llm.ollama_client import OllamaClient as LLMClient

def get_llm_client():
    """Factory function to get the appropriate LLM client based on configuration."""
    return LLMClient()
