from main.config import Config
from main.intent_detector.ollama_intent_detector import OllamaIntentDetector
from main.intent_detector.bedrock_intent_detector import BedrockIntentDetector


def create_intent_detector():
    provider = Config.LLM_PROVIDER.lower()
    if provider == "bedrock":
        return BedrockIntentDetector()
    elif provider == "ollama":
        return OllamaIntentDetector()
    else:
        raise ValueError(f"Unsupported intent detector provider: {provider}")
