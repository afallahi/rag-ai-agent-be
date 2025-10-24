from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from main.config import Config
from main.intent_detector.intent_detector_base import IntentDetectorBase


class OllamaIntentDetector(IntentDetectorBase):
    def __init__(self):
        self.model = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            streaming=True
        )
        self.intents = [
            "greeting", "thanks", "goodbye", "help",
            "chitchat", "question", "unclear"
        ]
        self.prompt_template = PromptTemplate.from_template(
            "Classify the following user message into one of these intents: {intents}.\n\n"
            "User Message: {text}\nIntent:"
        )

    def detect(self, text: str) -> str:
        prompt = self.prompt_template.format(
            text=text.strip(),
            intents=", ".join(self.intents)
        )
        response_chunks = self.model.stream(prompt)
        full_response = "".join(chunk.content for chunk in response_chunks)
        intent = full_response.strip().lower()
        if intent not in self.intents:
            intent = "unclear"
        return intent
