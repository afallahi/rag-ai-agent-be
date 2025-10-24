import requests
from main.config import Config
from main.llm.base import LLMBase

class OllamaClient(LLMBase):
    def __init__(self, model: str = Config.OLLAMA_MODEL, url: str = Config.OLLAMA_URL):
        self.provider = "ollama"
        self.model = model
        self.url = url.rstrip("/")

    def is_running(self) -> bool:
        try:
            response = requests.get(f"{self.url}/tags")
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    def generate_answer(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(f"{self.url}/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            print(f"[ERROR] Failed to call Ollama: {e}")
            return "LLM error: could not generate response"
