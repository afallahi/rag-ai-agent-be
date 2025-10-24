from abc import ABC, abstractmethod

class LLMBase(ABC):
    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        pass
