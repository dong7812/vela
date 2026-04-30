from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
