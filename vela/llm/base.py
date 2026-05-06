from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        ...

    @abstractmethod
    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
