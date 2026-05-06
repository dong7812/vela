from typing import Iterator

import anthropic

from vela.llm.base import BaseLLM

_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-7",
]
DEFAULT_MODEL = _MODELS[0]


class ClaudeLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        trimmed = list(messages)
        while trimmed and trimmed[-1]["role"] == "assistant":
            trimmed.pop()
        return trimmed or [{"role": "user", "content": "계속해주세요."}]

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system or "You are a helpful assistant.",
            messages=self._trim(messages),
        )
        return response.content[0].text

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=system or "You are a helpful assistant.",
            messages=self._trim(messages),
        ) as stream:
            yield from stream.text_stream

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
