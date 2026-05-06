import json
from typing import Iterator

import requests

from vela.llm.base import BaseLLM

_BASE_URL = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "qwen2.5:3b") -> None:
        self.model = model

    def _build_payload(self, messages: list[dict[str, str]], system: str) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)
        return payload

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        resp = requests.post(
            f"{_BASE_URL}/api/chat",
            json={"model": self.model, "messages": self._build_payload(messages, system), "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        with requests.post(
            f"{_BASE_URL}/api/chat",
            json={"model": self.model, "messages": self._build_payload(messages, system), "stream": True},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{_BASE_URL}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
