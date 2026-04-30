import requests

from vela.llm.base import BaseLLM

_BASE_URL = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "qwen2.5:3b") -> None:
        self.model = model

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        resp = requests.post(
            f"{_BASE_URL}/api/chat",
            json={"model": self.model, "messages": payload, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            f"{_BASE_URL}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{_BASE_URL}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
