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

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        # Claude requires the last message to be from the user
        trimmed = list(messages)
        while trimmed and trimmed[-1]["role"] == "assistant":
            trimmed.pop()
        if not trimmed:
            trimmed = [{"role": "user", "content": "계속해주세요."}]

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system or "You are a helpful assistant.",
            messages=trimmed,
        )
        return response.content[0].text

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
