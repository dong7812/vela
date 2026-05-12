from dataclasses import dataclass
from typing import Iterator

from vela.core.intent_classifier import IntentResult
from vela.core.signal_detector import SignalResult
from vela.llm.base import BaseLLM

_DEFAULT_SYSTEM = "사용자가 바로 행동할 수 있는 인사이트를 생성하라."


@dataclass
class OutputResult:
    message: str
    action: str | None
    reason: str


class OutputGenerator:
    """Layer 03 — LLM으로 최종 출력 생성."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def _prompt(self, signal: SignalResult, intent: IntentResult) -> str:
        return (
            f"신호: {signal.trigger} (score={signal.score:.2f})\n"
            f"분류: {intent.type} / {intent.intent} "
            f"(긴급도={intent.urgency:.2f}, 신뢰도={intent.confidence:.2f})\n"
            f"이유: {intent.reason}\n\n"
            "위 신호를 바탕으로 사용자에게 전달할 인사이트를 한두 문장으로 생성하세요."
        )

    def generate(
        self,
        signal: SignalResult,
        intent: IntentResult,
        system: str = _DEFAULT_SYSTEM,
    ) -> OutputResult:
        response = self._llm.chat(
            [{"role": "user", "content": self._prompt(signal, intent)}],
            system=system,
        )
        return OutputResult(message=response, action=None, reason=intent.reason)

    def generate_stream(
        self,
        signal: SignalResult,
        intent: IntentResult,
        system: str = _DEFAULT_SYSTEM,
    ) -> Iterator[str]:
        yield from self._llm.chat_stream(
            [{"role": "user", "content": self._prompt(signal, intent)}],
            system=system,
        )
