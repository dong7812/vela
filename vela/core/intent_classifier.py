from abc import ABC, abstractmethod
from dataclasses import dataclass

from vela.core.signal_detector import SignalResult


@dataclass
class IntentResult:
    type: str           # "WARNING" | "OPPORTUNITY" | "INFO"
    intent: str         # "NEED_ACTION" | "NEED_INFO" | "NEED_WARNING"
    urgency: float      # 0.0 ~ 1.0
    confidence: float   # 낮으면 Layer 03에서 보수적으로 처리
    reason: str


class BaseIntentClassifier(ABC):
    @abstractmethod
    def classify(self, signal: SignalResult) -> IntentResult:
        """단순 케이스: 조건 분기로 처리 (LLM 없음). 복합 케이스만 LLM 호출."""
        ...
