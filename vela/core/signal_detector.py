from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalResult:
    score: float        # 0.0 ~ 1.0, 편차 점수
    should_act: bool    # threshold 초과 여부
    trigger: str        # 감지된 신호 설명 (로깅용)
    raw: dict = field(default_factory=dict)  # 원본 신호값 (Layer 02 입력)


class BaseSignalDetector(ABC):
    THRESHOLD: float = 0.50  # 도메인별 오버라이드 가능

    @abstractmethod
    def compute_baseline(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """과거 데이터로 베이스라인 계산. 최소 2주치 데이터 필요."""
        ...

    @abstractmethod
    def detect(self, current: dict[str, Any], baseline: dict[str, Any]) -> SignalResult:
        """현재 데이터와 베이스라인 편차 점수화. LLM 호출 절대 금지."""
        ...
