from abc import ABC, abstractmethod

from vela.core.intent_classifier import BaseIntentClassifier
from vela.core.signal_detector import BaseSignalDetector


class DomainPlugin(ABC):
    """새 도메인 추가 시 이것만 구현하면 된다."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """도메인 식별자. 예: 'conversation', 'fandom', 'bizowner'"""
        ...

    @abstractmethod
    def get_signal_detector(self) -> BaseSignalDetector:
        """Layer 01 구현체 반환"""
        ...

    @abstractmethod
    def get_intent_classifier(self) -> BaseIntentClassifier:
        """Layer 02 구현체 반환"""
        ...

    def get_output_prompt(self) -> str:
        """Layer 03 시스템 프롬프트. 기본값 제공, 도메인별 오버라이드 가능."""
        return "사용자가 바로 행동할 수 있는 인사이트를 message, action, reason 구조로 생성하라."
