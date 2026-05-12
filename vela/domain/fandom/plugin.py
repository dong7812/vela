from vela.core.intent_classifier import BaseIntentClassifier
from vela.core.signal_detector import BaseSignalDetector
from vela.domain.base import DomainPlugin
from vela.domain.fandom.classifier import FanIntentClassifier
from vela.domain.fandom.detector import FanSignalDetector


class FandomDomainPlugin(DomainPlugin):
    """팬덤 도메인 — 첫 번째 실도메인."""

    @property
    def domain_name(self) -> str:
        return "fandom"

    def get_signal_detector(self) -> BaseSignalDetector:
        return FanSignalDetector()

    def get_intent_classifier(self) -> BaseIntentClassifier:
        return FanIntentClassifier()

    def get_output_prompt(self) -> str:
        return (
            "팬덤 맥락에서 사용자가 바로 행동할 수 있는 인사이트를 한두 문장으로 생성하라. "
            "예시: '오늘 밤 12시 컴백이에요. 알림 설정할까요?' / "
            "'오래됐죠? 최근 직캠 올라왔어요.' / '티켓 오픈 30분 전이에요.' 형식으로. "
            "반드시 한국어로."
        )
