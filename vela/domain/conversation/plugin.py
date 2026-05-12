from typing import Any

from vela.core.embedder import Embedder
from vela.core.intent_classifier import BaseIntentClassifier, IntentResult
from vela.core.prima import INTERVENTION_THRESHOLD, InitiativeDecision, InitiativeType, PRIMAEngine
from vela.core.signal_detector import BaseSignalDetector, SignalResult
from vela.core.state import ConversationState, StateDetector
from vela.domain.base import DomainPlugin


class PrimaSignalDetector(BaseSignalDetector):
    """기존 PRIMA score → SignalResult 변환. should_act = score >= 0.38."""

    THRESHOLD = INTERVENTION_THRESHOLD  # 0.38

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._state_detector = StateDetector(self._embedder)
        self._prima = PRIMAEngine()

    def compute_baseline(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        # PRIMA는 내부 상태 기반 — 별도 베이스라인 불필요
        return {}

    def detect(self, current: dict[str, Any], baseline: dict[str, Any]) -> SignalResult:
        """
        current 필드:
          user_turns    list[str]  최근 사용자 발화 목록
          wfc_total     int        WFC 전체 셀 수
          wfc_collapsed int        WFC 논의 완료 셀 수
        """
        user_turns: list[str] = current.get("user_turns", [])
        wfc_total: int = current.get("wfc_total", 0)
        wfc_collapsed: int = current.get("wfc_collapsed", 0)

        state: ConversationState = self._state_detector.detect(user_turns)
        decision: InitiativeDecision = self._prima.compute(
            user_turns=user_turns,
            state=state,
            wfc_total=wfc_total,
            wfc_collapsed=wfc_collapsed,
        )

        return SignalResult(
            score=decision.score,
            should_act=decision.should_intervene,
            trigger=f"{state.value} (score={decision.score:.2f})",
            raw={"state": state, "decision": decision},
        )

    def mark_intervened(self) -> None:
        self._prima.mark_intervened()

    def reset(self) -> None:
        self._prima.reset()


class WfcIntentClassifier(BaseIntentClassifier):
    """기존 WFC + ESConv 8가지 전략 선택을 IntentResult로 래핑."""

    def classify(self, signal: SignalResult) -> IntentResult:
        decision: InitiativeDecision | None = signal.raw.get("decision")

        if decision is None or not decision.initiative_type:
            return IntentResult(
                type="INFO",
                intent="NEED_INFO",
                urgency=signal.score,
                confidence=0.5,
                reason=signal.trigger,
            )

        i_type: InitiativeType = decision.initiative_type

        if i_type == InitiativeType.REFRAME:
            return IntentResult(type="WARNING", intent="NEED_WARNING", urgency=0.90, confidence=0.90, reason=signal.trigger)
        if i_type == InitiativeType.REFLECTION:
            return IntentResult(type="WARNING", intent="NEED_WARNING", urgency=0.70, confidence=0.80, reason=signal.trigger)
        if i_type in (InitiativeType.SUGGESTION, InitiativeType.AFFIRMATION):
            return IntentResult(type="OPPORTUNITY", intent="NEED_ACTION", urgency=0.70, confidence=0.80, reason=signal.trigger)
        if i_type == InitiativeType.INFORMATION:
            return IntentResult(type="OPPORTUNITY", intent="NEED_ACTION", urgency=0.60, confidence=0.80, reason=signal.trigger)
        if i_type in (InitiativeType.QUESTION, InitiativeType.RESTATEMENT):
            return IntentResult(type="INFO", intent="NEED_INFO", urgency=0.50, confidence=0.70, reason=signal.trigger)
        # SELF_DISCLOSURE
        return IntentResult(type="INFO", intent="NEED_INFO", urgency=0.40, confidence=0.60, reason=signal.trigger)


class ConversationDomainPlugin(DomainPlugin):
    """기존 Vela 대화 도메인 래퍼. 하위 호환성 유지."""

    def __init__(self) -> None:
        self._detector = PrimaSignalDetector()
        self._classifier = WfcIntentClassifier()

    @property
    def domain_name(self) -> str:
        return "conversation"

    def get_signal_detector(self) -> BaseSignalDetector:
        return self._detector

    def get_intent_classifier(self) -> BaseIntentClassifier:
        return self._classifier

    def get_output_prompt(self) -> str:
        return (
            "사용자와 대화하는 능동적 파트너로서 적절한 전략으로 개입하세요. "
            "반드시 한국어로 답변하세요."
        )
