from vela.core.intent_classifier import BaseIntentClassifier, IntentResult
from vela.core.signal_detector import SignalResult


class FanIntentClassifier(BaseIntentClassifier):
    """
    팬 유형 Layer 02 의도 분류. 조건 분기 우선, 복합 케이스만 LLM.
    """

    def classify(self, signal: SignalResult) -> IntentResult:
        raw = signal.raw

        # 단순 케이스 — LLM 없음
        if raw.get("comeback_imminent"):
            return IntentResult(
                type="OPPORTUNITY",
                intent="NEED_ACTION",
                urgency=0.95,
                confidence=0.95,
                reason="컴백 임박",
            )

        if raw.get("activity_deviation", 0) > 0.7 and raw.get("days_inactive", 0) >= 3:
            return IntentResult(
                type="WARNING",
                intent="NEED_WARNING",
                urgency=0.85,
                confidence=0.90,
                reason="활동 급감 + 이탈 위험",
            )

        if raw.get("purchase_signal", 0) > 0.6:
            return IntentResult(
                type="OPPORTUNITY",
                intent="NEED_ACTION",
                urgency=0.80,
                confidence=0.85,
                reason="구매 의향 높음",
            )

        if raw.get("is_new_fan"):
            return IntentResult(
                type="INFO",
                intent="NEED_INFO",
                urgency=0.50,
                confidence=0.80,
                reason="신규 팬 입문 가이드",
            )

        # 복합 케이스 — LLM 연동 추후 구현
        # 이유: 신규 팬 + 구매 의향 높음 + 활동 감소 동시 발생 → 조건 분기로 커버 불가
        return IntentResult(
            type="INFO",
            intent="NEED_INFO",
            urgency=signal.score,
            confidence=0.60,
            reason="복합 신호 감지",
        )
