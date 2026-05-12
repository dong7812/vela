from typing import Any

from vela.core.signal_detector import BaseSignalDetector, SignalResult


class FanSignalDetector(BaseSignalDetector):
    """
    팬 행동 데이터 기반 Layer 01 신호 감지.

    score = 0.40 * activity_deviation
          + 0.30 * content_skip_rate
          + 0.20 * session_time_shift
          + 0.10 * purchase_signal

    THRESHOLD = 0.50 근거: 단일 신호 오탐 방지. 복합 신호 2개 이상 기준.
    """

    THRESHOLD = 0.50

    def compute_baseline(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if len(history) < 14:  # 최소 2주치 데이터
            return {}

        avg_activity = sum(h.get("activity_count", 0) for h in history) / len(history)
        avg_session_hour = sum(h.get("session_hour", 12) for h in history) / len(history)

        return {
            "avg_activity": avg_activity,
            "avg_session_hour": avg_session_hour,
        }

    def detect(self, current: dict[str, Any], baseline: dict[str, Any]) -> SignalResult:
        if not baseline:
            return SignalResult(
                score=0.0,
                should_act=False,
                trigger="insufficient_baseline",
                raw=current,
            )

        avg_activity = baseline.get("avg_activity", 1)
        current_activity = current.get("activity_count", 0)
        activity_deviation = min(1.0, abs(current_activity - avg_activity) / max(1, avg_activity))

        content_skip_rate: float = current.get("content_skip_rate", 0.0)

        avg_hour = baseline.get("avg_session_hour", 12)
        current_hour = current.get("session_hour", 12)
        session_time_shift = min(1.0, abs(current_hour - avg_hour) / 12)

        purchase_signal: float = current.get("purchase_signal", 0.0)

        score = (
            0.40 * activity_deviation
            + 0.30 * content_skip_rate
            + 0.20 * session_time_shift
            + 0.10 * purchase_signal
        )

        raw = {
            "activity_deviation": round(activity_deviation, 4),
            "content_skip_rate": content_skip_rate,
            "session_time_shift": round(session_time_shift, 4),
            "purchase_signal": purchase_signal,
            "days_inactive": current.get("days_inactive", 0),
            "comeback_imminent": current.get("comeback_imminent", False),
            "is_new_fan": current.get("is_new_fan", False),
        }

        return SignalResult(
            score=round(score, 4),
            should_act=score >= self.THRESHOLD,
            trigger=f"fan_signal (score={score:.2f})",
            raw=raw,
        )
