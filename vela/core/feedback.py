"""
FeedbackLogger — PRIMA 개입 결과 자동 수집

수집 데이터:
  intervention: 개입 시점 메타데이터 (전략 타입, 점수, 신호값, 직전 상태)
  outcome:      다음 사용자 턴에서 자동 측정 (메시지 길이 변화, 상태 전이, 명시 평가)

포맷: JSONL (.vela_feedback.jsonl) — 한 줄 = 레코드 1개, append-only
  분석은 session_id + turn 기준으로 intervention↔outcome 조인.

성공 판정 기준 (암묵적):
  - length_ratio > 1.3  : 다음 메시지가 30% 이상 길어짐 → 참여도 증가
    근거: Murray & Levesque (2003) — 응답 길이 증가는 참여도 회복의 지표
  - state_improved      : LOOPING/STUCK → EXPLORING/DEEPENING 상태 전이
  - explicit_rating == 1: 사용자가 👍 클릭
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

FEEDBACK_PATH = Path(__file__).parents[2] / ".vela_feedback.jsonl"

# 암묵적 성공 판정 임계값
# Murray & Levesque (2003): 응답 길이 30% 이상 증가 = 참여도 회복 신호
_LENGTH_RATIO_POSITIVE = 1.3

_GOOD_STATES = {"EXPLORING", "DEEPENING"}
_BAD_STATES  = {"LOOPING", "STUCK"}


class FeedbackLogger:
    def __init__(self, path: Path = FEEDBACK_PATH) -> None:
        self._path = path

    # ── 기록 ─────────────────────────────────────────────────────────────────

    def record_intervention(
        self,
        session_id:      str,
        turn:            int,
        initiative_type: str,
        prima_score:     float,
        state_before:    str,
        msg_len_before:  int,
        signals:         dict,
    ) -> None:
        """PRIMA 개입 발생 시 즉시 호출."""
        self._append({
            "type":            "intervention",
            "session_id":      session_id,
            "turn":            turn,
            "timestamp":       _now(),
            "initiative_type": initiative_type,
            "prima_score":     round(prima_score, 4),
            "state_before":    state_before,
            "msg_len_before":  msg_len_before,
            "signals":         {k: round(v, 4) for k, v in signals.items()},
        })

    def record_outcome(
        self,
        session_id:     str,
        turn:           int,
        msg_len_before: int,
        msg_len_after:  int,
        state_before:   str,
        state_after:    str,
        explicit_rating: Optional[int] = None,  # +1 / -1 / None
    ) -> None:
        """개입 다음 사용자 턴에서 자동 호출."""
        length_ratio   = round(msg_len_after / max(1, msg_len_before), 4)
        state_improved = state_before in _BAD_STATES and state_after in _GOOD_STATES

        self._append({
            "type":            "outcome",
            "session_id":      session_id,
            "turn":            turn,
            "timestamp":       _now(),
            "msg_len_after":   msg_len_after,
            "length_ratio":    length_ratio,
            "state_after":     state_after,
            "state_improved":  state_improved,
            "explicit_rating": explicit_rating,
        })

    def record_explicit_rating(
        self,
        session_id: str,
        turn:       int,
        rating:     int,  # +1 또는 -1
    ) -> None:
        """👍/👎 클릭 시 별도 레코드로 저장 (outcome과 별개로 즉시 반영)."""
        self._append({
            "type":       "rating",
            "session_id": session_id,
            "turn":       turn,
            "timestamp":  _now(),
            "rating":     rating,
        })

    # ── 통계 ─────────────────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, dict]:
        """
        InitiativeType별 누적 통계 반환.
        {
          "REFRAME": {
            "count": 12,
            "outcomes_measured": 10,
            "success_rate": 0.70,       # 긍정 outcome 비율
            "avg_length_ratio": 1.45,   # 평균 메시지 길이 변화율
          }, ...
        }
        """
        if not self._path.exists():
            return {}

        interventions: dict[tuple, dict] = {}
        outcomes:      dict[tuple, dict] = {}
        ratings:       dict[tuple, int]  = {}

        for rec in self._iter_records():
            key = (rec["session_id"], rec["turn"])
            if rec["type"] == "intervention":
                interventions[key] = rec
            elif rec["type"] == "outcome":
                outcomes[key] = rec
            elif rec["type"] == "rating":
                ratings[key] = rec["rating"]

        stats: dict[str, dict] = {}
        for key, inv in interventions.items():
            i_type = inv["initiative_type"]
            if i_type not in stats:
                stats[i_type] = {"count": 0, "positive": 0, "length_ratios": []}
            stats[i_type]["count"] += 1

            out = outcomes.get(key)
            if out:
                explicit = ratings.get(key, out.get("explicit_rating"))
                is_positive = (
                    out["state_improved"]
                    or out["length_ratio"] >= _LENGTH_RATIO_POSITIVE
                    or explicit == 1
                )
                if is_positive:
                    stats[i_type]["positive"] += 1
                stats[i_type]["length_ratios"].append(out["length_ratio"])

        result = {}
        for i_type, s in stats.items():
            n = len(s["length_ratios"])
            result[i_type] = {
                "count":             s["count"],
                "outcomes_measured": n,
                "success_rate":      round(s["positive"] / n, 2) if n else None,
                "avg_length_ratio":  round(sum(s["length_ratios"]) / n, 2) if n else None,
            }
        return result

    def total_interventions(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for r in self._iter_records() if r.get("type") == "intervention")

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _iter_records(self):
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
