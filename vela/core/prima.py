"""
PRIMA — Proactive Response with Initiative and Multi-signal Analysis

Academic grounding:

  [WHEN to intervene]
  Horvitz (1999) Mixed-Initiative Interaction (CHI 1999):
      Intervene only when E[utility(act)] > E[utility(wait)].
      Translated: compute a multi-signal score; fire only above threshold.

  [WHAT type of intervention]
  Liu et al. (2021) ESConv — "Towards Emotional Support Dialog Systems" (ACL 2021):
      Defines 8 empirically validated dialogue strategies grounded in
      Hill's Helping Skills Theory (2009). InitiativeType enum mirrors this taxonomy,
      adapted from emotional support to general proactive assistant dialogue.

  [HOW to select the type]
  Deng et al. (2023) Survey on Proactive Dialogue Systems (IJCAI 2023):
      Classifies proactive dialogue into three modes:
        1. Clarification  (QUESTION, RESTATEMENT)
        2. Target-guided  (INFORMATION → WFC, REFLECTION)
        3. Non-collaborative (REFRAME, SUGGESTION, AFFIRMATION, SELF_DISCLOSURE)
      Signal → type mapping follows this 3-mode structure.

  [Signal proxies]
  Deng, Liao et al. (2023) Prompting LLMs for Proactive Dialogues (EMNLP 2023):
      Shows LLMs are inherently reactive; explicit strategy specification
      in system prompts is required to trigger proactive behaviour.
      Validates our per-type prompt architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vela.core.state import ConversationState

# ── Tunables ──────────────────────────────────────────────────────────────────

INTERVENTION_THRESHOLD = 0.42   # Score must exceed this to intervene
_DEBT_TURNS = 5                 # Consecutive reactive turns that max initiative debt
_DECAY_WORD_DELTA = 5.0         # Word-count drop per turn that maps to decay score 1.0
_SHORT_MSG_WORDS = 5            # Messages shorter than this get a decay boost
_CONFUSION_HITS_MAX = 3         # Confusion markers needed to saturate confusion score

# Signal weights (should sum to 1.0)
_W_STAGNATION = 0.30
_W_ENGAGEMENT = 0.20
_W_CONFUSION  = 0.20
_W_COVERAGE   = 0.15
_W_DEBT       = 0.15


# ── Types ─────────────────────────────────────────────────────────────────────

class InitiativeType(str, Enum):
    """
    8 strategies from ESConv (Liu et al., ACL 2021), adapted for general
    proactive assistant dialogue (original: emotional support counselling).

    ESConv original          → Vela adaptation
    ─────────────────────────────────────────────────────────────
    Question                 → QUESTION        (탐색적 질문)
    Restatement/Paraphrase   → RESTATEMENT     (이해 재확인)
    Reflection of feelings   → REFLECTION      (상황 반영)
    Affirmation/Reassurance  → AFFIRMATION     (확신 강화)
    Providing Suggestions    → SUGGESTION      (구체 제안)
    Providing Information    → INFORMATION     (정보 제공 / WFC 연계)
    [Non-collaborative]      → REFRAME         (문제 재정의, Deng 2023)
    Self-disclosure          → SELF_DISCLOSURE (AI 관점 공유)
    """
    QUESTION        = "QUESTION"        # Clarification mode
    RESTATEMENT     = "RESTATEMENT"     # Clarification mode
    REFLECTION      = "REFLECTION"      # Target-guided mode
    AFFIRMATION     = "AFFIRMATION"     # Non-collaborative mode
    SUGGESTION      = "SUGGESTION"      # Non-collaborative mode
    INFORMATION     = "INFORMATION"     # Target-guided mode (→ WFC)
    REFRAME         = "REFRAME"         # Non-collaborative mode
    SELF_DISCLOSURE = "SELF_DISCLOSURE" # Non-collaborative mode


@dataclass
class Signals:
    stagnation:       float   # 0–1  대화가 맴도는 정도  (state 기반)
    engagement_decay: float   # 0–1  참여도 하락 정도    (메시지 길이 추세)
    confusion:        float   # 0–1  혼란/막힘 정도      (마커 카운트)
    coverage_gap:     float   # 0–1  WFC 미논의 비율     (WFC 상태)
    initiative_debt:  float   # 0–1  에이전트 수동성 누적 (연속 반응 횟수)

    @property
    def score(self) -> float:
        return (
            _W_STAGNATION * self.stagnation
            + _W_ENGAGEMENT * self.engagement_decay
            + _W_CONFUSION  * self.confusion
            + _W_COVERAGE   * self.coverage_gap
            + _W_DEBT       * self.initiative_debt
        )


@dataclass
class InitiativeDecision:
    should_intervene: bool
    initiative_type:  InitiativeType | None
    score:            float
    signals:          Signals


# ── Confusion markers ─────────────────────────────────────────────────────────

_CONFUSION_MARKERS = (
    "이해가 안", "모르겠", "왜", "어떻게", "무슨", "뭔가", "잘 모",
    "헷갈", "복잡", "어렵", "??", "...",
    "what", "how", "why", "unclear", "confused",
)


# ── Engine ────────────────────────────────────────────────────────────────────

class PRIMAEngine:
    """
    Decides *whether* and *how* the agent should take conversational initiative.

    Call `compute()` after each user turn to get an `InitiativeDecision`.
    Call `mark_intervened()` when the agent acts proactively (resets debt).
    """

    def __init__(self) -> None:
        self._consecutive_reactive: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        user_turns:    list[str],
        state:         ConversationState,
        wfc_total:     int,
        wfc_collapsed: int,
    ) -> InitiativeDecision:
        signals = Signals(
            stagnation       = self._stagnation(state),
            engagement_decay = self._engagement_decay(user_turns),
            confusion        = self._confusion(user_turns),
            coverage_gap     = self._coverage_gap(wfc_total, wfc_collapsed),
            initiative_debt  = self._debt(),
        )

        # ── Hard triggers (Deng et al. 2023: non-collaborative mode) ─────────
        # Severe stagnation or high confusion override the soft scoring.
        # Differentiated by severity (STUCK vs LOOPING):
        #   STUCK   (stagnation=1.0) → REFRAME: problem is being framed wrong
        #   LOOPING (stagnation=0.7) → REFLECTION: gently surface the pattern first
        if signals.stagnation >= 0.7:
            self._consecutive_reactive = 0
            i_type = (
                InitiativeType.REFRAME
                if signals.stagnation >= 1.0
                else InitiativeType.REFLECTION
            )
            return InitiativeDecision(
                should_intervene=True,
                initiative_type=i_type,
                score=signals.score,
                signals=signals,
            )
        if signals.confusion >= 0.5:
            self._consecutive_reactive = 0
            return InitiativeDecision(
                should_intervene=True,
                initiative_type=InitiativeType.RESTATEMENT,
                score=signals.score,
                signals=signals,
            )

        # ── Soft scoring (Horvitz 1999: E[utility(act)] > E[utility(wait)]) ─
        score = signals.score
        if score < INTERVENTION_THRESHOLD:
            self._consecutive_reactive += 1
            return InitiativeDecision(
                should_intervene=False,
                initiative_type=None,
                score=score,
                signals=signals,
            )

        initiative_type = self._select_type(signals)
        self._consecutive_reactive = 0
        return InitiativeDecision(
            should_intervene=True,
            initiative_type=initiative_type,
            score=score,
            signals=signals,
        )

    def mark_intervened(self) -> None:
        """Call this when the agent successfully delivers a proactive message."""
        self._consecutive_reactive = 0

    def reset(self) -> None:
        self._consecutive_reactive = 0

    # ── Signal computations (no LLM calls) ───────────────────────────────────

    @staticmethod
    def _stagnation(state: ConversationState) -> float:
        return {
            ConversationState.STUCK:     1.0,
            ConversationState.LOOPING:   0.7,
            ConversationState.DEEPENING: 0.3,
            ConversationState.EXPLORING: 0.0,
        }[state]

    @staticmethod
    def _engagement_decay(turns: list[str]) -> float:
        """
        Measures declining participation by word-count trend.
        Inspired by Murray & Levesque (2003) engagement proxies.
        """
        if len(turns) < 3:
            return 0.0
        lengths = [len(t.split()) for t in turns[-4:]]
        diffs = [lengths[i + 1] - lengths[i] for i in range(len(lengths) - 1)]
        avg_delta = sum(diffs) / len(diffs)
        # avg_delta ≤ -_DECAY_WORD_DELTA  →  score 1.0
        score = max(0.0, min(1.0, -avg_delta / _DECAY_WORD_DELTA))
        if lengths[-1] < _SHORT_MSG_WORDS:
            score = min(1.0, score + 0.3)
        return score

    @staticmethod
    def _confusion(user_turns: list[str]) -> float:
        # Aggregate last 2 turns — confusion often spans multiple short messages
        recent = " ".join(user_turns[-2:]).lower() if user_turns else ""
        hits = sum(1 for m in _CONFUSION_MARKERS if m in recent)
        return min(1.0, hits / _CONFUSION_HITS_MAX)

    @staticmethod
    def _coverage_gap(total: int, collapsed: int) -> float:
        if total == 0:
            return 0.0   # WFC not initialised → no gap signal
        return (total - collapsed) / total

    def _debt(self) -> float:
        """
        Initiative debt: the longer the agent has been purely reactive,
        the stronger the pressure to take initiative (Horvitz 1999 utility model).
        """
        return min(1.0, self._consecutive_reactive / _DEBT_TURNS)

    # ── Type selection ────────────────────────────────────────────────────────

    @staticmethod
    def _select_type(s: Signals) -> InitiativeType:
        """
        Maps signal patterns to ESConv strategy types following Deng et al. (2023)
        three-mode structure:
          1. Clarification   → QUESTION, RESTATEMENT
          2. Target-guided   → INFORMATION, REFLECTION
          3. Non-collaborative → REFRAME, SUGGESTION, AFFIRMATION, SELF_DISCLOSURE
        """
        # Non-collaborative: stuck/looping → reframe the problem entirely
        if s.stagnation >= 0.7:
            return InitiativeType.REFRAME

        # Clarification: user seems confused → restate for shared understanding
        if s.confusion >= 0.4:
            return InitiativeType.RESTATEMENT

        # Target-guided: important topics not yet covered → fill via INFORMATION (→ WFC)
        if s.coverage_gap >= 0.5 and s.stagnation < 0.4:
            return InitiativeType.INFORMATION

        # Non-collaborative (mild): slight stagnation → reflect the situation back
        if s.stagnation >= 0.3:
            return InitiativeType.REFLECTION

        # Non-collaborative: participation dropping → concrete next step
        if s.engagement_decay >= 0.5:
            return InitiativeType.SUGGESTION

        # Non-collaborative: passive + slight decay → reinforce confidence
        if s.engagement_decay >= 0.2 and s.initiative_debt >= 0.6:
            return InitiativeType.AFFIRMATION

        # Clarification: long passive streak → open exploratory question
        if s.initiative_debt >= 0.6:
            return InitiativeType.QUESTION

        # Default: share AI perspective (Self-disclosure)
        return InitiativeType.SELF_DISCLOSURE
