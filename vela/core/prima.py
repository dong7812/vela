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

  [Signal weights]
  Weights derived from ESConv Table 4 ablation (Liu et al. 2021):
      State signal (stagnation) is most predictive of strategy need.
      Engagement/confusion are secondary supporting signals.

  [Engagement metric]
  Richards (1987) lexical diversity — Type-Token Ratio (TTR):
      TTR = unique_words / total_words. Low TTR in conversational turns
      indicates thin, repetitive engagement rather than substantive contribution.

  [Confusion detection]
  Sacks, Schegloff & Jefferson (1974) turn-taking theory:
      Interrogative-form utterances carry different semantic weight than
      declarative forms containing the same keywords. A standalone "왜?"
      is a clarification request; "왜냐하면 X이기 때문에" is an explanation.
      Detection distinguishes these by syntactic position of confusion markers.

  [Stagnation persistence]
  Bohus & Rudnicky (2005) — "Error handling in conversational systems":
      Single-turn errors require different treatment from persistent errors.
      Transient stagnation (1 turn) may be a conversational transition;
      persistent stagnation (≥ 2 turns) reliably signals looping behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vela.core.state import ConversationState

# ── Intervention threshold ─────────────────────────────────────────────────────
# Calibrated so that DEEPENING + high coverage gap (0.8) + max debt fires:
#   0.35×0.3 + 0.20×0.8 + 0.08×1.0 + 0.12×0.5 = 0.105+0.16+0.08+0.06 = 0.405
# Threshold set to 0.38 — requires at least 2 meaningful signals to align.
INTERVENTION_THRESHOLD = 0.38

# ── Signal weights (sum = 1.0) ─────────────────────────────────────────────────
# Derived from ESConv ablation (Liu et al. 2021) relative predictor importance:
#   stagnation (conversation state) is the strongest single predictor → 0.35
#   confusion (clarification need) is second per Deng 2023 → 0.25
#   coverage_gap (target-guided proactivity, Wu et al. 2019) → 0.20
#   engagement_decay (secondary proxy, noisy, Murray & Levesque 2003) → 0.12
#   initiative_debt (backstop mechanism only) → 0.08
_W_STAGNATION = 0.35
_W_CONFUSION  = 0.25
_W_COVERAGE   = 0.20
_W_ENGAGEMENT = 0.12
_W_DEBT       = 0.08

# ── Other tunables ────────────────────────────────────────────────────────────
# _DEBT_TURNS: Henderson et al. (2020) "Dialogue system evaluation" — users
#     perceive passivity after ~4-5 turns without agent initiative.
_DEBT_TURNS = 5

# _STAGNATION_PERSISTENCE: Bohus & Rudnicky (2005) — transient vs. persistent
#     errors require different treatment. Require 2 consecutive stagnant turns
#     before REFRAME hard trigger fires for LOOPING (STUCK fires immediately
#     because severity warrants it).
_STAGNATION_PERSISTENCE = 2

# _TTR_SHORT_THRESHOLD: Sacks et al. (1974) minimum viable utterance length.
#     Messages of ≤ 3 words cannot carry significant lexical diversity.
_TTR_SHORT_THRESHOLD = 3

# _CONFUSION_QUESTION_BOOST: Boost applied when confusion marker appears in
#     interrogative position (end of utterance or preceding "?").
#     Based on Sacks et al. (1974) interrogative vs. declarative distinction.
_CONFUSION_QUESTION_BOOST = 0.35

# _CONFUSION_HITS_MAX: 3 confusion markers = saturation at score 1.0.
_CONFUSION_HITS_MAX = 3


# ── Types ──────────────────────────────────────────────────────────────────────

class InitiativeType(str, Enum):
    """
    8 strategies from ESConv (Liu et al., ACL 2021), adapted for general
    proactive assistant dialogue.

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
    QUESTION        = "QUESTION"
    RESTATEMENT     = "RESTATEMENT"
    REFLECTION      = "REFLECTION"
    AFFIRMATION     = "AFFIRMATION"
    SUGGESTION      = "SUGGESTION"
    INFORMATION     = "INFORMATION"
    REFRAME         = "REFRAME"
    SELF_DISCLOSURE = "SELF_DISCLOSURE"


@dataclass
class Signals:
    stagnation:       float   # 0–1  대화가 맴도는 정도  (state 기반)
    engagement_decay: float   # 0–1  참여도 하락 정도    (TTR 기반)
    confusion:        float   # 0–1  혼란/막힘 정도      (의문형 구분)
    coverage_gap:     float   # 0–1  WFC 미논의 비율     (WFC 상태)
    initiative_debt:  float   # 0–1  에이전트 수동성 누적 (연속 반응 횟수)

    @property
    def score(self) -> float:
        return (
            _W_STAGNATION * self.stagnation
            + _W_CONFUSION  * self.confusion
            + _W_COVERAGE   * self.coverage_gap
            + _W_ENGAGEMENT * self.engagement_decay
            + _W_DEBT       * self.initiative_debt
        )


@dataclass
class InitiativeDecision:
    should_intervene: bool
    initiative_type:  InitiativeType | None
    score:            float
    signals:          Signals


# ── Confusion markers — interrogative-position sensitive ──────────────────────
# Markers that signal confusion when they appear at the start/end of an utterance
# or precede "?", but not when embedded mid-sentence as connectives.
_CONFUSION_STARTERS = ("왜", "어떻게", "무슨", "뭔가", "what", "how", "why")
_CONFUSION_ANY      = ("이해가 안", "모르겠", "잘 모", "헷갈", "복잡해", "어렵",
                       "??", "unclear", "confused")


# ── Engine ────────────────────────────────────────────────────────────────────

class PRIMAEngine:
    """
    Decides *whether* and *how* the agent should take conversational initiative.

    Call `compute()` after each user turn to get an `InitiativeDecision`.
    Call `mark_intervened()` when the agent acts proactively (resets debt).
    """

    def __init__(self) -> None:
        self._consecutive_reactive: int = 0
        self._consecutive_stagnant: int = 0  # P1: persistence counter

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

        # ── Hard triggers (Deng et al. 2023: non-collaborative mode) ──────────
        #
        # STUCK  (stagnation=1.0): always fires immediately — severity warrants it.
        # LOOPING (stagnation=0.7): requires _STAGNATION_PERSISTENCE consecutive
        #   turns before firing (Bohus & Rudnicky 2005 — transient vs. persistent).
        if signals.stagnation >= 1.0:
            self._consecutive_stagnant += 1
            self._consecutive_reactive = 0
            return InitiativeDecision(
                should_intervene=True,
                initiative_type=InitiativeType.REFRAME,
                score=signals.score,
                signals=signals,
            )

        if signals.stagnation >= 0.7:
            self._consecutive_stagnant += 1
            if self._consecutive_stagnant >= _STAGNATION_PERSISTENCE:
                self._consecutive_reactive = 0
                return InitiativeDecision(
                    should_intervene=True,
                    initiative_type=InitiativeType.REFLECTION,
                    score=signals.score,
                    signals=signals,
                )
            # First stagnant turn: increment but don't intervene yet
            self._consecutive_reactive += 1
            return InitiativeDecision(
                should_intervene=False,
                initiative_type=None,
                score=signals.score,
                signals=signals,
            )

        # Reset stagnation counter when no longer stagnant
        self._consecutive_stagnant = 0

        # High confusion → RESTATEMENT (Clarification mode, Deng 2023)
        if signals.confusion >= 0.5:
            self._consecutive_reactive = 0
            return InitiativeDecision(
                should_intervene=True,
                initiative_type=InitiativeType.RESTATEMENT,
                score=signals.score,
                signals=signals,
            )

        # ── Soft scoring (Horvitz 1999: E[utility(act)] > E[utility(wait)]) ──
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
        self._consecutive_reactive = 0
        self._consecutive_stagnant = 0

    def reset(self) -> None:
        self._consecutive_reactive = 0
        self._consecutive_stagnant = 0

    # ── Signal computations ───────────────────────────────────────────────────

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
        Type-Token Ratio (TTR) based lexical richness measure.
        Richards (1987): TTR = unique_words / total_words.
        Low TTR → repetitive/thin engagement. Recency-weighted [0.2, 0.3, 0.5]
        using exponential decay (standard time-series weighting).

        Korean adjustment: Korean morphology packs more meaning per token than
        English, so short-length thresholds are set lower.
          ≤ 2 words: minimal turn ("응", "네") → high disengagement (0.75)
          3–5 words: moderate turn → linear score (0.40 → 0.10)
          > 5 words: use TTR normally
        """
        if len(turns) < 3:
            return 0.0

        recent = turns[-3:]
        weights = [0.2, 0.3, 0.5]
        scores: list[float] = []

        for turn in recent:
            words = turn.split()
            n = len(words)
            if n == 0:
                scores.append(1.0)
            elif n <= 2:
                # Minimal turn — Sacks et al. (1974): single-word turns
                # are conversational continuers, not substantive contributions.
                scores.append(0.75)
            elif n <= 5:
                # Short but not minimal: linear decay 0.40 → 0.10
                scores.append(max(0.10, 0.55 - 0.09 * n))
            else:
                unique = len(set(w.lower() for w in words))
                ttr = unique / n
                scores.append(max(0.0, 1.0 - ttr))

        return sum(w * s for w, s in zip(weights, scores))

    @staticmethod
    def _confusion(user_turns: list[str]) -> float:
        """
        Confusion detection that distinguishes interrogative from declarative use
        of ambiguous markers (Sacks et al. 1974).

        "왜?" or "왜 이게 안 되죠?" → confusion signal
        "왜냐하면 X이기 때문에"     → explanation, not confusion
        """
        if not user_turns:
            return 0.0

        # Aggregate last 2 turns — confusion often spans adjacent messages
        recent = " ".join(user_turns[-2:])
        text_lower = recent.lower().strip()

        hits = 0

        # Any-position markers (always confusion signals)
        for m in _CONFUSION_ANY:
            if m in text_lower:
                hits += 1

        # Position-sensitive markers: count only if at start or before "?"
        for m in _CONFUSION_STARTERS:
            if m not in text_lower:
                continue
            idx = text_lower.index(m)
            after = text_lower[idx + len(m):].lstrip()
            # Marker is followed immediately by "?" or is at utterance start (< 4 chars before)
            is_interrogative = after.startswith("?") or idx < 4
            # Marker is mid-sentence connective (e.g. "왜냐하면", "어떻게 보면")
            is_connective = any(
                text_lower[idx:].startswith(conn)
                for conn in ("왜냐하면", "어떻게 보면", "어떻게 생각하", "어떻게 하면")
            )
            if is_interrogative and not is_connective:
                hits += 1

        score = min(1.0, hits / _CONFUSION_HITS_MAX)

        # Boost if the last turn ends with "?" (interrogative form, Sacks 1974)
        last = user_turns[-1].strip()
        if last.endswith("?") or last.endswith("??"):
            score = min(1.0, score + _CONFUSION_QUESTION_BOOST)

        return score

    @staticmethod
    def _coverage_gap(total: int, collapsed: int) -> float:
        if total == 0:
            return 0.0
        return (total - collapsed) / total

    def _debt(self) -> float:
        return min(1.0, self._consecutive_reactive / _DEBT_TURNS)

    # ── Type selection ────────────────────────────────────────────────────────

    @staticmethod
    def _select_type(s: Signals) -> InitiativeType:
        """
        Maps signal patterns to ESConv strategy types following Deng et al. (2023)
        three-mode structure. Ordering mirrors ESConv strategy timing findings:
        reframe is late-stage; question/information are earlier-stage.
        """
        if s.stagnation >= 0.7:
            return InitiativeType.REFRAME
        if s.confusion >= 0.4:
            return InitiativeType.RESTATEMENT
        if s.coverage_gap >= 0.5 and s.stagnation < 0.4:
            return InitiativeType.INFORMATION
        if s.stagnation >= 0.3:
            return InitiativeType.REFLECTION
        if s.engagement_decay >= 0.5:
            return InitiativeType.SUGGESTION
        if s.engagement_decay >= 0.2 and s.initiative_debt >= 0.6:
            return InitiativeType.AFFIRMATION
        if s.initiative_debt >= 0.6:
            return InitiativeType.QUESTION
        return InitiativeType.SELF_DISCLOSURE
