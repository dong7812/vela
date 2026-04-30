from enum import Enum

from vela.core.embedder import Embedder

LOOP_THRESHOLD = 0.85
STUCK_THRESHOLD = 0.95
MIN_TURNS_TO_JUDGE = 3


class ConversationState(str, Enum):
    EXPLORING = "EXPLORING"
    DEEPENING = "DEEPENING"
    LOOPING = "LOOPING"
    STUCK = "STUCK"


class StateDetector:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def detect(self, user_turns: list[str]) -> ConversationState:
        if len(user_turns) < MIN_TURNS_TO_JUDGE:
            return ConversationState.EXPLORING

        recent = user_turns[-MIN_TURNS_TO_JUDGE:]
        similarities = self._embedder.pairwise_similarities(recent)

        if not similarities:
            return ConversationState.EXPLORING

        avg = sum(similarities) / len(similarities)

        if avg >= STUCK_THRESHOLD:
            return ConversationState.STUCK
        if avg >= LOOP_THRESHOLD:
            return ConversationState.LOOPING
        if avg >= 0.60:
            return ConversationState.DEEPENING
        return ConversationState.EXPLORING
