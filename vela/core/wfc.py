from dataclasses import dataclass, field
from enum import Enum


class CellState(str, Enum):
    SUPERPOSITION = "SUPERPOSITION"  # 아직 논의 안 됨
    COLLAPSED = "COLLAPSED"          # 논의 완료


@dataclass
class ConversationCell:
    topic: str
    description: str
    entropy: float          # 낮을수록 먼저 논의 필요
    state: CellState = CellState.SUPERPOSITION
    related: list[str] = field(default_factory=list)


class ConversationWFC:
    def __init__(self) -> None:
        self._cells: dict[str, ConversationCell] = {}

    def initialize(self, cells: list[dict]) -> None:
        """LLM이 생성한 셀 데이터로 대화 공간 초기화."""
        self._cells = {}
        for c in cells:
            topic = c.get("topic", "").strip()
            if not topic:
                continue
            self._cells[topic] = ConversationCell(
                topic=topic,
                description=c.get("description", topic),
                entropy=float(c.get("entropy", 0.5)),
                related=[r for r in c.get("related", []) if isinstance(r, str)],
            )

    def collapse(self, topic: str) -> None:
        """셀을 논의 완료로 표시하고 관련 셀 우선순위 갱신."""
        cell = self._cells.get(topic)
        if cell and cell.state == CellState.SUPERPOSITION:
            cell.state = CellState.COLLAPSED
            self._propagate(cell)

    def _propagate(self, collapsed: ConversationCell) -> None:
        """관련 셀의 entropy 감소 → 더 빨리 꺼내야 하는 주제로 승격."""
        for related_topic in collapsed.related:
            related = self._cells.get(related_topic)
            if related and related.state == CellState.SUPERPOSITION:
                related.entropy = max(0.05, related.entropy - 0.2)

    def get_next(self) -> ConversationCell | None:
        """WFC 핵심: entropy 가장 낮은 미논의 셀 반환."""
        candidates = [c for c in self._cells.values() if c.state == CellState.SUPERPOSITION]
        return min(candidates, key=lambda c: c.entropy) if candidates else None

    def is_initialized(self) -> bool:
        return bool(self._cells)

    def get_all(self) -> list[ConversationCell]:
        return list(self._cells.values())

    def reset(self) -> None:
        self._cells = {}
