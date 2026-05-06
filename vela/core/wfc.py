from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vela.core.embedder import Embedder

# Minimum cosine similarity between a cell's description and the source context
# for the cell to be retained after quality filtering.
# Reimers & Gurevych (2019): all-MiniLM-L6-v2 similarity < 0.15 indicates
# semantically unrelated content. Set permissively to only remove clear outliers.
_MIN_CELL_RELEVANCE = 0.15


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

    def filter_by_relevance(self, context: str, embedder: "Embedder") -> int:
        """
        P5: WFC 셀 품질 필터.
        각 셀의 description을 소스 context와 비교해 관련성이 낮은 셀을 제거한다.
        임계값 _MIN_CELL_RELEVANCE (0.15) 미만 셀은 LLM 환각으로 판단.

        Reimers & Gurevych (2019): all-MiniLM-L6-v2에서 코사인 유사도 < 0.15는
        의미적으로 무관한 내용임을 의미.

        Returns: 제거된 셀 수
        """
        if not self._cells or not context.strip():
            return 0

        topics = list(self._cells.keys())
        texts = [context] + [f"{t}: {self._cells[t].description}" for t in topics]
        embeddings = embedder.embed(texts)
        context_emb = embeddings[0]

        from vela.core.embedder import Embedder as _Embedder
        to_remove = [
            topic
            for i, topic in enumerate(topics)
            if _Embedder.cosine_similarity(context_emb, embeddings[i + 1]) < _MIN_CELL_RELEVANCE
        ]
        for topic in to_remove:
            del self._cells[topic]
        return len(to_remove)

    def is_initialized(self) -> bool:
        return bool(self._cells)

    def get_all(self) -> list[ConversationCell]:
        return list(self._cells.values())

    def reset(self) -> None:
        self._cells = {}
