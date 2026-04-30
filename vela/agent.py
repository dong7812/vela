import json
import re

from vela.core.context import ContextWindow
from vela.core.embedder import Embedder
from vela.core.state import ConversationState, StateDetector
from vela.core.wfc import CellState, ConversationCell, ConversationWFC
from vela.llm.base import BaseLLM
from vela.llm.ollama import OllamaLLM
from vela.rag.loader import load_document
from vela.rag.retriever import Retriever

_LANG = "반드시 한국어로 답변하세요."

_SYSTEM_PROMPTS: dict[ConversationState, str] = {
    ConversationState.EXPLORING: (
        f"당신은 유능한 어시스턴트입니다. 사용자의 질문에 명확하고 간결하게 답변하세요. {_LANG}"
    ),
    ConversationState.DEEPENING: (
        f"사용자가 특정 주제를 깊이 파고들고 있습니다. 구조화된 상세 정보를 제공하고 "
        f"추가 탐구를 유도하는 후속 질문을 제시하세요. {_LANG}"
    ),
    ConversationState.LOOPING: (
        f"대화가 같은 주제를 맴돌고 있습니다. 새로운 관점을 제안하거나 문제를 재구성해 "
        f"사용자가 앞으로 나아갈 수 있도록 도우세요. "
        f"반복 패턴을 인식했음을 언급하고 신선한 시각을 제공하세요. {_LANG}"
    ),
    ConversationState.STUCK: (
        f"사용자가 완전히 막혀 있습니다. 즉시 구체적인 대안 방향을 제시하세요. "
        f"직접적이고 실행 가능하게, 아직 시도하지 않은 2-3가지 접근법을 제안하세요. {_LANG}"
    ),
}

_PROACTIVE_PROMPTS: dict[ConversationState, str] = {
    ConversationState.LOOPING: (
        f"대화가 반복되고 있습니다. 사용자가 묻기를 기다리지 말고 먼저 능동적으로:\n"
        f"1. 반복 패턴을 부드럽게 짚어주세요 (1문장).\n"
        f"2. 완전히 새로운 각도나 재구성을 제안하세요.\n"
        f"3. 구체적인 방향 전환 질문으로 마무리하세요.\n"
        f"간결하게 — 총 3-4문장. {_LANG}"
    ),
    ConversationState.STUCK: (
        f"사용자가 완전히 막혀 있습니다. 묻기를 기다리지 말고 즉시 "
        f"시도하지 않은 2-3가지 구체적 대안을 제시하세요. "
        f"가장 유망한 것을 골라 직접적으로 밀어붙이세요. "
        f"3-5문장. {_LANG}"
    ),
}

_DOCUMENT_ANALYSIS_PROMPT = (
    f"방금 문서를 받았습니다. 사용자가 묻기 전에 먼저 능동적으로 분석하세요:\n"
    f"1. 주요 주제를 1-2문장으로 요약하세요.\n"
    f"2. 모호하거나 명확히 해야 할 부분 1-2가지를 짚어주세요.\n"
    f"3. 논의를 시작할 구체적인 출발점을 제안하세요.\n"
    f"간결하게 작성하고 사용자가 탐색을 시작하도록 초대하며 마무리하세요. {_LANG}"
)

_WFC_INIT_PROMPT = (
    "이 내용을 분석해 논의해야 할 주제 공간을 JSON으로 생성하세요.\n"
    "형식 — JSON 배열만 반환, 설명 없이:\n"
    '[{"topic":"주제명","description":"한줄설명","entropy":0.0~1.0,"related":["관련주제"]},...]\n'
    "entropy가 낮을수록 먼저 논의할 중요한 주제. 4~7개 항목. 한국어로."
)


def _parse_json_list(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        return [c for c in data if isinstance(c, dict) and c.get("topic")]
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_list(text: str, max_items: int = 5) -> list[str]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    items = []
    for line in lines:
        cleaned = re.sub(r"^[\d]+[.)]\s*|^[-•*]\s*", "", line).strip()
        if cleaned and len(cleaned) > 3:
            items.append(cleaned)
    return items[:max_items]


class VelaAgent:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm or OllamaLLM()
        self._context = ContextWindow()
        self._embedder = Embedder()
        self._state_detector = StateDetector(self._embedder)
        self._retriever = Retriever()
        self._wfc = ConversationWFC()

    # ── 문서 ─────────────────────────────────────────────────────────────

    def load_document(self, path: str) -> int:
        chunks = load_document(path)
        self._retriever.add_chunks(chunks, source=path)
        return len(chunks)

    def analyze_document(self) -> str:
        """문서 로드 직후 호출 — 에이전트가 먼저 분석 메시지를 생성하고 WFC를 초기화한다."""
        sample = self._retriever.search("주요 주제 개요 요약", top_k=3)
        if not sample:
            return ""
        preview = "\n\n".join(sample)
        messages = [{"role": "user", "content": f"문서 내용:\n{preview}"}]
        response = self._llm.chat(messages, system=_DOCUMENT_ANALYSIS_PROMPT)
        self._context.add("assistant", response)
        self._init_wfc_from_text(preview)
        return response

    # ── WFC ──────────────────────────────────────────────────────────────

    def _init_wfc_from_text(self, context: str) -> None:
        """주어진 텍스트를 기반으로 WFC 셀 공간을 초기화한다."""
        messages = [{"role": "user", "content": context}]
        response = self._llm.chat(messages, system=_WFC_INIT_PROMPT)
        cells = _parse_json_list(response)
        if cells:
            self._wfc.initialize(cells)

    def _init_wfc(self) -> None:
        """대화 맥락을 기반으로 WFC를 초기화 (문서 없을 때)."""
        context = " ".join(self._context.get_user_turns()[-3:])
        if context.strip():
            self._init_wfc_from_text(context)

    def _detect_discussed_cells(self, text: str) -> list[str]:
        """임베딩 유사도로 텍스트에서 논의된 셀을 감지."""
        candidates = [c for c in self._wfc.get_all() if c.state == CellState.SUPERPOSITION]
        if not candidates:
            return []
        all_texts = [text] + [f"{c.topic}: {c.description}" for c in candidates]
        embeddings = self._embedder.embed(all_texts)
        text_emb = embeddings[0]
        return [
            candidates[i].topic
            for i, cell_emb in enumerate(embeddings[1:])
            if Embedder.cosine_similarity(text_emb, cell_emb) > 0.55
        ]

    def wfc_proactive(self) -> str | None:
        """WFC 기반 능동 발화 — entropy 가장 낮은 셀을 꺼낸다."""
        next_cell = self._wfc.get_next()
        if not next_cell:
            return None
        self._wfc.collapse(next_cell.topic)
        system = (
            f"지금 '{next_cell.topic}'({next_cell.description})에 대해 "
            f"자연스럽게 먼저 꺼내세요. "
            f"사용자가 묻기 전에 능동적으로 이 주제를 제기하세요. "
            f"1-2문장으로 간결하게. {_LANG}"
        )
        response = self._llm.chat(self._context.to_messages(), system=system)
        self._context.add("assistant", response)
        return response

    def get_wfc_cells(self) -> list[ConversationCell]:
        return self._wfc.get_all()

    def get_wfc_next(self) -> ConversationCell | None:
        return self._wfc.get_next()

    # ── 대화 ─────────────────────────────────────────────────────────────

    def is_wfc_initialized(self) -> bool:
        return self._wfc.is_initialized()

    def init_wfc(self) -> None:
        """대화 맥락 기반 WFC 초기화 — UI에서 chat() 직후 별도 호출."""
        self._init_wfc()

    def chat(self, user_input: str) -> tuple[str, ConversationState]:
        self._context.add("user", user_input)
        rag_results = self._retriever.search(user_input)
        state = self._state_detector.detect(self._context.get_user_turns())

        system = _SYSTEM_PROMPTS[state]
        if rag_results:
            system += f"\n\nRelevant context from documents:\n{chr(10).join(rag_results)}"

        response = self._llm.chat(self._context.to_messages(), system=system)
        self._context.add("assistant", response)

        # WFC가 초기화된 경우에만 논의된 셀 collapse
        if self._wfc.is_initialized():
            for topic in self._detect_discussed_cells(user_input):
                self._wfc.collapse(topic)

        return response, state

    def proactive_nudge(self, state: ConversationState) -> str | None:
        """LOOPING/STUCK 상태일 때 에이전트가 먼저 개입 메시지를 생성한다."""
        system = _PROACTIVE_PROMPTS.get(state)
        if not system:
            return None
        response = self._llm.chat(self._context.to_messages(), system=system)
        self._context.add("assistant", response)
        return response

    def suggest_questions(self) -> list[str]:
        """WFC 소진 후 fallback — 선제 질문 3개 생성."""
        if not self._context.to_messages():
            return []
        system = (
            "지금까지의 대화를 바탕으로, 사용자가 다음에 물어볼 만한 질문 3개를 생성하세요. "
            "각 질문은 25자 이내로 간결하게. "
            "번호나 부가 설명 없이 질문만 한 줄씩. 한국어로."
        )
        response = self._llm.chat(self._context.to_messages(), system=system)
        return _parse_list(response, max_items=3)

    def reset(self) -> None:
        self._context.clear()
        self._wfc.reset()
