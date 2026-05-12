import json
import re
from typing import TYPE_CHECKING, Iterator

from vela.core.context import ContextWindow
from vela.core.embedder import Embedder
from vela.core.feedback import FeedbackLogger
from vela.core.prima import InitiativeDecision, InitiativeType, PRIMAEngine
from vela.core.state import ConversationState, StateDetector
from vela.core.wfc import CellState, ConversationCell, ConversationWFC
from vela.domain.conversation.esconv import (
    DOCUMENT_ANALYSIS_PROMPT as _DOCUMENT_ANALYSIS_PROMPT,
    INITIATIVE_PROMPTS as _INITIATIVE_PROMPTS,
    SYSTEM_PROMPTS as _SYSTEM_PROMPTS,
    WFC_INIT_PROMPT as _WFC_INIT_PROMPT,
)
from vela.llm.base import BaseLLM
from vela.llm.ollama import OllamaLLM
from vela.rag.loader import load_document
from vela.rag.retriever import Retriever

if TYPE_CHECKING:
    from vela.core.output_generator import OutputResult
    from vela.core.pipeline import VelaPipeline
    from vela.domain.base import DomainPlugin


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
    def __init__(
        self,
        llm: BaseLLM | None = None,
        plugin: "DomainPlugin | None" = None,
    ) -> None:
        self._llm = llm or OllamaLLM()
        self._plugin = plugin

        # 범용 파이프라인 (non-conversation 도메인용)
        if plugin is not None:
            from vela.core.pipeline import VelaPipeline
            self._pipeline: VelaPipeline = VelaPipeline(plugin, llm=self._llm)

        # 대화 도메인 (기존 로직 — UI 하위 호환)
        self._context = ContextWindow()
        self._embedder = Embedder()
        self._state_detector = StateDetector(self._embedder)
        self._retriever = Retriever(embedder=self._embedder)
        self._wfc = ConversationWFC()
        self._prima = PRIMAEngine()
        self._last_user_input: str = ""
        self._doc_preview: str = ""
        self.feedback = FeedbackLogger()

    def run(self, current: dict, history: list[dict] | None = None) -> "OutputResult | None":
        """DomainPlugin 기반 3-layer pipeline 실행."""
        if self._plugin is None:
            raise ValueError("plugin이 설정되지 않았습니다. VelaAgent(plugin=...) 로 초기화하세요.")
        return self._pipeline.run(current, history or [])

    # ── 문서 ─────────────────────────────────────────────────────────────

    def load_document(self, path: str) -> int:
        chunks = load_document(path)
        self._retriever.add_chunks(chunks, source=path)
        return len(chunks)

    def analyze_document(self) -> str:
        """비스트리밍 SDK용."""
        sample = self._retriever.search("주요 주제 개요 요약", top_k=3)
        if not sample:
            return ""
        self._doc_preview = "\n\n".join(sample)
        messages = [{"role": "user", "content": f"문서 내용:\n{self._doc_preview}"}]
        response = self._llm.chat(messages, system=_DOCUMENT_ANALYSIS_PROMPT)
        self._context.add("assistant", response)
        self._init_wfc_from_text(self._doc_preview)
        return response

    def analyze_document_stream(self) -> Iterator[str]:
        """UI용 스트리밍 분석 — WFC 초기화는 포함하지 않음."""
        sample = self._retriever.search("주요 주제 개요 요약", top_k=3)
        if not sample:
            return
        self._doc_preview = "\n\n".join(sample)
        messages = [{"role": "user", "content": f"문서 내용:\n{self._doc_preview}"}]
        accumulated: list[str] = []
        for token in self._llm.chat_stream(messages, system=_DOCUMENT_ANALYSIS_PROMPT):
            accumulated.append(token)
            yield token
        if accumulated:
            self._context.add("assistant", "".join(accumulated))

    def init_wfc_from_document(self) -> None:
        """analyze_document_stream() 완료 후 호출 — 저장된 문서 미리보기로 WFC 초기화."""
        if self._doc_preview:
            self._init_wfc_from_text(self._doc_preview)

    # ── WFC ──────────────────────────────────────────────────────────────

    def _init_wfc_from_text(self, context: str) -> None:
        messages = [{"role": "user", "content": context}]
        response = self._llm.chat(messages, system=_WFC_INIT_PROMPT)
        cells = _parse_json_list(response)
        if cells:
            self._wfc.initialize(cells)
            # P5: 생성된 셀 중 문맥과 무관한 셀 제거 (Reimers & Gurevych 2019)
            self._wfc.filter_by_relevance(context, self._embedder)

    def _init_wfc(self) -> None:
        context = " ".join(self._context.get_user_turns()[-3:])
        if context.strip():
            self._init_wfc_from_text(context)

    def _detect_discussed_cells(self, text: str) -> list[str]:
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

    def _wfc_system(self, cell: ConversationCell) -> str:
        return (
            f"당신은 사용자의 목표를 함께 달성하려는 파트너입니다.\n"
            f"지금 반드시 짚어야 할 사안: '{cell.topic}'\n"
            f"이유: {cell.description}\n\n"
            f"사용자가 묻기를 기다리지 말고 먼저 이 주제를 꺼내세요. "
            f"단순히 '이 주제 얘기해봐요'가 아니라, "
            f"왜 지금 이게 중요한지 구체적인 이유와 함께 당신의 의견을 직접 말하세요. "
            f"필요하다면 사용자의 선택이나 방향에 의문을 제기해도 됩니다. "
            f"2-3문장. {_LANG}"
        )

    def wfc_proactive_stream(self) -> Iterator[str]:
        """WFC 기반 능동 발화 스트리밍 버전."""
        next_cell = self._wfc.get_next()
        if not next_cell:
            return
        self._wfc.collapse(next_cell.topic)
        accumulated: list[str] = []
        for token in self._llm.chat_stream(self._context.to_messages(), system=self._wfc_system(next_cell)):
            accumulated.append(token)
            yield token
        self._context.add("assistant", "".join(accumulated))
        self._prima.mark_intervened()

    def wfc_proactive(self) -> str | None:
        """비스트리밍 SDK용."""
        next_cell = self._wfc.get_next()
        if not next_cell:
            return None
        self._wfc.collapse(next_cell.topic)
        response = self._llm.chat(self._context.to_messages(), system=self._wfc_system(next_cell))
        self._context.add("assistant", response)
        self._prima.mark_intervened()
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

    def prepare_chat(self, user_input: str) -> tuple[list[dict], str, ConversationState]:
        """스트리밍 전 단계: 유저 턴 추가, 상태 감지, messages/system 반환."""
        self._last_user_input = user_input
        self._context.add("user", user_input)
        rag_results = self._retriever.search(user_input)
        state = self._state_detector.detect(self._context.get_user_turns())
        system = _SYSTEM_PROMPTS[state]
        if rag_results:
            system += f"\n\nRelevant context from documents:\n{chr(10).join(rag_results)}"
        return self._context.to_messages(), system, state

    def finalize_chat(self, response: str, state: ConversationState) -> InitiativeDecision:
        """스트리밍 완료 후 단계: 응답 저장, WFC collapse, PRIMA 판단."""
        self._context.add("assistant", response)
        if self._wfc.is_initialized():
            for topic in self._detect_discussed_cells(self._last_user_input):
                self._wfc.collapse(topic)
        all_cells = self._wfc.get_all()
        return self._prima.compute(
            user_turns=self._context.get_user_turns(),
            state=state,
            wfc_total=len(all_cells),
            wfc_collapsed=sum(1 for c in all_cells if c.state == CellState.COLLAPSED),
        )

    def chat(self, user_input: str) -> tuple[str, ConversationState, InitiativeDecision]:
        """비스트리밍 SDK용. UI는 prepare_chat / finalize_chat을 사용할 것."""
        messages, system, state = self.prepare_chat(user_input)
        response = self._llm.chat(messages, system=system)
        decision = self.finalize_chat(response, state)
        return response, state, decision

    def prima_intervene_stream(self, initiative_type: InitiativeType) -> Iterator[str]:
        """PRIMA 전략에 맞는 능동 발화를 스트리밍으로 생성."""
        system = _INITIATIVE_PROMPTS[initiative_type]
        accumulated: list[str] = []
        for token in self._llm.chat_stream(self._context.to_messages(), system=system):
            accumulated.append(token)
            yield token
        self._context.add("assistant", "".join(accumulated))
        self._prima.mark_intervened()

    def prima_intervene(self, initiative_type: InitiativeType) -> str:
        """비스트리밍 SDK용."""
        system = _INITIATIVE_PROMPTS[initiative_type]
        response = self._llm.chat(self._context.to_messages(), system=system)
        self._context.add("assistant", response)
        self._prima.mark_intervened()
        return response

    def suggest_questions(self) -> list[str]:
        """PRIMA 점수 미달 시 fallback — 선제 질문 3개 생성."""
        if not self._context.to_messages():
            return []
        system = (
            "지금까지의 대화를 분석해서, 사용자가 목표를 달성하려면 "
            "반드시 짚어야 하는데 아직 다루지 않은 핵심 질문 3개를 만드세요. "
            "사용자가 스스로 떠올리기 어려운 날카로운 질문이어야 합니다. "
            "각 질문은 25자 이내. 번호나 부가 설명 없이 질문만 한 줄씩. 한국어로."
        )
        response = self._llm.chat(self._context.to_messages(), system=system)
        return _parse_list(response, max_items=3)

    def reset(self) -> None:
        self._context.clear()
        self._wfc.reset()
        self._prima.reset()
