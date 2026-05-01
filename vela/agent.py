import json
import re

from vela.core.context import ContextWindow
from vela.core.embedder import Embedder
from vela.core.prima import InitiativeDecision, InitiativeType, PRIMAEngine
from vela.core.state import ConversationState, StateDetector
from vela.core.wfc import CellState, ConversationCell, ConversationWFC
from vela.llm.base import BaseLLM
from vela.llm.ollama import OllamaLLM
from vela.rag.loader import load_document
from vela.rag.retriever import Retriever

_LANG = "반드시 한국어로 답변하세요."

_SYSTEM_PROMPTS: dict[ConversationState, str] = {
    ConversationState.EXPLORING: (
        f"당신은 사용자의 목표를 깊이 이해하는 파트너입니다. "
        f"단순히 질문에 답하는 게 아니라, 사용자가 미처 생각하지 못한 부분까지 "
        f"자연스럽게 짚어주세요. 의견이 있으면 근거와 함께 직접적으로 말하세요. {_LANG}"
    ),
    ConversationState.DEEPENING: (
        f"사용자가 특정 주제를 깊이 파고들고 있습니다. "
        f"표면적인 답변 대신, 실제로 중요한 트레이드오프나 리스크를 짚어주세요. "
        f"전문가처럼 구체적인 수치, 사례, 근거를 들어 말하세요. {_LANG}"
    ),
    ConversationState.LOOPING: (
        f"대화가 같은 자리를 맴돌고 있습니다. 정중하게 멈추고 "
        f"'왜 이 부분에서 계속 머무는지'를 파악해 근본 원인을 직접 짚어주세요. "
        f"다른 프레임으로 문제를 재정의해서 새로운 방향을 제시하세요. {_LANG}"
    ),
    ConversationState.STUCK: (
        f"사용자가 막혀 있습니다. 현재 접근 방식의 어디가 문제인지 솔직하게 말하고, "
        f"완전히 다른 각도에서 2-3가지 구체적인 대안을 제시하세요. "
        f"가장 좋다고 생각하는 것을 직접 추천하고 이유를 말하세요. {_LANG}"
    ),
}

# ── PRIMA 능동 개입 프롬프트 ──────────────────────────────────────────────────
# ESConv (Liu et al., ACL 2021) 8가지 전략을 범용 능동 어시스턴트 맥락으로 적용.

_INITIATIVE_PROMPTS: dict[InitiativeType, str] = {
    # [Clarification mode]
    InitiativeType.QUESTION: (
        f"사용자의 진짜 목표나 의도를 더 깊이 파악하기 위해 핵심 질문을 하나 던지세요. "
        f"'X를 원하시는 건가요, 아니면 Y를 원하시는 건가요?'처럼 "
        f"사용자 스스로도 아직 명확히 하지 못한 부분을 짚는 질문이어야 합니다. "
        f"답변보다 질문 먼저. 2문장. {_LANG}"
    ),
    InitiativeType.RESTATEMENT: (
        f"사용자가 말한 내용을 당신의 언어로 다시 정리해서 이해가 맞는지 확인하세요. "
        f"'제가 이해한 바로는 X인데, 맞나요?'처럼 구체적으로 재진술하고, "
        f"이해가 맞으면 다음 단계를, 틀리면 수정 기회를 주세요. 2-3문장. {_LANG}"
    ),
    # [Target-guided mode]
    InitiativeType.REFLECTION: (
        f"지금 대화의 흐름을 있는 그대로 짚어주세요. "
        f"'지금 X 부분에서 계속 머물고 있는 것 같습니다'처럼 "
        f"판단 없이 상황을 반영하고, 왜 여기서 막히는지 함께 탐색해보자고 제안하세요. "
        f"2-3문장. {_LANG}"
    ),
    InitiativeType.INFORMATION: (
        f"대화에서 아직 다루지 않았지만 지금 알아야 할 중요한 정보가 있습니다. "
        f"사용자가 요청하기 전에 먼저 꺼내세요. "
        f"왜 지금 이게 중요한지 이유와 함께, 구체적인 수치나 사례를 들어 전달하세요. "
        f"2-3문장. {_LANG}"
    ),
    # [Non-collaborative mode]
    InitiativeType.AFFIRMATION: (
        f"사용자의 접근 방식이나 판단에서 잘 되고 있는 부분을 구체적으로 짚어주세요. "
        f"막연한 칭찬이 아니라 '이 부분이 좋은 이유는 X 때문입니다'처럼 "
        f"근거 있는 확인을 해주세요. 그리고 자연스럽게 이어질 방향도 제시하세요. "
        f"2-3문장. {_LANG}"
    ),
    InitiativeType.SUGGESTION: (
        f"지금 대화에서 다음으로 할 수 있는 가장 구체적인 한 가지를 제안하세요. "
        f"'제 생각엔 지금 당장 X를 해보는 게 좋겠습니다. 왜냐하면...'처럼 "
        f"이유와 함께 단호하게 제안하세요. 두루뭉술하면 안 됩니다. 2-3문장. {_LANG}"
    ),
    InitiativeType.REFRAME: (
        f"지금 접근 방식 자체가 문제일 수 있습니다. "
        f"'사실 진짜 문제는 X가 아니라 Y입니다'처럼 "
        f"현재 프레임을 완전히 다른 각도에서 재정의하세요. "
        f"왜 그렇게 보는지 근거를 들어 대담하게 말하세요. 2-3문장. {_LANG}"
    ),
    InitiativeType.SELF_DISCLOSURE: (
        f"파트너로서 당신의 관점을 직접 공유하세요. "
        f"'제가 이 상황을 보면서 느끼는 건...' 또는 '저라면 이렇게 할 것 같습니다'처럼 "
        f"사용자가 놓치고 있을 수 있는 부분을 AI의 시각에서 솔직하게 말하세요. "
        f"2-3문장. {_LANG}"
    ),
}

_DOCUMENT_ANALYSIS_PROMPT = (
    f"방금 문서를 받았습니다. 단순 요약이 아니라 파트너로서 분석하세요.\n"
    f"1. 이 문서의 핵심 목적과 가장 중요한 결정 사항을 한 문장으로.\n"
    f"2. 읽으면서 걱정되거나 빠져 있다고 느낀 부분을 솔직하게 말하세요.\n"
    f"3. '저라면 여기서부터 시작하겠습니다'로 시작하는 구체적인 제안을 하세요.\n"
    f"전문가 동료가 문서를 처음 받아보고 바로 반응하듯 자연스럽고 직접적으로. {_LANG}"
)

_WFC_INIT_PROMPT = (
    "대화 맥락을 분석해 사용자가 목표를 달성하려면 반드시 다뤄야 할 핵심 사안들을 JSON으로 생성하세요.\n"
    "단순한 주제 목록이 아니라, 각 항목은 '이걸 안 짚으면 나중에 문제가 생길 수 있는 것'이어야 합니다.\n"
    "형식 — JSON 배열만 반환, 설명 없이:\n"
    '[{"topic":"주제명","description":"왜 이게 중요한지 한줄","entropy":0.0~1.0,"related":["관련주제"]},...]\n'
    "entropy가 낮을수록 더 시급하고 중요한 사안. 4~6개 항목. 한국어로."
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
        self._prima = PRIMAEngine()

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
        messages = [{"role": "user", "content": context}]
        response = self._llm.chat(messages, system=_WFC_INIT_PROMPT)
        cells = _parse_json_list(response)
        if cells:
            self._wfc.initialize(cells)

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

    def wfc_proactive(self) -> str | None:
        """WFC 기반 능동 발화 — entropy 가장 낮은 셀을 꺼낸다."""
        next_cell = self._wfc.get_next()
        if not next_cell:
            return None
        self._wfc.collapse(next_cell.topic)
        system = (
            f"당신은 사용자의 목표를 함께 달성하려는 파트너입니다.\n"
            f"지금 반드시 짚어야 할 사안: '{next_cell.topic}'\n"
            f"이유: {next_cell.description}\n\n"
            f"사용자가 묻기를 기다리지 말고 먼저 이 주제를 꺼내세요. "
            f"단순히 '이 주제 얘기해봐요'가 아니라, "
            f"왜 지금 이게 중요한지 구체적인 이유와 함께 당신의 의견을 직접 말하세요. "
            f"필요하다면 사용자의 선택이나 방향에 의문을 제기해도 됩니다. "
            f"2-3문장. {_LANG}"
        )
        response = self._llm.chat(self._context.to_messages(), system=system)
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

    def chat(self, user_input: str) -> tuple[str, ConversationState, InitiativeDecision]:
        self._context.add("user", user_input)
        rag_results = self._retriever.search(user_input)
        state = self._state_detector.detect(self._context.get_user_turns())

        system = _SYSTEM_PROMPTS[state]
        if rag_results:
            system += f"\n\nRelevant context from documents:\n{chr(10).join(rag_results)}"

        response = self._llm.chat(self._context.to_messages(), system=system)
        self._context.add("assistant", response)

        # WFC collapse
        if self._wfc.is_initialized():
            for topic in self._detect_discussed_cells(user_input):
                self._wfc.collapse(topic)

        # PRIMA 개입 판단 (LLM 호출 없음 — 순수 신호 계산)
        all_cells = self._wfc.get_all()
        wfc_total = len(all_cells)
        wfc_collapsed = sum(1 for c in all_cells if c.state == CellState.COLLAPSED)
        decision = self._prima.compute(
            user_turns=self._context.get_user_turns(),
            state=state,
            wfc_total=wfc_total,
            wfc_collapsed=wfc_collapsed,
        )

        return response, state, decision

    def prima_intervene(self, initiative_type: InitiativeType) -> str:
        """PRIMA가 결정한 initiative_type에 맞는 능동 발화를 생성한다."""
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
