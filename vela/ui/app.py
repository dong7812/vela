import streamlit as st

from vela.agent import VelaAgent
from vela.core.state import ConversationState
from vela.core.wfc import CellState
from vela.llm.claude import ClaudeLLM, _MODELS as CLAUDE_MODELS

_STATE_LABELS = {
    ConversationState.EXPLORING: ("🔍 탐색 중", "blue"),
    ConversationState.DEEPENING: ("🔎 심화 중", "green"),
    ConversationState.LOOPING: ("🔄 반복 감지됨", "orange"),
    ConversationState.STUCK: ("⛔ 막힘 감지됨", "red"),
}

_STATE_DESCRIPTIONS = {
    ConversationState.EXPLORING: "새로운 주제를 탐색하고 있습니다.",
    ConversationState.DEEPENING: "주제가 깊어지고 있습니다.",
    ConversationState.LOOPING: "Vela가 새로운 방향을 제안합니다.",
    ConversationState.STUCK: "Vela가 즉시 대안을 제시합니다.",
}


@st.cache_resource
def _get_agent(llm_key: str = "ollama") -> VelaAgent:
    if llm_key.startswith("claude:"):
        _, api_key, model = llm_key.split(":", 2)
        return VelaAgent(llm=ClaudeLLM(api_key=api_key, model=model))
    return VelaAgent()


def _build_llm_key() -> str:
    llm_choice = st.session_state.get("llm_choice", "Ollama (로컬)")
    if llm_choice == "Claude API":
        api_key = st.session_state.get("claude_api_key", "")
        model = st.session_state.get("claude_model", CLAUDE_MODELS[0])
        if api_key:
            return f"claude:{api_key}:{model}"
    return "ollama"


def _init_session() -> None:
    defaults = {
        "messages": [],
        "last_state": None,
        "analyzed_files": set(),
        "last_suggestions": [],
        "suggestion_key": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_sidebar(agent: VelaAgent) -> None:
    with st.sidebar:
        # LLM 설정
        st.header("LLM 설정")
        st.selectbox(
            "모델 선택",
            ["Ollama (로컬)", "Claude API"],
            key="llm_choice",
        )
        if st.session_state.get("llm_choice") == "Claude API":
            st.text_input("API Key", type="password", key="claude_api_key", placeholder="sk-ant-...")
            st.selectbox("Claude 모델", CLAUDE_MODELS, key="claude_model")
            if st.session_state.get("claude_api_key"):
                st.success("✅ API Key 입력됨")
            else:
                st.warning("API Key를 입력하세요")

        st.divider()
        st.header("문서 로드")
        uploaded = st.file_uploader(
            "파일을 업로드하세요 (txt, md, pdf)",
            type=["txt", "md", "pdf"],
        )
        if uploaded is not None:
            import tempfile, os

            file_key = f"{uploaded.name}_{uploaded.size}"
            if file_key not in st.session_state.analyzed_files:
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    count = agent.load_document(tmp_path)
                    st.success(f"✅ {uploaded.name} 로드 완료 ({count}개 청크)")
                    with st.spinner("문서 분석 + 대화 공간 구성 중..."):
                        analysis = agent.analyze_document()
                    st.session_state.analyzed_files.add(file_key)
                    st.session_state.last_suggestions = []
                    if analysis:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": analysis, "tag": "📄 문서 분석"}
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"문서 로드 실패: {e}")
                finally:
                    os.unlink(tmp_path)

        # WFC 대화 공간 시각화
        cells = agent.get_wfc_cells()
        if cells:
            st.divider()
            next_cell = agent.get_wfc_next()
            n_collapsed = sum(1 for c in cells if c.state == CellState.COLLAPSED)
            st.header(f"대화 공간 ({n_collapsed}/{len(cells)})")

            sorted_cells = sorted(
                cells,
                key=lambda c: (c.state == CellState.COLLAPSED, c.entropy),
            )
            for cell in sorted_cells:
                if cell.state == CellState.COLLAPSED:
                    st.markdown(f"✅ ~~{cell.topic}~~")
                elif next_cell and cell.topic == next_cell.topic:
                    st.markdown(f"▶️ **{cell.topic}**")
                    st.caption(cell.description)
                else:
                    st.markdown(f"○ {cell.topic}")

        st.divider()
        st.header("대화 상태")
        state = st.session_state.last_state
        if state:
            label, color = _STATE_LABELS[state]
            st.markdown(f":{color}[**{label}**]")
            st.caption(_STATE_DESCRIPTIONS[state])
        else:
            st.caption("아직 대화가 시작되지 않았습니다.")

        if st.button("대화 초기화"):
            agent.reset()
            st.session_state.messages = []
            st.session_state.last_state = None
            st.session_state.analyzed_files = set()
            st.session_state.last_suggestions = []
            st.session_state.suggestion_key = 0
            st.rerun()


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if tag := msg.get("tag"):
                st.caption(tag)
            elif state := msg.get("state"):
                label, color = _STATE_LABELS[state]
                st.caption(f":{color}[{label}]")


def _render_suggestions() -> None:
    suggestions = st.session_state.last_suggestions
    if not suggestions:
        return
    st.markdown("**💡 다음 질문 제안:**")
    cols = st.columns(len(suggestions))
    for j, (col, q) in enumerate(zip(cols, suggestions)):
        if col.button(q, key=f"sq_{st.session_state.suggestion_key}_{j}", use_container_width=True):
            st.session_state.pending_input = q
            st.session_state.last_suggestions = []
            st.session_state.suggestion_key += 1
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Vela", page_icon="🌌", layout="centered")
    st.title("🌌 Vela")
    st.caption("AI SDK that acts before you ask — 로컬 LLM 기반 능동 대화 에이전트")

    _init_session()
    agent = _get_agent(_build_llm_key())
    _render_sidebar(agent)
    _render_history()
    _render_suggestions()

    pending = st.session_state.pop("pending_input", None)
    user_input = pending or st.chat_input("메시지를 입력하세요...")

    if user_input:
        st.session_state.last_suggestions = []
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 1. 응답
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    response, detected_state = agent.chat(user_input)
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}\nOllama가 실행 중인지 확인해 주세요.")
                    st.stop()
            st.markdown(response)
            label, color = _STATE_LABELS[detected_state]
            st.caption(f":{color}[{label}]")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "state": detected_state}
        )
        st.session_state.last_state = detected_state

        # 2. WFC 초기화 (첫 턴, 문서 없을 때)
        if not agent.is_wfc_initialized():
            with st.spinner("대화 공간 구성 중..."):
                agent.init_wfc()

        # 3. 능동 후속 — 우선순위: LOOPING/STUCK > WFC > 선제 질문
        proactive_msg = None
        proactive_tag = None

        if detected_state in (ConversationState.LOOPING, ConversationState.STUCK):
            with st.spinner("Vela 개입 중..."):
                proactive_msg = agent.proactive_nudge(detected_state)
            proactive_tag = f":{color}[{label}] — 🎯 능동 개입"

        elif agent.get_wfc_next():
            with st.spinner("다음 주제 꺼내는 중..."):
                proactive_msg = agent.wfc_proactive()
            next_cell = agent.get_wfc_next()
            proactive_tag = f"🌊 WFC — {next_cell.topic if next_cell else '대화 공간 탐색'}"

        if proactive_msg:
            with st.chat_message("assistant"):
                st.markdown(proactive_msg)
                st.caption(proactive_tag or "🎯 Vela 능동 개입")
            st.session_state.messages.append(
                {"role": "assistant", "content": proactive_msg,
                 "state": detected_state, "tag": proactive_tag}
            )
        else:
            with st.spinner("질문 제안 중..."):
                st.session_state.last_suggestions = agent.suggest_questions()
            st.session_state.suggestion_key += 1
            st.rerun()


main()
