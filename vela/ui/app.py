import streamlit as st

from vela.agent import VelaAgent
from vela.core.state import ConversationState

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
def _get_agent() -> VelaAgent:
    return VelaAgent()


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "analyzed_files" not in st.session_state:
        st.session_state.analyzed_files = set()
    if "last_suggestions" not in st.session_state:
        st.session_state.last_suggestions = []
    if "suggestion_key" not in st.session_state:
        st.session_state.suggestion_key = 0


def _render_sidebar(agent: VelaAgent) -> None:
    with st.sidebar:
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
                    with st.spinner("문서 분석 중..."):
                        analysis = agent.analyze_document()
                    with st.spinner("논의 계획 수립 중..."):
                        agent.generate_agenda()
                    first_item = agent.advance_agenda()

                    st.session_state.analyzed_files.add(file_key)
                    st.session_state.last_suggestions = []
                    if analysis:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": analysis, "proactive": True}
                        )
                    if first_item:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": first_item, "proactive": True, "agenda": True}
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"문서 로드 실패: {e}")
                finally:
                    os.unlink(tmp_path)

        # 논의 계획 진행 상황
        agenda = agent.get_agenda()
        if agenda:
            st.divider()
            idx = agent.get_agenda_index()
            st.header(f"논의 계획 ({min(idx, len(agenda))}/{len(agenda)})")
            for i, item in enumerate(agenda):
                if i < idx:
                    st.markdown(f"✅ ~~{item}~~")
                elif i == idx:
                    st.markdown(f"▶️ **{item}**")
                else:
                    st.markdown(f"○ {item}")

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
            if msg.get("agenda"):
                st.caption("📋 Vela Agenda — 능동 진행")
            elif msg.get("proactive") and msg["role"] == "assistant":
                st.caption("🎯 Vela 능동 개입")
            elif msg.get("state"):
                s = msg["state"]
                label, color = _STATE_LABELS[s]
                st.caption(f":{color}[{label}]")


def _render_suggestions() -> None:
    suggestions = st.session_state.last_suggestions
    if not suggestions:
        return
    st.markdown("**💡 다음 질문 제안:**")
    cols = st.columns(len(suggestions))
    for j, (col, q) in enumerate(zip(cols, suggestions)):
        key = f"sq_{st.session_state.suggestion_key}_{j}"
        if col.button(q, key=key, use_container_width=True):
            st.session_state.pending_input = q
            st.session_state.last_suggestions = []
            st.session_state.suggestion_key += 1
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Vela", page_icon="🌌", layout="centered")
    st.title("🌌 Vela")
    st.caption("AI SDK that acts before you ask — 로컬 LLM 기반 능동 대화 에이전트")

    _init_session()
    agent = _get_agent()
    _render_sidebar(agent)
    _render_history()
    _render_suggestions()

    # 버튼 클릭 또는 직접 입력 처리
    pending = st.session_state.pop("pending_input", None)
    user_input = pending or st.chat_input("메시지를 입력하세요...")

    if user_input:
        st.session_state.last_suggestions = []
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 1. 일반 응답
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

        # 2. 능동 후속 처리 — 우선순위: LOOPING/STUCK > Agenda > 선제 질문
        proactive_msg = None
        if detected_state in (ConversationState.LOOPING, ConversationState.STUCK):
            with st.spinner("Vela 개입 중..."):
                proactive_msg = agent.proactive_nudge(detected_state)
        elif agent.has_agenda():
            with st.spinner("다음 주제 준비 중..."):
                proactive_msg = agent.advance_agenda()

        if proactive_msg:
            is_agenda = agent.get_agenda_index() <= len(agent.get_agenda())
            with st.chat_message("assistant"):
                st.markdown(proactive_msg)
                st.caption("📋 Vela Agenda — 능동 진행" if is_agenda and detected_state not in (ConversationState.LOOPING, ConversationState.STUCK) else f":{color}[{label}] — 🎯 능동 개입")
            st.session_state.messages.append(
                {"role": "assistant", "content": proactive_msg,
                 "state": detected_state,
                 "proactive": True,
                 "agenda": is_agenda and detected_state not in (ConversationState.LOOPING, ConversationState.STUCK)}
            )
        else:
            # Agenda 없고 정상 상태 → 선제 질문 생성
            with st.spinner("질문 제안 중..."):
                st.session_state.last_suggestions = agent.suggest_questions()
            st.session_state.suggestion_key += 1
            st.rerun()


main()
