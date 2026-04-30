"""
requirements_bot: 요구사항 명세서 능동 분석 챗봇 데모

사용법:
    streamlit run examples/requirements_bot/app.py
"""

import streamlit as st

from vela.agent import VelaAgent
from vela.core.state import ConversationState
from vela.core.wfc import CellState

_STATE_LABELS = {
    ConversationState.EXPLORING: ("🔍 탐색 중", "blue"),
    ConversationState.DEEPENING: ("🔎 심화 분석 중", "green"),
    ConversationState.LOOPING: ("🔄 반복 감지 — 재구성 제안", "orange"),
    ConversationState.STUCK: ("⛔ 막힘 — 대안 제시", "red"),
}


@st.cache_resource
def _get_agent() -> VelaAgent:
    return VelaAgent()


def main() -> None:
    st.set_page_config(page_title="Requirements Bot", page_icon="📋", layout="centered")
    st.title("📋 Requirements Bot")
    st.caption("요구사항 명세서를 업로드하면 Vela가 먼저 분석하고 대화 공간을 구성합니다.")

    defaults = {
        "req_messages": [],
        "analyzed_files": set(),
        "last_suggestions": [],
        "suggestion_key": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    agent = _get_agent()

    with st.sidebar:
        st.header("명세서 로드")
        uploaded = st.file_uploader("명세서 파일 (txt, md, pdf)", type=["txt", "md", "pdf"])
        if uploaded:
            import os, tempfile

            file_key = f"{uploaded.name}_{uploaded.size}"
            if file_key not in st.session_state.analyzed_files:
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    count = agent.load_document(tmp_path)
                    st.success(f"✅ 로드 완료 — {count}개 청크")
                    with st.spinner("명세서 분석 + 대화 공간 구성 중..."):
                        analysis = agent.analyze_document()
                    st.session_state.analyzed_files.add(file_key)
                    st.session_state.last_suggestions = []
                    if analysis:
                        st.session_state.req_messages.append(
                            {"role": "assistant", "content": analysis, "tag": "📄 명세서 분석"}
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"로드 실패: {e}")
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

        if st.button("초기화"):
            agent.reset()
            st.session_state.req_messages = []
            st.session_state.analyzed_files = set()
            st.session_state.last_suggestions = []
            st.session_state.suggestion_key = 0
            st.rerun()

    for msg in st.session_state.req_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if tag := msg.get("tag"):
                st.caption(tag)
            elif state := msg.get("state"):
                label, color = _STATE_LABELS[state]
                st.caption(f":{color}[{label}]")

    suggestions = st.session_state.last_suggestions
    if suggestions:
        st.markdown("**💡 다음 질문 제안:**")
        cols = st.columns(len(suggestions))
        for j, (col, q) in enumerate(zip(cols, suggestions)):
            if col.button(q, key=f"sq_{st.session_state.suggestion_key}_{j}", use_container_width=True):
                st.session_state.pending_input = q
                st.session_state.last_suggestions = []
                st.session_state.suggestion_key += 1
                st.rerun()

    pending = st.session_state.pop("pending_input", None)
    user_input = pending or st.chat_input("요구사항에 대해 질문하세요...")

    if user_input:
        st.session_state.last_suggestions = []
        st.session_state.req_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                try:
                    response, state = agent.chat(user_input)
                except Exception as e:
                    st.error(f"오류: {e}")
                    st.stop()
            st.markdown(response)
            label, color = _STATE_LABELS[state]
            st.caption(f":{color}[{label}]")

        st.session_state.req_messages.append(
            {"role": "assistant", "content": response, "state": state}
        )

        if not agent.is_wfc_initialized():
            with st.spinner("대화 공간 구성 중..."):
                agent.init_wfc()

        proactive_msg = None
        proactive_tag = None

        if state in (ConversationState.LOOPING, ConversationState.STUCK):
            with st.spinner("Vela 개입 중..."):
                proactive_msg = agent.proactive_nudge(state)
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
            st.session_state.req_messages.append(
                {"role": "assistant", "content": proactive_msg,
                 "state": state, "tag": proactive_tag}
            )
        else:
            with st.spinner("질문 제안 중..."):
                st.session_state.last_suggestions = agent.suggest_questions()
            st.session_state.suggestion_key += 1
            st.rerun()


main()
