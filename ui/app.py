"""
Streamlit 메인 앱
- Slow Banking UI (큰 글씨, 단순 구조)
- 팬 차트 시각화 (불확실성 표현)
- Agent 대화 인터페이스
"""

import streamlit as st
from agent.agent import run_agent


st.set_page_config(
    page_title="LifeLong WM",
    page_icon="🏦",
    layout="centered",
)

# Slow Banking UI 스타일
st.markdown("""
<style>
    * { font-size: 18px !important; }
    .stButton button { font-size: 20px !important; padding: 12px 24px; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("내 노후 재무 AI 동반자")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 사용자 입력
    if prompt := st.chat_input("궁금한 것을 물어보세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # response = run_agent(prompt, st.session_state.messages)
        # st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
