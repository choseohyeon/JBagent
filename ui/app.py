"""
Streamlit 메인 앱
- Slow Banking UI (연령별 글씨 크기, 고대비)
- 단계별 온보딩 → 버튼 기반 메인 화면
- LLM은 뒤에서 작동, 사용자는 버튼만 누름
- 음성 모드 옵션 (토글)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import streamlit as st
import plotly.graph_objects as go
from agent.agent import run_agent

st.set_page_config(page_title="LifeLong WM", layout="centered")

# ── 온보딩 질문 ───────────────────────────────────────────────────────────────

ONBOARDING = [
    ("age",              "나이가 어떻게 되세요?",                      "세"),
    ("assets",           "현재 갖고 계신 자산이 얼마나 되세요?",        "만원"),
    ("monthly_expense",  "한 달에 생활비를 얼마나 쓰시나요?",           "만원"),
    ("pension",          "국민연금은 한 달에 얼마나 받으실 예정인가요?", "만원"),
    ("pension_start_age","연금은 몇 살부터 받으실 예정인가요?",          "세"),
]

# ── 버튼 메뉴 ────────────────────────────────────────────────────────────────

MENU = [
    ("💰", "내 자산 얼마나\n버티나요?",
     "내 자산이 얼마나 버티는지 Monte Carlo 시뮬레이션으로 계산해주세요."),
    ("📅", "연금 언제 받는 게\n좋을까요?",
     "국민연금을 언제 받는 게 가장 유리한지 비교해주세요."),
    ("👥", "또래랑\n비교해주세요",
     "비슷한 나이와 자산을 가진 또래와 비교해서 내 재무 상태가 어느 정도인지 알려주세요."),
    ("📉", "지출 줄이면\n어떻게 되나요?",
     "월 지출을 20% 줄이면 자산이 얼마나 더 오래 버티는지 계산해주세요."),
]

# ── 세션 초기화 ───────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "step": 0,
        "profile": {},
        "conv_history": [],
        "result_text": None,
        "sim_result": None,
        "voice_mode": False,
        "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────

def _apply_css(age: int = 65):
    size = "22px" if age >= 75 else "20px" if age >= 65 else "17px"
    st.markdown(f"""
    <style>
        .main .block-container {{ max-width: 820px; }}
        p, li, .stMarkdown {{ font-size: {size} !important; line-height: 1.9; }}
        div[data-testid="stChatMessage"] {{ font-size: {size} !important; }}
        .menu-btn button {{
            font-size: 18px !important;
            padding: 20px 12px !important;
            border-radius: 16px !important;
            background-color: #1B4F8A !important;
            color: white !important;
            border: none !important;
            width: 100% !important;
            height: 110px !important;
            white-space: pre-wrap !important;
            line-height: 1.5 !important;
        }}
        .stButton > button {{
            font-size: {size} !important;
            border-radius: 10px;
        }}
        .stTextInput input {{ font-size: {size} !important; padding: 12px; }}
        .result-card {{
            background: #f0f4ff;
            border-left: 6px solid #1B4F8A;
            border-radius: 12px;
            padding: 20px 24px;
            margin: 16px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

# ── TTS ───────────────────────────────────────────────────────────────────────

def _speak(text: str):
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="ko")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            st.audio(f.name, autoplay=True)
    except Exception:
        pass

# ── STT ───────────────────────────────────────────────────────────────────────

def _transcribe(audio_bytes: bytes) -> str:
    try:
        import whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            model = whisper.load_model("tiny")
            return model.transcribe(f.name, language="ko")["text"].strip()
    except Exception:
        return ""

# ── 팬 차트 ───────────────────────────────────────────────────────────────────

def _fan_chart(profile: dict, sim: dict) -> go.Figure:
    assets        = profile.get("assets", 300_000_000)
    monthly_exp   = profile.get("monthly_expense", 2_000_000)
    pension       = profile.get("pension", 800_000)
    pension_start = profile.get("pension_start_age", 65)
    current_age   = profile.get("age", 65)
    years         = list(range(0, 41))

    def path(ret, inf):
        a, r = assets, [assets]
        for y in years[1:]:
            income  = pension * 12 if (current_age + y) >= pension_start else 0
            expense = monthly_exp * 12 * ((1 + inf) ** y)
            a = max(0.0, a * (1 + ret) + income - expense)
            r.append(a)
        return r

    p90 = path(0.055, 0.015)
    p50 = path(0.030, 0.025)
    p10 = path(0.005, 0.040)
    lbl = [f"{current_age + y}세" for y in years]
    eok = lambda v: [x / 1e8 for x in v]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lbl, y=eok(p90), name="낙관적",
                             line=dict(color="#4A9EDB", dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=lbl, y=eok(p10), name="비관적",
                             line=dict(color="#4A9EDB", dash="dot", width=1.5),
                             fill="tonexty", fillcolor="rgba(74,158,219,0.15)"))
    fig.add_trace(go.Scatter(x=lbl, y=eok(p50), name="중간 예상",
                             line=dict(color="#1B4F8A", width=3)))
    fig.add_hline(y=0, line_color="red", line_width=1.5,
                  annotation_text="자산 고갈선", annotation_position="right")

    depletion = int(sim.get("depletion_probability", 0) * 100)
    fig.update_layout(
        title=dict(text=f"<b>자산 예측 경로</b>  ·  고갈 확률: "
                        f"<span style='color:red;font-weight:bold'>{depletion}%</span>",
                   font=dict(size=17)),
        xaxis_title="나이", yaxis_title="자산 (억원)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        height=400, margin=dict(l=20, r=20, t=60, b=90),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-45, dtick=5)
    return fig

def _extract_sim(history: list):
    """대화 기록에서 monte carlo 결과 추출 (팬 차트용)"""
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg["content"])
            if "depletion_prob" in data:
                return {"depletion_probability": data["depletion_prob"]}
        except Exception:
            pass
    return None

# ── 온보딩 ────────────────────────────────────────────────────────────────────

def _show_onboarding():
    step = st.session_state.step
    key, question, unit = ONBOARDING[step]

    st.markdown(f"### {question}")
    if unit != "세":
        st.caption(f"{unit} 단위로 입력해주세요")

    if st.session_state.voice_mode:
        audio_file = st.file_uploader("음성 파일 (wav/mp3/m4a)", type=["wav","mp3","m4a"],
                                       key=f"audio_{step}", label_visibility="collapsed")
        if audio_file:
            with st.spinner("음성 인식 중..."):
                text = _transcribe(audio_file.read())
            if text:
                st.info(f"인식된 내용: **{text}**")
                import re
                nums = re.findall(r"\d+", text.replace(",", ""))
                if nums:
                    _save_step(key, int(nums[0]))
        st.markdown("또는 직접 입력:")

    val = st.text_input("입력", key=f"input_{step}", label_visibility="collapsed",
                        placeholder=f"숫자만 입력 (단위: {unit})")
    if st.button("다음 →", key=f"btn_{step}"):
        cleaned = val.strip().replace(",", "")
        if cleaned.isdigit():
            _save_step(key, int(cleaned))
        else:
            st.warning("숫자만 입력해주세요.")

    st.progress(step / len(ONBOARDING))
    st.caption(f"{step + 1} / {len(ONBOARDING)} 단계")


def _save_step(key, value):
    mult = 10000 if key in ("assets", "monthly_expense", "pension") else 1
    st.session_state.profile[key] = value * mult
    st.session_state.step += 1
    st.rerun()

# ── 메인 화면 (버튼 기반) ─────────────────────────────────────────────────────

def _show_main():
    p = st.session_state.profile
    age     = p.get("age", 65)
    assets  = p.get("assets", 0) // 10000
    expense = p.get("monthly_expense", 0) // 10000
    pension = p.get("pension", 0) // 10000

    # 프로필 요약 카드
    st.markdown(f"""
    <div style='background:#EBF2FF; border-radius:14px; padding:18px 24px; margin-bottom:24px;'>
        <div style='font-size:20px; font-weight:bold; color:#1B4F8A;'>
            {age}세 · 자산 {assets:,}만원
        </div>
        <div style='font-size:17px; color:#444; margin-top:6px;'>
            월 지출 {expense}만원 &nbsp;|&nbsp; 월 연금 {pension}만원
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 버튼 그리드 (2x2)
    col1, col2 = st.columns(2)
    buttons = [col1, col2, col1, col2]

    for i, (icon, label, prompt) in enumerate(MENU):
        with buttons[i]:
            with st.container():
                st.markdown('<div class="menu-btn">', unsafe_allow_html=True)
                if st.button(f"{icon}  {label}", key=f"menu_{i}"):
                    st.session_state.pending_question = _inject_profile(prompt)
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # 결과 표시
    if st.session_state.pending_question and not st.session_state.result_text:
        with st.spinner("계산하는 중..."):
            reply, updated = run_agent(
                st.session_state.pending_question,
                st.session_state.conv_history,
                age=age,
            )
        st.session_state.conv_history = updated
        st.session_state.result_text = reply
        sim = _extract_sim(updated)
        if sim:
            st.session_state.sim_result = sim
        st.rerun()

    if st.session_state.result_text:
        # 팬 차트
        if st.session_state.sim_result:
            fig = _fan_chart(p, st.session_state.sim_result)
            st.plotly_chart(fig, use_container_width=True)

        # 결과 카드
        st.markdown("---")
        st.markdown(st.session_state.result_text)

        if st.session_state.voice_mode:
            _speak(st.session_state.result_text)

    # 더 물어보기 (접힌 상태)
    with st.expander("💬 더 물어보기"):
        if st.session_state.voice_mode:
            audio_file = st.file_uploader("음성 파일", type=["wav","mp3","m4a"],
                                           key="extra_audio", label_visibility="collapsed")
            if audio_file:
                with st.spinner("음성 인식 중..."):
                    q = _transcribe(audio_file.read())
                if q:
                    st.info(f"인식된 내용: **{q}**")
                    st.session_state.pending_question = _inject_profile(q)
                    st.session_state.result_text = None
                    st.rerun()

        user_q = st.text_input("질문을 입력하세요", key="extra_input",
                               label_visibility="collapsed",
                               placeholder="예: 지출을 150만원으로 줄이면 어떻게 되나요?")
        if st.button("질문하기", key="extra_btn"):
            if user_q.strip():
                st.session_state.pending_question = _inject_profile(user_q)
                st.session_state.result_text = None
                st.rerun()


def _inject_profile(question: str) -> str:
    if st.session_state.conv_history:
        return question
    p = st.session_state.profile
    return (
        f"[내 정보: 나이={p.get('age')}세, "
        f"총자산={p.get('assets',0)//10000:,}만원, "
        f"월지출={p.get('monthly_expense',0)//10000}만원, "
        f"월연금={p.get('pension',0)//10000}만원"
        f"(수령시작{p.get('pension_start_age')}세)]\n\n{question}"
    )

# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    _init()
    age = st.session_state.profile.get("age", 65)
    _apply_css(age)

    with st.sidebar:
        st.markdown("## LifeLong WM")
        st.divider()

        voice = st.toggle("🎤 음성 모드", value=st.session_state.voice_mode)
        if voice != st.session_state.voice_mode:
            st.session_state.voice_mode = voice
            st.rerun()

        if voice:
            st.success("음성 입출력 켜짐")
        else:
            st.info("텍스트 모드 (기본)")

        st.divider()
        if st.button("처음부터 다시"):
            st.session_state.update({
                "step": 0, "profile": {}, "conv_history": [],
                "result_text": None, "sim_result": None,
                "pending_question": None,
            })
            st.rerun()

        if age >= 75:
            st.info("모드: 후기 고령층")
        elif age >= 65:
            st.info("모드: 초기 고령층")
        else:
            st.info("모드: 은퇴 준비층")

    st.markdown("# 내 노후 재무 AI 동반자")

    if st.session_state.step < len(ONBOARDING):
        _show_onboarding()
    else:
        _show_main()


if __name__ == "__main__":
    main()
