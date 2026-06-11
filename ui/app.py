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
        "pension_result": None,
        "active_button": None,
        "voice_mode": False,
        "pending_question": None,
        "ui_mode": None,              # None | "버튼 모드" | "채팅 모드"
        "selected_feature": None,     # 선택한 기능 인덱스 (0~3)
        "chat_messages": [],          # 채팅 모드용 표시 메시지
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


def _extract_pension(history: list):
    """대화 기록에서 연금 최적화 결과 추출 (막대 차트용)"""
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg["content"])
            if "optimal_claim_age" in data:
                return data
        except Exception:
            pass
    return None


def _pension_chart(pension_data: dict, base_monthly: float) -> go.Figure:
    """연금 수령 시기별 월 수령액 비교 막대 차트"""
    # 국민연금 조정 규칙: 조기 -0.5%/월, 연기 +0.7%/월 (기준 65세)
    ages    = [62, 63, 64, 65, 66, 67, 68, 69, 70]
    amounts = []
    for a in ages:
        diff_months = (a - 65) * 12
        if diff_months < 0:
            rate = 1 + 0.005 * diff_months      # 조기 감액
        else:
            rate = 1 + 0.007 * diff_months      # 연기 가산
        amounts.append(round(base_monthly * rate, 1))

    optimal = pension_data.get("optimal_claim_age", 65)
    colors  = ["#E53935" if a == optimal else "#1B4F8A" for a in ages]
    labels  = [f"{a}세" for a in ages]

    fig = go.Figure(go.Bar(
        x=labels,
        y=amounts,
        marker_color=colors,
        text=[f"{v:.0f}만원" for v in amounts],
        textposition="outside",
    ))

    breakeven = pension_data.get("breakeven_age", None)
    title_txt = (
        f"<b>연금 수령 시기별 월 수령액</b>"
        f"  ·  최적: <span style='color:#E53935'>{optimal}세</span>"
    )
    if breakeven:
        title_txt += f"  ·  손익분기: {breakeven:.0f}세"

    fig.update_layout(
        title=dict(text=title_txt, font=dict(size=16)),
        xaxis_title="수령 시작 나이",
        yaxis_title="월 수령액 (만원)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, max(amounts) * 1.2])
    return fig

# ── 모드 선택 ────────────────────────────────────────────────────────────────

def _show_mode_select():
    st.markdown("## 어떤 방식으로 이용하시겠어요?")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#EBF2FF; border:2px solid #1B4F8A; border-radius:16px;
                    padding:32px 20px; text-align:center;'>
            <div style='font-size:48px;'>🔘</div>
            <div style='font-size:22px; font-weight:bold; color:#1B4F8A; margin-top:12px;'>버튼 모드</div>
            <div style='font-size:15px; color:#555; margin-top:10px;'>
                준비된 질문 버튼을 눌러<br>간편하게 확인하는 방식<br><br>
                <b>어르신께 추천</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("버튼 모드 선택", key="select_btn", use_container_width=True):
            st.session_state.ui_mode = "버튼 모드"
            st.rerun()

    with col2:
        st.markdown("""
        <div style='background:#E8F5E9; border:2px solid #2E7D32; border-radius:16px;
                    padding:32px 20px; text-align:center;'>
            <div style='font-size:48px;'>💬</div>
            <div style='font-size:22px; font-weight:bold; color:#2E7D32; margin-top:12px;'>채팅 모드</div>
            <div style='font-size:15px; color:#555; margin-top:10px;'>
                궁금한 것을 자유롭게<br>대화하듯 물어보는 방식<br><br>
                <b>자유로운 질문 가능</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("채팅 모드 선택", key="select_chat", use_container_width=True):
            st.session_state.ui_mode = "채팅 모드"
            st.rerun()


# ── 기능 선택 ────────────────────────────────────────────────────────────────

def _show_feature_select():
    st.markdown("## 어떤 것이 궁금하세요?")
    st.markdown("")

    col1, col2 = st.columns(2)
    grid = [col1, col2, col1, col2]

    features = [
        ("💰", "내 자산\n얼마나 버티나요?",   "#1B4F8A", "#EBF2FF"),
        ("📅", "연금 언제 받는 게\n좋을까요?", "#1B4F8A", "#EBF2FF"),
        ("👥", "또래랑\n비교해주세요",          "#2E7D32", "#E8F5E9"),
        ("📉", "지출 줄이면\n어떻게 되나요?",   "#2E7D32", "#E8F5E9"),
    ]

    for i, (icon, label, fc, bg) in enumerate(features):
        with grid[i]:
            st.markdown(f"""
            <div style='background:{bg}; border:2px solid {fc}; border-radius:16px;
                        padding:28px 16px; text-align:center; margin-bottom:8px;'>
                <div style='font-size:40px;'>{icon}</div>
                <div style='font-size:18px; font-weight:bold; color:{fc};
                            margin-top:10px; white-space:pre-line;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("선택", key=f"feat_{i}", use_container_width=True):
                st.session_state.selected_feature = i
                st.rerun()


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

    # 기능 선택 후 첫 진입 시 자동으로 해당 기능 실행
    if st.session_state.selected_feature is not None and not st.session_state.pending_question and not st.session_state.result_text:
        i = st.session_state.selected_feature
        prompt = MENU[i][2]
        st.session_state.pending_question = _inject_profile(prompt)
        st.session_state.active_button = i

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
                    st.session_state.pension_result = None
                    st.session_state.active_button = i
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
        pension = _extract_pension(updated)
        if pension:
            st.session_state.pension_result = pension
        st.rerun()

    if st.session_state.result_text:
        # 팬 차트 (버튼1·4 — Monte Carlo)
        if st.session_state.sim_result and st.session_state.active_button != 1:
            fig = _fan_chart(p, st.session_state.sim_result)
            st.plotly_chart(fig, use_container_width=True)

        # 연금 막대 차트 (버튼2 — 연금 시기)
        if st.session_state.pension_result and st.session_state.active_button == 1:
            pr = st.session_state.pension_result
            adj = pr.get("adjustment_rate", 1.0)
            monthly_opt = pr.get("monthly_at_optimal", 0)
            # 원 단위로 반환된 경우 만원으로 변환 (1만원 이상이면 원 단위로 판단)
            if monthly_opt > 10000:
                monthly_opt = monthly_opt / 10000
            base_monthly = round(monthly_opt / (1 + adj), 1)
            fig = _pension_chart(pr, base_monthly)
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


def _show_chat():
    """채팅 모드 화면"""
    p   = st.session_state.profile
    age = p.get("age", 65)

    # 대화 기록 표시
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar="🧓" if msg["role"] == "user" else "🤖"):
            if msg.get("chart") == "fan" and msg.get("sim"):
                st.plotly_chart(_fan_chart(p, msg["sim"]), use_container_width=True)
            elif msg.get("chart") == "pension" and msg.get("pension"):
                pr  = msg["pension"]
                adj = pr.get("adjustment_rate", 1.0)
                opt = pr.get("monthly_at_optimal", 0)
                if opt > 10000:
                    opt /= 10000
                st.plotly_chart(_pension_chart(pr, round(opt / (1 + adj), 1)), use_container_width=True)
            st.markdown(msg["content"])

    # 입력창
    user_input = st.chat_input("궁금한 것을 자유롭게 물어보세요")
    if user_input:
        full_q = _inject_profile(user_input)
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="🧓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("계산 중..."):
                reply, updated = run_agent(full_q, st.session_state.conv_history, age=age)
            st.session_state.conv_history = updated

            sim     = _extract_sim(updated)
            pension = _extract_pension(updated)

            ai_msg: dict = {"role": "assistant", "content": reply}
            if sim:
                ai_msg["chart"] = "fan"
                ai_msg["sim"]   = sim
                st.plotly_chart(_fan_chart(p, sim), use_container_width=True)
            elif pension:
                pr  = pension
                adj = pr.get("adjustment_rate", 1.0)
                opt = pr.get("monthly_at_optimal", 0)
                if opt > 10000:
                    opt /= 10000
                ai_msg["chart"]   = "pension"
                ai_msg["pension"] = pension
                st.plotly_chart(_pension_chart(pr, round(opt / (1 + adj), 1)), use_container_width=True)

            st.markdown(reply)
            st.session_state.chat_messages.append(ai_msg)


def _inject_profile(question: str) -> str:
    p = st.session_state.profile
    profile_str = (
        f"[사용자 정보: 나이={p.get('age')}세, "
        f"총자산={p.get('assets',0)//10000:,}만원, "
        f"월지출={p.get('monthly_expense',0)//10000}만원, "
        f"월연금={p.get('pension',0)//10000}만원"
        f"(수령시작{p.get('pension_start_age')}세). "
        f"추가 정보를 묻지 말고 이 정보로 바로 계산해주세요.]"
    )
    return f"{profile_str}\n\n{question}"

# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    _init()
    age = st.session_state.profile.get("age", 65)
    _apply_css(age)

    with st.sidebar:
        st.markdown("## LifeLong WM")
        st.divider()

        # UI 모드 전환 (모드 선택 완료 후에만 표시)
        if st.session_state.ui_mode is not None:
            ui_mode = st.radio(
                "화면 방식",
                ["버튼 모드", "채팅 모드"],
                index=0 if st.session_state.ui_mode == "버튼 모드" else 1,
            )
            if ui_mode != st.session_state.ui_mode:
                st.session_state.ui_mode = ui_mode
                st.rerun()

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
                "pension_result": None, "active_button": None,
                "pending_question": None, "chat_messages": [],
                "ui_mode": None, "selected_feature": None,
            })
            st.rerun()

        if age >= 75:
            st.info("모드: 후기 고령층")
        elif age >= 65:
            st.info("모드: 초기 고령층")
        else:
            st.info("모드: 은퇴 준비층")

    st.markdown("# 내 노후 재무 AI 동반자")

    if st.session_state.ui_mode is None:
        _show_mode_select()
    elif st.session_state.ui_mode == "버튼 모드" and st.session_state.selected_feature is None:
        _show_feature_select()
    elif st.session_state.step < len(ONBOARDING):
        _show_onboarding()
    elif st.session_state.ui_mode == "채팅 모드":
        _show_chat()
    else:
        _show_main()


if __name__ == "__main__":
    main()
