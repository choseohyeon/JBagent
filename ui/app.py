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
import streamlit as st
import plotly.graph_objects as go
from agent.agent import run_agent

st.set_page_config(page_title="LifeLong WM", layout="centered")

# ── 온보딩 질문 ───────────────────────────────────────────────────────────────

ONBOARDING = [
    ("age",              "나이가 어떻게 되세요?",                            "세"),
    ("monthly_income",   "한 달 근로소득(월급·사업소득)이 얼마나 되세요?",   "만원"),
    ("assets",           "현재 갖고 계신 자산이 얼마나 되세요?",              "만원"),
    ("monthly_expense",  "한 달에 생활비를 얼마나 쓰시나요?",                "만원"),
    ("pension",          "국민연금은 한 달에 얼마나 받으실 예정인가요?",      "만원"),
    ("pension_start_age","연금은 몇 살부터 받으실 예정인가요?",               "세"),
]

# ── 구간 선택 옵션 (잘 모르는 사용자용) ──────────────────────────────────────────

RANGES = {
    "monthly_income": [
        ("없음 (무직)",   0),
        ("100만원 미만",  70),
        ("100~200만원",  150),
        ("200~300만원",  250),
        ("300~500만원",  400),
        ("500만원 이상", 600),
    ],
    "assets": [
        ("5천만원 미만",  3000),
        ("5천만~1억",    7500),
        ("1억~3억",     20000),
        ("3억~5억",     40000),
        ("5억~10억",    75000),
        ("10억 이상",  150000),
    ],
    "monthly_expense": [
        ("100만원 미만",  70),
        ("100~150만원",  125),
        ("150~200만원",  175),
        ("200~300만원",  250),
        ("300만원 이상", 350),
    ],
    "pension": [
        ("없음 / 모름",    0),
        ("50만원 미만",   30),
        ("50~100만원",   75),
        ("100만원 이상", 120),
    ],
    "pension_start_age": [
        ("60세 (조기)",  60),
        ("62세",         62),
        ("63세",         63),
        ("65세 (기본)", 65),
        ("68세 이상",   68),
    ],
}

# ── 버튼 메뉴 ────────────────────────────────────────────────────────────────

MENU = [
    ("내 자산 얼마나\n버티나요?",
     "내 자산이 얼마나 버티는지 Monte Carlo 시뮬레이션으로 계산해주세요."),
    ("연금 언제 받는 게\n좋을까요?",
     "국민연금을 언제 받는 게 가장 유리한지 비교해주세요."),
    ("또래랑\n비교해주세요",
     "비슷한 나이와 자산을 가진 또래와 비교해서 내 재무 상태가 어느 정도인지 알려주세요."),
    ("지출 줄이면\n어떻게 되나요?",
     "월 지출을 20% 줄이면 자산이 얼마나 더 오래 버티는지 계산해주세요."),
]

# ── 세션 초기화 ───────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "step": 0,
        "profile": {},
        "profile_labels": {},         # 구간 선택 시 원래 레이블 저장
        "conv_history": [],
        "result_text": None,
        "sim_result": None,
        "pension_result": None,
        "active_button": None,
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
    size = "20px" if age >= 75 else "18px" if age >= 65 else "16px"
    st.markdown(f"""
    <style>
        .main .block-container {{ max-width: 720px; padding-top: 1.5rem; }}
        p, li, .stMarkdown {{ font-size: {size} !important; line-height: 1.75; color: #111; }}
        div[data-testid="stChatMessage"] {{ font-size: {size} !important; }}

        /* 모든 버튼 기본: 아웃라인 스타일 */
        .stButton > button {{
            font-size: {size} !important;
            border-radius: 6px !important;
            border: 1.5px solid #D0D5DD !important;
            background: #fff !important;
            color: #111 !important;
            font-weight: 400 !important;
            transition: border-color 0.15s !important;
        }}
        .stButton > button:hover {{
            border-color: #1B4F8A !important;
            color: #1B4F8A !important;
        }}

        /* 메인 메뉴 버튼: 더 크게 */
        .menu-btn button {{
            height: 80px !important;
            white-space: pre-wrap !important;
            line-height: 1.5 !important;
            font-size: {size} !important;
            font-weight: 500 !important;
        }}

        /* 프라이머리 액션 (다음, 선택 등) */
        .btn-primary button {{
            background: #1B4F8A !important;
            color: #fff !important;
            border-color: #1B4F8A !important;
            font-weight: 500 !important;
        }}

        .stTextInput input {{ font-size: {size} !important; padding: 10px 12px; border-radius: 6px; }}
        .stProgress > div > div {{ background-color: #1B4F8A; }}
        hr {{ border-color: #F0F0F0; }}
    </style>
    """, unsafe_allow_html=True)


# ── 팬 차트 ───────────────────────────────────────────────────────────────────

INCOME_UNTIL_AGE = 60  # 근로소득 종료 나이 (시뮬레이션 가정)

def _fan_chart(profile: dict, sim: dict) -> go.Figure:
    assets         = profile.get("assets", 300_000_000)
    monthly_exp    = profile.get("monthly_expense", 2_000_000)
    pension        = profile.get("pension", 800_000)
    pension_start  = profile.get("pension_start_age", 65)
    monthly_income = profile.get("monthly_income", 0)
    current_age    = profile.get("age", 65)
    years          = list(range(0, 41))

    def path(ret, inf):
        a, r = assets, [assets]
        for y in years[1:]:
            age_now    = current_age + y
            work_inc   = monthly_income * 12 if age_now < INCOME_UNTIL_AGE else 0
            pension_yr = pension * 12 if age_now >= pension_start else 0
            expense    = monthly_exp * 12 * ((1 + inf) ** y)
            a = max(0.0, a * (1 + ret) + work_inc + pension_yr - expense)
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
    st.markdown("### 이용 방식을 선택해주세요")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**버튼 모드**")
        st.caption("준비된 항목 중에서 골라 확인하는 방식")
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("버튼 모드로 시작", key="select_btn", use_container_width=True):
            st.session_state.ui_mode = "버튼 모드"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**채팅 모드**")
        st.caption("궁금한 것을 자유롭게 대화하는 방식")
        if st.button("채팅 모드로 시작", key="select_chat", use_container_width=True):
            st.session_state.ui_mode = "채팅 모드"
            st.rerun()


# ── 기능 선택 ────────────────────────────────────────────────────────────────

def _show_feature_select():
    st.markdown("### 무엇이 궁금하세요?")
    st.markdown("")

    features = [
        ("내 자산 얼마나 버티나요?",     "Monte Carlo 시뮬레이션으로 자산 수명과 고갈 확률 계산"),
        ("연금 언제 받는 게 좋을까요?",   "수령 시기별 총 수령액·손익분기 연령 비교"),
        ("또래랑 비교해주세요",            "비슷한 처지 가구 대비 내 자산 순위"),
        ("지출 줄이면 어떻게 되나요?",     "지출 20% 절감 시 자산 수명 재시뮬레이션"),
    ]

    for i, (title, desc) in enumerate(features):
        col_txt, col_btn = st.columns([5, 1])
        with col_txt:
            st.markdown(f"**{title}**")
            st.caption(desc)
        with col_btn:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            if st.button("선택", key=f"feat_{i}", use_container_width=True):
                st.session_state.selected_feature = i
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.divider()


# ── 온보딩 ────────────────────────────────────────────────────────────────────

def _show_onboarding():
    step = st.session_state.step
    key, question, unit = ONBOARDING[step]

    st.markdown(f"### {question}")

    ranges = RANGES.get(key)

    if key == "monthly_income":
        st.caption("근로소득·사업소득·임대소득 합산. 없으면 '없음' 선택. (60세까지 유지된다고 가정합니다)")

    if ranges:
        # ── 구간 선택 버튼 ────────────────────────────────────────────
        st.markdown("**아래에서 가장 가까운 구간을 골라주세요**")
        ncols = 3 if key in ("assets", "pension_start_age") else 2
        cols = st.columns(ncols)
        for idx, (label, val) in enumerate(ranges):
            with cols[idx % ncols]:
                if st.button(label, key=f"range_{step}_{idx}", use_container_width=True):
                    _save_step(key, val, label=label)

        st.markdown("---")
        st.markdown("**정확한 금액을 알고 계신 분은 직접 입력하세요**")
        st.caption(f"{unit} 단위 숫자만 입력")
    else:
        # age 항목: 직접 입력만
        st.caption(f"{unit} 단위로 입력해주세요")

    val = st.text_input("입력", key=f"input_{step}", label_visibility="collapsed",
                        placeholder=f"숫자만 입력 (단위: {unit})")

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← 이전", key=f"back_{step}"):
            if step == 0:
                if st.session_state.ui_mode == "채팅 모드":
                    st.session_state.ui_mode = None
                else:
                    st.session_state.selected_feature = None
            else:
                st.session_state.step -= 1
            st.rerun()
    with col_next:
        if st.button("다음 →", key=f"btn_{step}", use_container_width=True):
            cleaned = val.strip().replace(",", "")
            if cleaned.isdigit():
                _save_step(key, int(cleaned))
            else:
                st.warning("숫자만 입력해주세요.")

    st.progress(step / len(ONBOARDING))
    st.caption(f"{step + 1} / {len(ONBOARDING)} 단계")


def _save_step(key, value, label: str = None):
    mult = 10000 if key in ("assets", "monthly_expense", "pension", "monthly_income") else 1
    st.session_state.profile[key] = value * mult
    if label is not None:
        st.session_state.profile_labels[key] = label
    elif key in st.session_state.profile_labels:
        # 직접 입력 시 레이블 제거
        del st.session_state.profile_labels[key]
    st.session_state.step += 1
    st.rerun()

# ── 메인 화면 (버튼 기반) ─────────────────────────────────────────────────────

def _show_main():
    p      = st.session_state.profile
    lbl    = st.session_state.get("profile_labels", {})
    age    = p.get("age", 65)
    income = p.get("monthly_income", 0) // 10000

    # 구간 선택이면 레이블, 직접 입력이면 숫자
    assets_disp  = lbl.get("assets",          f"{p.get('assets',0)//10000:,}만원")
    expense_disp = lbl.get("monthly_expense", f"{p.get('monthly_expense',0)//10000}만원")
    pension_disp = lbl.get("pension",         f"{p.get('pension',0)//10000}만원")
    income_disp  = lbl.get("monthly_income",  f"{income}만원" if income > 0 else "없음")

    # 기능 선택 후 첫 진입 시 자동으로 해당 기능 실행
    if st.session_state.selected_feature is not None and not st.session_state.pending_question and not st.session_state.result_text:
        i = st.session_state.selected_feature
        prompt = MENU[i][1]
        st.session_state.pending_question = _inject_profile(prompt)
        st.session_state.active_button = i

    # 프로필 요약
    income_line = f"월 소득 {income_disp}　" if income > 0 or "monthly_income" in lbl else ""
    st.markdown(f"""
    <div style='border:1px solid #E0E0E0; border-radius:8px; padding:14px 18px; margin-bottom:20px;'>
        <div style='font-size:18px; font-weight:600; color:#111;'>{age}세 · 자산 {assets_disp}</div>
        <div style='font-size:15px; color:#666; margin-top:4px;'>
            {income_line}월 지출 {expense_disp}　월 연금 {pension_disp}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 버튼 그리드 (2x2)
    col1, col2 = st.columns(2)
    buttons = [col1, col2, col1, col2]

    for i, (label, prompt) in enumerate(MENU):
        with buttons[i]:
            with st.container():
                st.markdown('<div class="menu-btn">', unsafe_allow_html=True)
                if st.button(label, key=f"menu_{i}", use_container_width=True):
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
            if monthly_opt > 10000:
                monthly_opt = monthly_opt / 10000
            base_monthly = round(monthly_opt / (1 + adj), 1)
            fig = _pension_chart(pr, base_monthly)
            st.plotly_chart(fig, use_container_width=True)

        # 결과 카드
        st.markdown("---")
        st.markdown(st.session_state.result_text)

        # ── 고위험 시 개선 방안 버튼 ─────────────────────────────────────────
        sim = st.session_state.sim_result
        if sim and sim.get("depletion_probability", 0) >= 0.5:
            depletion_pct = int(sim["depletion_probability"] * 100)
            st.markdown(f"""
            <div style='border-left:3px solid #C0392B; padding:10px 16px; margin:16px 0;
                        background:#FFF8F8; border-radius:0 6px 6px 0;'>
                <span style='color:#C0392B; font-weight:600;'>자산 고갈 확률 {depletion_pct}%</span>
                <span style='color:#555; font-size:14px;'> — 아래 방법으로 개선할 수 있는지 확인해보세요.</span>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("지출 20% 줄이면?", key="advice_expense", use_container_width=True):
                    expense_reduced = p.get("monthly_expense", 0) * 0.8
                    new_p = dict(p)
                    new_p["monthly_expense"] = expense_reduced
                    q = _inject_profile_custom(
                        new_p,
                        f"월 지출을 {int(expense_reduced//10000)}만원으로 줄이면 자산이 얼마나 더 버티나요?"
                    )
                    st.session_state.pending_question = q
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.session_state.pension_result = None
                    st.rerun()
            with col2:
                if st.button("소득 유지 기간 늘리면?", key="advice_income", use_container_width=True):
                    st.session_state.pending_question = _inject_profile(
                        "근로소득을 65세까지 유지하면 자산 고갈 확률이 어떻게 달라지나요? income_until_age=65로 재계산해주세요."
                    )
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.session_state.pension_result = None
                    st.rerun()
            with col3:
                if st.button("자산 배분 최적화하면?", key="advice_portfolio", use_container_width=True):
                    st.session_state.pending_question = _inject_profile(
                        "현재 상황에서 자산 배분을 최적화하면 위험을 얼마나 줄일 수 있나요? 보수적 포트폴리오로 계산해주세요."
                    )
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.session_state.pension_result = None
                    st.rerun()

    # 더 물어보기 (접힌 상태)
    with st.expander("더 물어보기"):
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
    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"], avatar="🧓" if msg["role"] == "user" else "🤖"):
            if msg.get("chart") == "fan" and msg.get("sim"):
                st.plotly_chart(_fan_chart(p, msg["sim"]), use_container_width=True, key=f"chat_fan_{idx}")
            elif msg.get("chart") == "pension" and msg.get("pension"):
                pr  = msg["pension"]
                adj = pr.get("adjustment_rate", 1.0)
                opt = pr.get("monthly_at_optimal", 0)
                if opt > 10000:
                    opt /= 10000
                st.plotly_chart(_pension_chart(pr, round(opt / (1 + adj), 1)), use_container_width=True, key=f"chat_pension_{idx}")
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
                prev_len = len(st.session_state.conv_history)
                reply, updated = run_agent(full_q, st.session_state.conv_history, age=age)
            st.session_state.conv_history = updated

            # 현재 턴에서 새로 추가된 메시지만 검색 (이전 기록 재사용 방지)
            current_turn = updated[prev_len:]
            sim     = _extract_sim(current_turn)
            pension = _extract_pension(current_turn)

            new_idx = len(st.session_state.chat_messages)
            ai_msg: dict = {"role": "assistant", "content": reply}
            if sim:
                ai_msg["chart"] = "fan"
                ai_msg["sim"]   = sim
                st.plotly_chart(_fan_chart(p, sim), use_container_width=True, key=f"new_fan_{new_idx}")
            elif pension:
                pr  = pension
                adj = pr.get("adjustment_rate", 1.0)
                opt = pr.get("monthly_at_optimal", 0)
                if opt > 10000:
                    opt /= 10000
                ai_msg["chart"]   = "pension"
                ai_msg["pension"] = pension
                st.plotly_chart(_pension_chart(pr, round(opt / (1 + adj), 1)), use_container_width=True, key=f"new_pension_{new_idx}")

            st.markdown(reply)
            st.session_state.chat_messages.append(ai_msg)


def _inject_profile_custom(p: dict, question: str) -> str:
    """커스텀 프로필 dict로 질문 주입 (시뮬레이션 수정값 적용 시 — 레이블 없이 숫자만)"""
    income_val = p.get('monthly_income', 0) // 10000
    income_txt = f"월근로소득={income_val}만원(60세까지), " if income_val > 0 else "근로소득=없음, "
    profile_str = (
        f"[사용자 정보: 나이={p.get('age')}세, "
        f"{income_txt}"
        f"총자산={p.get('assets',0)//10000:,}만원, "
        f"월지출={p.get('monthly_expense',0)//10000}만원, "
        f"월연금={p.get('pension',0)//10000}만원"
        f"(수령시작{p.get('pension_start_age')}세). "
        f"추가 정보를 묻지 말고 이 정보로 바로 계산해주세요.]"
    )
    return f"{profile_str}\n\n{question}"


def _fmt(p: dict, lbl: dict, key: str, unit: str = "만원") -> str:
    """계산값은 그대로 두되, 구간 선택 시 범위 표현 함께 전달"""
    num = p.get(key, 0) // (10000 if unit == "만원" else 1)
    label = lbl.get(key)
    if label:
        return f"{num}{unit}(사용자가 선택한 구간: {label})"
    return f"{num}{unit}"


def _inject_profile(question: str) -> str:
    p   = st.session_state.profile
    lbl = st.session_state.get("profile_labels", {})

    income_val = p.get('monthly_income', 0) // 10000
    if income_val > 0 or "monthly_income" in lbl:
        income_txt = f"월근로소득={_fmt(p, lbl, 'monthly_income')}(60세까지), "
    else:
        income_txt = "근로소득=없음, "

    profile_str = (
        f"[사용자 정보: 나이={p.get('age')}세, "
        f"{income_txt}"
        f"총자산={_fmt(p, lbl, 'assets')}, "
        f"월지출={_fmt(p, lbl, 'monthly_expense')}, "
        f"월연금={_fmt(p, lbl, 'pension')}"
        f"(수령시작{p.get('pension_start_age')}세). "
        f"계산은 괄호 안의 구체적인 수치를 사용하고, "
        f"대화에서는 사용자가 선택한 구간 표현을 우선 사용하세요. "
        f"추가 정보를 묻지 말고 바로 계산해주세요.]"
    )
    return f"{profile_str}\n\n{question}"

# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    _init()
    age = st.session_state.profile.get("age", 65)
    _apply_css(age)

    with st.sidebar:
        st.markdown("**LifeLong WM**")
        st.divider()

        if st.session_state.ui_mode is not None:
            ui_mode = st.radio(
                "화면 방식",
                ["버튼 모드", "채팅 모드"],
                index=0 if st.session_state.ui_mode == "버튼 모드" else 1,
                label_visibility="collapsed",
            )
            if ui_mode != st.session_state.ui_mode:
                st.session_state.ui_mode = ui_mode
                st.rerun()
            st.divider()

        if st.button("처음부터 다시", use_container_width=True):
            st.session_state.update({
                "step": 0, "profile": {}, "profile_labels": {}, "conv_history": [],
                "result_text": None, "sim_result": None,
                "pension_result": None, "active_button": None,
                "pending_question": None, "chat_messages": [],
                "ui_mode": None, "selected_feature": None,
            })
            st.rerun()

        st.divider()
        segment = "후기 고령층" if age >= 75 else "초기 고령층" if age >= 65 else "은퇴 준비층"
        st.caption(segment)

    st.markdown("## LifeLong WM")

    if st.session_state.ui_mode is None:
        _show_mode_select()
    elif st.session_state.ui_mode == "채팅 모드":
        if st.session_state.step < len(ONBOARDING):
            _show_onboarding()
        else:
            _show_chat()
    else:  # 버튼 모드
        if st.session_state.selected_feature is None:
            _show_feature_select()
        elif st.session_state.step < len(ONBOARDING):
            _show_onboarding()
        else:
            _show_main()


if __name__ == "__main__":
    main()
