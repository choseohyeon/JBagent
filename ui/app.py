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
    ("monthly_income",   "한 달 근로소득(월급,사업소득)이 얼마나 되세요?",   "만원"),
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
        "chat_init_attempted": False, # 채팅 첫 자동 실행 여부 (무한루프 방지)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────

def _apply_css(age: int = 65):
    size = "20px" if age >= 75 else "18px" if age >= 65 else "16px"
    st.markdown(f"""
    <style>
        .main .block-container {{ max-width: 560px; padding-top: 0.6rem; }}
        p, li, .stMarkdown {{ font-size: {size} !important; line-height: 1.8; color: #1A2744; }}
        div[data-testid="stChatMessage"] {{ font-size: {size} !important; }}

        /* ── 서비스 헤더 ── */
        .app-header {{
            text-align: center;
            padding: 28px 0 20px;
            margin-bottom: 8px;
        }}
        .app-header-title {{
            font-size: 26px;
            font-weight: 800;
            color: #1B4F8A;
            letter-spacing: -0.5px;
        }}
        .app-header-sub {{
            font-size: 13px;
            color: #8FA3BE;
            margin-top: 4px;
        }}

        /* ── 기본 버튼 ── */
        .stButton > button {{
            font-size: {size} !important;
            border-radius: 10px !important;
            border: 1.5px solid #D5E0EF !important;
            background: #FFFFFF !important;
            color: #1A2744 !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
            box-shadow: 0 1px 3px rgba(27,79,138,0.06) !important;
        }}
        .stButton > button:hover {{
            border-color: #1B4F8A !important;
            color: #1B4F8A !important;
            background: #F0F6FF !important;
            box-shadow: 0 3px 10px rgba(27,79,138,0.12) !important;
        }}

        /* ── 모드 선택 카드 (클릭 가능한 전체 영역) ── */
        .mode-card button {{
            height: 120px !important;
            white-space: pre-wrap !important;
            line-height: 1.8 !important;
            font-size: {size} !important;
            font-weight: 600 !important;
            background: #FFFFFF !important;
            border: 1.5px solid #D5E3F5 !important;
            color: #1A2744 !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 10px rgba(27,79,138,0.06) !important;
        }}
        .mode-card button:hover {{
            background: #F0F6FF !important;
            border-color: #1B4F8A !important;
            color: #1B4F8A !important;
            box-shadow: 0 4px 16px rgba(27,79,138,0.13) !important;
        }}
        .mode-card-active button {{
            background: #EBF3FF !important;
            border-color: #1B4F8A !important;
            color: #1B4F8A !important;
        }}

        /* ── 메인 메뉴 버튼 2×2 ── */
        .menu-btn button {{
            height: 100px !important;
            white-space: pre-wrap !important;
            line-height: 1.7 !important;
            font-size: {size} !important;
            font-weight: 600 !important;
            background: #F7FAFF !important;
            border: 1.5px solid #C8D8F0 !important;
            color: #1A2744 !important;
            border-radius: 14px !important;
        }}
        .menu-btn button:hover {{
            background: #E4EFFF !important;
            border-color: #1B4F8A !important;
            color: #1B4F8A !important;
            box-shadow: 0 4px 14px rgba(27,79,138,0.13) !important;
        }}
        .menu-btn-active button {{
            background: #1B4F8A !important;
            color: #FFFFFF !important;
            border-color: #1B4F8A !important;
        }}

        /* ── 기능 선택 카드 ── */
        .feature-card button {{
            height: 90px !important;
            white-space: pre-wrap !important;
            line-height: 1.65 !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            background: #F7FAFF !important;
            border: 1.5px solid #D5E3F5 !important;
            color: #1B4F8A !important;
            border-radius: 14px !important;
            text-align: left !important;
            padding: 14px 16px !important;
        }}
        .feature-card button:hover {{
            background: #E4EFFF !important;
            border-color: #1B4F8A !important;
            box-shadow: 0 4px 14px rgba(27,79,138,0.13) !important;
        }}
        .feature-desc {{
            font-size: 11px;
            color: #8FA3BE;
            margin: 3px 0 8px 1px;
            line-height: 1.5;
        }}

        /* ── 프라이머리 버튼 ── */
        .btn-primary button {{
            background: #1B4F8A !important;
            color: #FFFFFF !important;
            border-color: #1B4F8A !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(27,79,138,0.22) !important;
            border-radius: 10px !important;
        }}
        .btn-primary button:hover {{
            background: #163F70 !important;
            box-shadow: 0 4px 14px rgba(27,79,138,0.32) !important;
        }}
        .btn-secondary button {{
            border-radius: 10px !important;
            font-weight: 500 !important;
            color: #5A6A80 !important;
        }}

        /* ── 구간 선택 버튼 ── */
        .range-btn button {{
            border-radius: 8px !important;
            border: 1.5px solid #D5E0EF !important;
            background: #F7FAFF !important;
            color: #1A2744 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            padding: 10px 6px !important;
        }}
        .range-btn button:hover {{
            border-color: #1B4F8A !important;
            background: #E4EFFF !important;
            color: #1B4F8A !important;
        }}

        /* ── 프로필 카드 ── */
        .profile-card {{
            border: 1.5px solid #E2EAF6;
            border-radius: 12px;
            padding: 14px 20px;
            background: #F7FAFF;
            margin-bottom: 16px;
        }}

        /* ── 결과 카드 ── */
        .result-card {{
            background: #F7FAFF;
            border: 1.5px solid #E2EAF6;
            border-radius: 14px;
            padding: 20px 24px;
            margin: 14px 0;
            line-height: 1.9;
            font-size: {size};
            color: #1A2744;
        }}

        /* ── 개선 방안 버튼 (소형) ── */
        .advice-btn button {{
            font-size: 13px !important;
            font-weight: 500 !important;
            color: #1B4F8A !important;
            background: #EBF3FF !important;
            border: 1.5px solid #C0D4EE !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
        }}
        .advice-btn button:hover {{
            background: #D6E8FF !important;
            border-color: #1B4F8A !important;
        }}

        /* ── 입력창 ── */
        .stTextInput {{
            max-width: 320px !important;
            margin: 0 auto !important;
        }}
        .stTextInput input {{
            font-size: {size} !important;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1.5px solid #D5E0EF !important;
            background: #FFFFFF !important;
            color: #1A2744 !important;
            text-align: center !important;
        }}

        hr {{ border-color: #E8EFF8; margin: 1.2rem 0; }}

        /* ── 사이드바 ── */
        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E2EAF6;
        }}

        /* ── 컬럼 간격 ── */
        div[data-testid="column"] {{ padding: 0 5px !important; }}
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
        title=dict(text=f"<b>자산 예측 경로</b>  ,  고갈 확률: "
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
        f"  ,  최적: <span style='color:#E53935'>{optimal}세</span>"
    )
    if breakeven:
        title_txt += f"  ,  손익분기: {breakeven:.0f}세"

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
    st.markdown(
        "<div style='text-align:center; margin-bottom:24px; color:#8FA3BE; font-size:14px;'>"
        "이용 방식을 선택해 주세요</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="mode-card">', unsafe_allow_html=True)
        if st.button("버튼 모드\n\n준비된 항목 중에서\n골라 확인하는 방식", key="select_btn", use_container_width=True):
            st.session_state.ui_mode = "버튼 모드"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="mode-card">', unsafe_allow_html=True)
        if st.button("채팅 모드\n\n궁금한 것을\n자유롭게 대화하는 방식", key="select_chat", use_container_width=True):
            st.session_state.ui_mode = "채팅 모드"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ── 기능 선택 ────────────────────────────────────────────────────────────────

def _show_feature_select():
    st.markdown(
        "<div style='text-align:center; margin-bottom:24px; color:#8FA3BE; font-size:14px;'>"
        "원하는 기능을 선택해 주세요</div>",
        unsafe_allow_html=True,
    )

    features = [
        ("내 자산 얼마나 버티나요?",    "다양한 미래 상황을 따져 자산 고갈 위험 확인"),
        ("연금 언제 받는 게 좋을까요?", "일찍 vs 늦게 받을 때 총 수령액 비교"),
        ("또래랑 비교해주세요",          "비슷한 처지 가구 대비 내 자산 순위"),
        ("지출 줄이면 어떻게 되나요?",  "지출 20% 아끼면 자산이 얼마나 더 버티나"),
    ]

    col1, col2 = st.columns(2, gap="medium")
    cols = [col1, col2, col1, col2]
    for i, (title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f'<div class="feature-desc">{desc}</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            if st.button(title, key=f"feat_{i}", use_container_width=True):
                st.session_state.selected_feature = i
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        if i == 1:
            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)


# ── 온보딩 ────────────────────────────────────────────────────────────────────

def _show_onboarding():
    step = st.session_state.step
    key, question, unit = ONBOARDING[step]

    # ── 진행 상태 (스텝 도트) ─────────────────────────────────────────────────
    step_labels = ["나이", "소득", "자산", "지출", "연금", "수령"]
    dots_html = "<div style='display:flex; justify-content:center; gap:20px; margin:8px 0 20px;'>"
    for i, label in enumerate(step_labels):
        if i < step:
            dot_color = "#1B4F8A"
            dot_border = "#1B4F8A"
            label_color = "#1B4F8A"
            dot_inner = f"<div style='width:16px;height:16px;border-radius:50%;background:{dot_color};'></div>"
        elif i == step:
            dot_color = "#1B4F8A"
            dot_border = "#1B4F8A"
            label_color = "#1B4F8A"
            dot_inner = (
                f"<div style='width:20px;height:20px;border-radius:50%;background:{dot_color};"
                f"box-shadow:0 0 0 3px #BDD5F7;'></div>"
            )
        else:
            dot_color = "transparent"
            dot_border = "#C5D3E0"
            label_color = "#B0BEC5"
            dot_inner = f"<div style='width:16px;height:16px;border-radius:50%;border:2px solid {dot_border};'></div>"
        dots_html += (
            f"<div style='display:flex;flex-direction:column;align-items:center;gap:5px;'>"
            f"{dot_inner}"
            f"<span style='font-size:11px;color:{label_color};font-weight:{'600' if i <= step else '400'};'>{label}</span>"
            f"</div>"
        )
    dots_html += "</div>"
    st.markdown(dots_html, unsafe_allow_html=True)

    st.markdown(f"### {question}")

    if key == "monthly_income":
        st.caption("근로소득,사업소득,임대소득 합산. 없으면 '없음' 선택. (60세까지 유지된다고 가정합니다)")

    ranges = RANGES.get(key)

    if ranges:
        # ── 구간 선택 버튼 ────────────────────────────────────────────
        st.markdown("**아래에서 가장 가까운 구간을 골라주세요**")
        ncols = 3 if key in ("assets", "pension_start_age") else 2
        cols = st.columns(ncols, gap="small")
        for idx, (label, val) in enumerate(ranges):
            with cols[idx % ncols]:
                st.markdown('<div class="range-btn">', unsafe_allow_html=True)
                if st.button(label, key=f"range_{step}_{idx}", use_container_width=True):
                    _save_step(key, val, label=label)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin:20px 0 8px; font-size:14px; color:#7A8FA6; text-align:center;'>─── 또는 직접 입력 ───</div>", unsafe_allow_html=True)
        st.caption(f"{unit} 단위 숫자만 입력")
    else:
        st.caption(f"{unit} 단위로 입력해주세요")

    val = st.text_input("입력", key=f"input_{step}", label_visibility="collapsed",
                        placeholder=f"숫자만 입력 (단위: {unit})")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    _, col_back, col_next, _ = st.columns([1, 2, 5, 1], gap="small")
    with col_back:
        st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
        if st.button("← 이전", key=f"back_{step}", use_container_width=True):
            if step == 0:
                if st.session_state.ui_mode == "채팅 모드":
                    st.session_state.ui_mode = None
                else:
                    st.session_state.selected_feature = None
            else:
                st.session_state.step -= 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_next:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("다음 →", key=f"btn_{step}", use_container_width=True):
            cleaned = val.strip().replace(",", "")
            if cleaned.isdigit():
                _save_step(key, int(cleaned))
            else:
                st.warning("숫자만 입력해주세요.")
        st.markdown('</div>', unsafe_allow_html=True)


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
        st.session_state.conv_history = []
        st.rerun()

    # ── 프로필 요약 카드 ──────────────────────────────────────────────────────
    income_line = f"<span style='margin-right:24px;'>월 소득 {income_disp}</span>" if income > 0 or "monthly_income" in lbl else ""
    st.markdown(f"""
    <div class="profile-card">
        <div style='font-size:17px; font-weight:700; color:#1A2744; margin-bottom:6px;'>
            {age}세 &nbsp;,&nbsp; 자산 {assets_disp}
        </div>
        <div style='font-size:14px; color:#5A6A80; display:flex; flex-wrap:wrap; gap:4px 0;'>
            {income_line}
            <span style='margin-right:24px;'>월 지출 {expense_disp}</span>
            <span>월 연금 {pension_disp}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 버튼 그리드 (2×2) ────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    buttons = [col1, col2, col1, col2]

    for i, (label, prompt) in enumerate(MENU):
        with buttons[i]:
            st.markdown('<div class="menu-btn">', unsafe_allow_html=True)
            if st.button(label, key=f"menu_{i}", use_container_width=True):
                st.session_state.pending_question = _inject_profile(prompt)
                st.session_state.result_text = None
                st.session_state.sim_result = None
                st.session_state.pension_result = None
                st.session_state.active_button = i
                st.session_state.conv_history = []
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
        # 팬 차트 (Monte Carlo)
        if st.session_state.sim_result and st.session_state.active_button != 1:
            st.plotly_chart(_fan_chart(p, st.session_state.sim_result), use_container_width=True)

        # 연금 막대 차트
        if st.session_state.pension_result and st.session_state.active_button == 1:
            pr = st.session_state.pension_result
            adj = pr.get("adjustment_rate", 1.0)
            monthly_opt = pr.get("monthly_at_optimal", 0)
            if monthly_opt > 10000:
                monthly_opt = monthly_opt / 10000
            st.plotly_chart(_pension_chart(pr, round(monthly_opt / (1 + adj), 1)), use_container_width=True)

        # 결과 카드
        st.markdown(
            f"<div class='result-card'>{st.session_state.result_text}</div>",
            unsafe_allow_html=True,
        )

        # ── 고위험 시 개선 방안 버튼 ─────────────────────────────────────────
        sim = st.session_state.sim_result
        if sim and sim.get("depletion_probability", 0) >= 0.5:
            depletion_pct = int(sim["depletion_probability"] * 100)
            st.markdown(f"""
            <div style='border-left:4px solid #C0392B; padding:10px 16px; margin:12px 0 16px;
                        background:#FFF5F5; border-radius:0 8px 8px 0;'>
                <span style='color:#C0392B; font-weight:700;'>자산 고갈 확률 {depletion_pct}%</span>
                <span style='color:#666; font-size:14px;'> — 아래 방법으로 개선해 보세요.</span>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3, gap="small")
            with col1:
                st.markdown('<div class="advice-btn">', unsafe_allow_html=True)
                if st.button("지출 20% 줄이면?", key="advice_expense", use_container_width=True):
                    expense_reduced = p.get("monthly_expense", 0) * 0.8
                    new_p = dict(p)
                    new_p["monthly_expense"] = expense_reduced
                    st.session_state.pending_question = _inject_profile_custom(
                        new_p,
                        f"월 지출을 {int(expense_reduced//10000)}만원으로 줄이면 자산이 얼마나 더 버티나요?"
                    )
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.session_state.pension_result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="advice-btn">', unsafe_allow_html=True)
                if st.button("소득 기간 늘리면?", key="advice_income", use_container_width=True):
                    st.session_state.pending_question = _inject_profile(
                        "근로소득을 65세까지 유지하면 자산 고갈 확률이 어떻게 달라지나요? income_until_age=65로 재계산해주세요."
                    )
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.session_state.pension_result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="advice-btn">', unsafe_allow_html=True)
                if st.button("자산 배분 바꾸면?", key="advice_portfolio", use_container_width=True):
                    st.session_state.pending_question = _inject_profile(
                        "현재 상황에서 자산 배분을 최적화하면 위험을 얼마나 줄일 수 있나요? 보수적 포트폴리오로 계산해주세요."
                    )
                    st.session_state.result_text = None
                    st.session_state.sim_result = None
                    st.session_state.pension_result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── 이상 거래 탐지 폼 ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("이상 거래 탐지 — 거래 정보 직접 입력"):
        st.caption("보이스피싱,이상 이체 여부를 분석합니다. 의심스러운 거래 정보를 입력해주세요.")
        col_a, col_b = st.columns(2)
        with col_a:
            tx_amount = st.number_input("거래 금액 (만원)", min_value=0.0, value=100.0, step=10.0, key="tx_amount")
            tx_new = st.radio("상대 계좌", ["처음 거래하는 계좌", "기존 거래 계좌"], key="tx_new") == "처음 거래하는 계좌"
        with col_b:
            tx_hour = st.slider("거래 시각", 0, 23, 14, key="tx_hour", format="%d시")
            tx_type_label = st.selectbox("거래 유형", ["이체", "결제", "출금"], key="tx_type")
        tx_count = st.number_input("연속 거래 횟수", min_value=1, max_value=20, value=1, key="tx_count")

        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("이상 거래 분석하기", key="anomaly_btn", use_container_width=True):
            type_map = {"이체": "transfer", "결제": "payment", "출금": "withdrawal"}
            from agent.tools import run_anomaly_score
            try:
                res = run_anomaly_score(tx_amount, tx_hour, tx_new, type_map[tx_type_label], tx_count)
                score = res['score']
                level = res['level']
                color = "#C0392B" if level == "high" else "#E67E22" if level == "medium" else "#27AE60"
                badge = "위험" if level == "high" else "주의" if level == "medium" else "정상"
                st.markdown(f"""
                <div style='border:2px solid {color}; border-radius:12px; padding:20px 24px; margin:12px 0; background:#FAFBFF;'>
                    <div style='display:flex; align-items:center; gap:16px;'>
                        <div style='font-size:42px; font-weight:800; color:{color};'>{score}</div>
                        <div>
                            <div style='font-size:18px; font-weight:700; color:{color};'>{badge}</div>
                            <div style='font-size:13px; color:#666; margin-top:2px;'>이상도 점수 (0=정상 / 100=매우 위험)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if res.get('triggers'):
                    st.markdown("**감지된 위험 신호**")
                    for t in res['triggers']:
                        st.markdown(f"- {t}")
                st.info(res.get('recommendation', ''))
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 더 물어보기 (접힌 상태)
    with st.expander("더 물어보기"):
        user_q = st.text_input("질문을 입력하세요", key="extra_input",
                               label_visibility="collapsed",
                               placeholder="예: 지출을 150만원으로 줄이면 어떻게 되나요?")
        if st.button("질문하기", key="extra_btn"):
            if user_q.strip():
                st.session_state.pending_question = _inject_profile(user_q)
                st.session_state.result_text = None
                st.session_state.conv_history = []
                st.rerun()


def _show_chat():
    """채팅 모드 화면"""
    p   = st.session_state.profile
    age = p.get("age", 65)

    # 첫 진입 시 자산 시뮬레이션 자동 실행 (플래그를 API 호출 전에 세팅해 무한 루프 방지)
    if not st.session_state.chat_messages and not st.session_state.chat_init_attempted:
        st.session_state.chat_init_attempted = True
        welcome = (
            f"안녕하세요! 입력하신 정보를 바탕으로 재무 상담을 도와드리겠습니다.\n\n"
            "아래와 같은 질문을 해보세요:\n"
            "- 내 자산이 얼마나 버티나요?\n"
            "- 연금을 언제 받는 게 좋을까요?\n"
            "- 또래랑 비교해주세요\n"
            "- 지출을 줄이면 어떻게 되나요?"
        )
        is_error = False
        with st.spinner("기본 자산 분석 중..."):
            try:
                init_q = _inject_profile("내 자산이 얼마나 버티는지 Monte Carlo 시뮬레이션으로 계산해주세요.")
                reply, updated = run_agent(init_q, [], age=age)
                if any(kw in reply for kw in ["오류", "잠시 후", "실패", "error", "Error"]):
                    is_error = True
                    reply = welcome
                    updated = []
            except Exception:
                is_error = True
                reply = welcome
                updated = []
        st.session_state.conv_history = updated
        sim = _extract_sim(updated)
        ai_msg: dict = {"role": "assistant", "content": reply}
        if sim and not is_error:
            ai_msg["chart"] = "fan"
            ai_msg["sim"]   = sim
        st.session_state.chat_messages.append(ai_msg)
        st.rerun()

    # 대화 기록 표시
    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
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

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
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
                "chat_init_attempted": False,
                "ui_mode": None, "selected_feature": None,
            })
            st.rerun()

        st.divider()
        if age >= 75:
            segment = "후기 고령층"
        elif age >= 65:
            segment = "초기 고령층"
        elif age >= 50:
            segment = "은퇴 준비층"
        else:
            segment = None
        if segment:
            st.caption(segment)

    st.markdown("""
    <div class="app-header">
        <div class="app-header-title">LifeLong WM</div>
        <div class="app-header-sub">통계가 계산하고, AI가 쉽게 전달합니다</div>
    </div>
    <hr style='border-color:#E8EEF8; margin:0 0 20px;'>
    """, unsafe_allow_html=True)

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
