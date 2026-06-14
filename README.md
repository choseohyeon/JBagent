# 평생 자산관리 AI Agent

> **통계가 불확실성을 계산하고, AI가 그것을 누구나 이해할 수 있는 말로 전달한다.**  
> JB금융그룹 Fin:AI Challenge — 지정주제1 개인 라이프케어 AI

**배포 주소**: https://jbagent-igjf4uinkqxp4gw9nwagnf.streamlit.app/  
**GitHub**: https://github.com/choseohyeon/JBagent  
**팀명**: RF | 조서현, 조호평

---

## 서비스 개요

기존 자산관리 서비스는 미래를 단일 숫자로 표현합니다.  
평생 자산관리 AI Agent는 미래를 **확률 분포**로 표현합니다.

```
기존 서비스: "83세까지 버팁니다"
이 서비스:  "83세 전 자산 고갈 확률은 41%입니다"
```

**대상**: 50세 이상 은퇴 준비층 및 고령층 (독거 어르신 포함)  
**핵심 구조**: 통계 모델이 계산 → AI가 쉬운 한국어로 설명 → 버튼 하나로 조회

---

## 주요 기능

| 기능 | 설명 | 사용 모델 |
|------|------|---------|
| 내 자산 얼마나 버티나요? | Monte Carlo 10,000회 시뮬레이션으로 자산 고갈 확률 계산 | `run_monte_carlo` |
| 연금 언제 받는 게 좋을까요? | 수령 시기별 총 수령액, 손익분기 연령 비교 | `run_pension` |
| 또래랑 비교해주세요 | K-means 군집 내 자산 퍼센타일 랭킹 | `run_clustering` |
| 지출 줄이면 어떻게 되나요? | 지출 20% 절감 시 자산 수명 재시뮬레이션 | `run_monte_carlo` |
| 이상 거래 탐지 | Z-score, Isolation Forest, CUSUM 앙상블로 이상도 계산 | `run_anomaly_score` |

---

## 아키텍처

```
[사용자 — 버튼 모드 or 직접 질문 모드]
        ↓
[Groq API — llama-3.1-8b-instant]
  어떤 통계 도구를 쓸지 결정 + Tool 호출
        ↓
[통계 모델 (순수 Python)]
  survival.py    — Cox 생존 분석 (생명표 기반)
  monte_carlo.py — 10,000회 자산 경로 시뮬레이션
  pension.py     — 연금 수령 시기 최적화
  portfolio.py   — CVaR 자산 배분
  clustering.py  — K-means 또래 비교
  anomaly/       — 이상 거래 탐지
        ↓
[Groq API — llama-3.1-8b-instant]
  계산 결과를 자연스러운 한국어로 설명
        ↓
[Streamlit UI]
  팬 차트 + 결과 텍스트 (연령별 글씨 크기 자동 조정)
```

---

## 모드 선택

| 모드 | 설명 |
|------|------|
| 버튼 모드 | 메뉴에서 기능을 선택해 단계별로 정보를 입력하는 방식 |
| 직접 질문 모드 | 궁금한 내용을 자유롭게 입력하면 Agent가 멀티턴으로 응답 |

---

## 평가 지표

| 항목 | 결과 |
|------|------|
| Tool 선택 정확도 | **95%** (19 / 20) |
| 다중 턴 맥락 유지 | **80%** (4 / 5) |
| 위험 조언 차단 F1 | **71%** (Recall 73.3%) |

> 평가 방법: `python evaluation/test_cases.py`

---

## 실행 방법

### 1. 환경 설정

```bash
conda create -n jbfinai python=3.11
conda activate jbfinai
pip install -r requirements.txt
```

### 2. API 키 설정

`.streamlit/secrets.toml` 파일을 생성하고 Groq API 키를 입력합니다.

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

> Groq API 키는 https://console.groq.com 에서 무료로 발급받을 수 있습니다.

### 3. 앱 실행

```bash
conda activate jbfinai
streamlit run ui/app.py
```

브라우저에서 `http://localhost:8501` 접속

또는 배포된 서비스 바로 이용: https://jbagent-igjf4uinkqxp4gw9nwagnf.streamlit.app/

### 4. 평가 실행 (선택)

```bash
conda activate jbfinai
python evaluation/test_cases.py
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| LLM | Groq API (llama-3.1-8b-instant) |
| 통계 모델 | lifelines, scipy, cvxpy, scikit-learn, ruptures |
| UI | Streamlit, Plotly |
| 데이터 | 통계청 국민생명표, 한국은행 ECOS, KRX, 가계금융복지조사 |

---

## 데이터 소스

| 데이터 | 출처 | 사용처 |
|-------|------|--------|
| 국민생명표 | 통계청 | 생존 분석 |
| 가계금융복지조사 마이크로데이터 | 통계청 MDIS | 또래 군집 비교 |
| 금리, CPI 시계열 | 한국은행 ECOS API | Monte Carlo 확률 변수 |
| 주식, 채권 수익률 | KRX | 포트폴리오 최적화 |
| 국민연금 수령 통계 | 국민연금공단 | 연금 최적화 |

---

## 프로젝트 구조

```
JB_FinAI/
├── agent/
│   ├── agent.py       # Agent 루프 (Groq API Tool Use)
│   ├── tools.py       # Tool 정의 및 call_tool() 디스패처
│   └── prompts.py     # 연령 세그먼트별 시스템 프롬프트
├── models/
│   ├── survival.py    # 생존 분석
│   ├── monte_carlo.py # Monte Carlo 시뮬레이션
│   ├── pension.py     # 연금 최적화
│   ├── portfolio.py   # CVaR 포트폴리오 최적화
│   ├── clustering.py  # K-means 또래 비교
│   └── anomaly/       # 이상 거래 탐지
├── ui/
│   └── app.py         # Streamlit 앱 (Slow Banking UI)
├── evaluation/
│   ├── test_cases.py  # 자동 평가 실행기
│   └── metrics.py     # 평가 지표 계산
└── requirements.txt
```

---

## 연령별 UI 설계 (Slow Banking)

| 세그먼트 | 연령 | 글씨 크기 | 언어 수준 |
|---------|------|---------|---------|
| 은퇴 준비층 | 50~64세 | 17px | 일반 성인 수준 |
| 초기 고령층 | 65~74세 | 20px | 쉬운 단어, 핵심 수치 강조 |
| 후기 고령층 | 75세 이상 | 22px | 최대 단순화, 전문가 상담 권고 |

버튼 기반 인터페이스 — 어르신이 AI와 직접 대화하지 않아도 됩니다.
