# 역할 분담

## 역할 A — 통계 모델

### 생존 분석 (`models/survival.py`)
- 통계청 국민생명표 전처리
- Cox Proportional Hazards 모델 학습
- 성별, 흡연, 만성질환 등 개인 변수 반영
- 연령별 생존 확률 S(t) 출력

### Monte Carlo 시뮬레이션 (`models/monte_carlo.py`)
- 한국은행 ECOS에서 금리/CPI 시계열 수집 및 분포 추정
- KRX 수익률 데이터 기반 자산 수익률 분포 추정
- 생존 분석 결과를 수명 분포로 연결
- 10,000회 시뮬레이션 → 자산 고갈 확률, 10/50/90 퍼센타일 경로 출력

### 연금 최적화 (`models/pension.py`)
- 국민연금 수령 시기별 총 수령액 시나리오 계산
- 손익분기 연령 산출
- 생존 확률과 결합한 최적 수령 시기 권고

### CVaR 포트폴리오 최적화 (`models/portfolio.py`)
- KRX 자산별 수익률/공분산 행렬 계산
- cvxpy로 CVaR 최소화 최적화 문제 풀기
- 리스크 허용도별 최적 자산 배분 비율 출력

### K-means 군집 분석 (`models/clustering.py`)
- 통계청 가계금융복지조사 마이크로데이터 전처리
- 나이/자산/소득/부양가족 수로 군집화
- 동일 군집 내 퍼센타일 랭킹 및 상위 30% 행동 패턴 추출

### 이상치 탐지 (`models/anomaly/`)
- L1: 개인 거래 기저선 학습 → Z-score + Isolation Forest + CUSUM으로 이상 거래 탐지
- L2: 통계청 가계동향조사 기반 소비 Prior → CUSUM + Change Point Detection으로 지출 이탈 감지
- L3: 카드 결제/대화 응답 패턴 기반 복수 신호 합산 → 생활 패턴 이상 감지

### 데이터 수집 및 전처리 (`data/`)
- 국민생명표, ECOS API, KRX, 가계금융복지조사 수집
- 각 모델 입력 형식에 맞게 전처리

### 통계 모델 평가 (`evaluation/metrics.py`)
- Monte Carlo Coverage Probability 계산
- 생존 분석 C-index 계산
- 이상치 탐지 F1-score, AUC-ROC 계산
- 군집 분석 Silhouette Score 계산

---

## 역할 B — Agent + UI

### Claude API Agent 구현 (`agent/agent.py`)
- Claude API Tool Use 기반 다중 턴 대화 루프 구현
- 사용자 질문 → 적절한 통계 모듈 Tool 선택 → 결과 해석 흐름 구현
- 대화 맥락 유지 (이전 입력 파라미터 연속성 보장)
- "지출 줄이면 어떻게 돼요?" 같은 조건 변경 질문 시 해당 파라미터만 수정해 재계산

### 프롬프트 튜닝 (`agent/prompts.py`)
- 고령층 언어 변환 규칙 설정 (전문 용어 → 쉬운 말)
- 연령 세그먼트별 (50~64 / 65~74 / 75+) 대화 방식 조정
- 위험 조언 차단 규칙 설정
- 수치를 반드시 포함하는 응답 유도

### Streamlit UI (`ui/app.py`)
- Slow Banking UI 구현 (최소 18px 글씨, 고대비, 버튼 단순화)
- 연령 세그먼트 자동 감지 후 UI 모드 전환
- 음성 입력(Whisper API) + 음성 출력(gTTS) 연결
- 신뢰 연락처 없는 독거 고령층 보호 모드 구현

### 시각화 (`ui/`)
- Plotly 팬 차트로 Monte Carlo 불확실성 구간 시각화
- 연금 시나리오 비교 차트
- 군집 내 퍼센타일 위치 시각화

### Agent 평가 (`evaluation/test_cases.py`)
- Tool 선택 정확도 테스트 케이스 구성 (20개)
- 다중 턴 맥락 유지 테스트
- 위험 조언 유도 질문 테스트 (20개)
- LLM-as-Judge 방식 응답 품질 자동 평가

---

## 공통

### 전체 통합 디버깅
- 역할 A 모듈 → 역할 B Agent Tool 연결 테스트
- 입출력 형식 불일치 수정
- end-to-end 시나리오 3~4개 검증

### 최종 평가 실행
- 통계 모델 + Agent 전체 평가 지표 측정
- 발표용 성능 수치 정리

---

## Git 브랜치 규칙

```
main          최종 제출용 (항상 동작하는 코드만)
dev           통합 브랜치
feat/models   역할 A 작업 브랜치
feat/agent    역할 B 작업 브랜치
```

## 충돌 방지 규칙

- 역할 A는 `agent/`, `ui/` 수정 금지
- 역할 B는 `models/` 수정 금지
- `agent/tools.py`는 역할 A가 함수 시그니처 확정 후 역할 B가 참고
- 함수 시그니처 변경 시 반드시 상대방에게 먼저 알리기
- `requirements.txt`, `data/` 수정 시 상대방에게 알리기
