"""
Claude API Tool 정의
- 각 통계 모듈을 Claude Tool Use 형식으로 래핑
- agent.py에서 tools 리스트로 사용
"""

TOOLS = [
    {
        "name": "run_monte_carlo",
        "description": "자산 고갈 확률과 안전 기간을 Monte Carlo 시뮬레이션으로 계산합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "assets": {"type": "number", "description": "총 자산 (원)"},
                "monthly_expense": {"type": "number", "description": "월 지출 (원)"},
                "pension": {"type": "number", "description": "월 연금 수령액 (원)"},
                "pension_start_age": {"type": "integer", "description": "연금 수령 시작 연령"},
            },
            "required": ["assets", "monthly_expense", "pension", "pension_start_age"],
        },
    },
    {
        "name": "estimate_survival",
        "description": "개인 건강 정보를 반영한 생존 확률을 추정합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "현재 나이"},
                "sex": {"type": "string", "description": "성별 (M 또는 F)"},
                "smoking": {"type": "boolean", "description": "흡연 여부"},
                "hypertension": {"type": "boolean", "description": "고혈압 여부"},
                "diabetes": {"type": "boolean", "description": "당뇨 여부"},
            },
            "required": ["age", "sex"],
        },
    },
    {
        "name": "optimize_pension",
        "description": "국민연금 수령 시기별 손익을 비교하고 최적 전략을 제안합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_age": {"type": "integer", "description": "현재 나이"},
                "expected_monthly_pension": {"type": "number", "description": "예상 월 연금액 (원)"},
            },
            "required": ["current_age", "expected_monthly_pension"],
        },
    },
    {
        "name": "optimize_portfolio",
        "description": "CVaR 기반 최적 자산 배분 비율을 계산합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_allocation": {"type": "object", "description": "현재 자산 배분 (예: {'주식': 0.4, '채권': 0.6})"},
                "risk_tolerance": {"type": "string", "description": "리스크 허용도 (low/medium/high)"},
                "investment_horizon": {"type": "integer", "description": "투자 기간 (년)"},
            },
            "required": ["risk_tolerance", "investment_horizon"],
        },
    },
    {
        "name": "get_percentile_rank",
        "description": "비슷한 조건의 또래 집단 대비 재무 상태 퍼센타일을 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "나이"},
                "net_assets": {"type": "number", "description": "순자산 (원)"},
                "monthly_income": {"type": "number", "description": "월 소득 (원)"},
                "dependents": {"type": "integer", "description": "부양가족 수"},
            },
            "required": ["age", "net_assets"],
        },
    },
    {
        "name": "detect_transaction_anomaly",
        "description": "거래 이상 여부를 탐지합니다. 금융사기 및 보이스피싱 방어에 활용됩니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "거래 금액 (원)"},
                "hour": {"type": "integer", "description": "거래 시간 (0~23)"},
                "is_new_account": {"type": "boolean", "description": "처음 거래하는 계좌 여부"},
                "transaction_type": {"type": "string", "description": "거래 유형 (이체/결제/출금)"},
            },
            "required": ["amount", "is_new_account"],
        },
    },
]
