"""
Monte Carlo 시뮬레이션 모듈
- 10,000회 생애 현금흐름 시뮬레이션
- 자산 고갈 확률 및 불확실성 범위 산출
"""


def run_monte_carlo(
    assets: float,
    monthly_expense: float,
    pension: float,
    pension_start_age: int,
    survival_probs: list,
    n_simulations: int = 10000,
) -> dict:
    """
    Args:
        assets: 총 자산 (원)
        monthly_expense: 월 지출 (원)
        pension: 월 연금 수령액 (원)
        pension_start_age: 연금 수령 시작 연령
        survival_probs: 생존 분석 결과 (survival.py 출력)
        n_simulations: 시뮬레이션 횟수

    Returns:
        {
            "depletion_probability": float,  # 고갈 확률 (0~1)
            "safe_until_age": int,           # 중앙값 기준 안전 연령
            "percentile_10": int,            # 비관적 시나리오
            "percentile_50": int,            # 중앙 시나리오
            "percentile_90": int,            # 낙관적 시나리오
            "paths": list,                   # 시각화용 샘플 경로
        }
    """
    pass
