"""
연금 최적 인출 전략 모듈
- 국민연금 수령 시기별 시나리오 비교
- 손익분기 연령 및 생존 확률 결합 권고
"""


def optimize_pension(
    current_age: int,
    expected_monthly_pension: float,
    earliest_start_age: int = 63,
    survival_probs: list = None,
) -> dict:
    """
    연금 수령 시기별 비교 분석

    Returns:
        {
            "scenarios": list,           # 수령 시기별 시나리오
            "recommended_age": int,      # 권고 수령 시작 연령
            "breakeven_age": int,        # 손익분기 연령
            "survival_prob_at_breakeven": float,
        }
    """
    pass
