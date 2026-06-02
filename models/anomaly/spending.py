"""
이상치 탐지 Layer 2 — 소비 패턴 이상 탐지
- 재무 계획 이탈 조기 경보
- CUSUM + Change Point Detection
"""


def detect_spending_anomaly(
    monthly_spending: list,
    expected_range: dict,
) -> dict:
    """
    월별 지출의 구조적 변화 탐지

    Returns:
        {
            "is_anomaly": bool,
            "change_type": str,     # 'one_time', 'structural'
            "deviation_pct": float,
            "message": str,
        }
    """
    pass
