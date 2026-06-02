"""
이상치 탐지 Layer 1 — 거래 이상 탐지
- 금융사기 / 보이스피싱 방어
- Z-score + Isolation Forest + CUSUM 조합
"""


def fit_baseline(transaction_history: list):
    """개인 거래 이력으로 기저선 학습"""
    pass


def detect_anomaly(transaction: dict, baseline) -> dict:
    """
    단일 거래 이상도 점수 계산

    Args:
        transaction: {"amount": float, "hour": int, "is_new_account": bool, "type": str}

    Returns:
        {
            "anomaly_score": float,   # 0~1
            "level": str,             # 'normal', 'warning', 'danger'
            "reasons": list,          # 이상 판정 근거
        }
    """
    pass
