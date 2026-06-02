"""
이상치 탐지 Layer 3 — 생활 패턴 이상 탐지
- 독거 고령층 안부 확인
- 카드 결제 패턴 + AI 대화 응답 패턴 기반
- 합성 데이터로 기저선 구성
"""


def compute_risk_score(signals: dict) -> dict:
    """
    복수 신호 합산으로 위험도 점수 계산

    Args:
        signals: {
            "days_no_convenience_store": int,
            "days_no_response": int,
            "pharmacy_stopped": bool,
            "no_regular_payment": bool,
        }

    Returns:
        {
            "risk_score": int,
            "level": str,       # 'normal', 'check', 'alert'
            "action": str,      # 권고 행동
        }
    """
    pass
