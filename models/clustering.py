"""
K-means 군집 분석 모듈
- 유사 조건 또래 집단과의 재무 상태 비교
- 군집 내 퍼센타일 랭킹 산출
"""


def fit_clusters(data_path: str, n_clusters: int = 5):
    """가계금융복지조사 마이크로데이터로 군집 학습"""
    pass


def get_percentile_rank(
    age: int,
    net_assets: float,
    monthly_income: float,
    dependents: int,
) -> dict:
    """
    동일 군집 내 퍼센타일 랭킹 반환

    Returns:
        {
            "cluster_id": int,
            "percentile": float,         # 0~100
            "cluster_size": int,
            "top30_behavior": dict,      # 상위 30% 재무 행동 패턴
        }
    """
    pass
