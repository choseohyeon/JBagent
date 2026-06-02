"""
성능 평가 지표 계산
- Tool 선택 정확도
- 수치 전달 정확도
- 위험 조언 차단율
"""


def tool_selection_accuracy(results: list) -> float:
    """올바른 Tool을 선택한 비율"""
    pass


def numerical_accuracy(results: list) -> float:
    """통계 모델 수치를 정확히 전달한 비율"""
    pass


def safety_block_rate(results: list) -> float:
    """위험 유도 질문에 경고/거부한 비율"""
    pass


def run_evaluation(test_cases: list, agent_fn) -> dict:
    """전체 평가 실행 후 리포트 반환"""
    pass
