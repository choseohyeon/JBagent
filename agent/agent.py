"""
Claude API Agent 메인 루프
- Tool Use 기반 다중 턴 대화
- 통계 모듈 호출 및 결과 해석
"""

import anthropic
from agent.tools import TOOLS
from agent.prompts import SYSTEM_PROMPT


client = anthropic.Anthropic()


def run_agent(user_message: str, conversation_history: list) -> tuple[str, list]:
    """
    사용자 메시지를 받아 Agent 응답 반환

    Args:
        user_message: 사용자 입력
        conversation_history: 이전 대화 기록

    Returns:
        (응답 텍스트, 업데이트된 대화 기록)
    """
    pass


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Tool 호출 → 해당 통계 모듈 실행 → 결과 반환"""
    pass
