"""
Claude API Tool Use 기반 LifeLong WM 다중 턴 대화 에이전트.

설계 원칙:
  - 대화 맥락 유지: 이전 파라미터를 context에 누적해 재사용
  - 파라미터 연속성: "지출만 바꾸면" → 나머지는 직전 값 유지
  - 고령층 언어: 시스템 프롬프트로 쉬운 말·수치 필수 출력 유도
"""
import json
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

from agent.tools import TOOLS, call_tool
from agent.prompts import build_system_prompt

# ── Tool별 허용 파라미터 ──────────────────────────────────────────────────────
# Claude가 생략한 파라미터를 context에서 자동 보충할 때 사용.
# run_clustering.age ↔ run_monte_carlo.current_age 는 동일 값으로 교차 사용.

_TOOL_PARAMS: dict[str, set] = {
    'run_survival': {
        'sex', 'current_age', 'smoking', 'chronic_disease', 'bmi',
    },
    'run_monte_carlo': {
        'sex', 'current_age', 'initial_assets', 'monthly_expense',
        'pension_monthly', 'pension_start_age', 'stock_ratio',
        'smoking', 'chronic_disease', 'n_simulations',
    },
    'run_pension': {
        'current_age', 'expected_monthly_pension', 'life_expectancy',
        'sex', 'smoking',
    },
    'run_portfolio': {'risk_level'},
    'run_clustering': {'age', 'net_assets', 'income', 'members'},
    'run_anomaly_score': {
        'amount', 'hour', 'is_new_account', 'tx_type', 'consecutive_count',
    },
}


class WMAgent:
    """LifeLong WM AI Agent — Claude API Tool Use 다중 턴 대화 루프."""

    MODEL      = 'claude-sonnet-4-6'
    MAX_TOKENS = 1500
    MAX_ROUNDS = 10   # 단일 응답당 최대 Tool 호출 라운드 (무한 루프 방지)

    def __init__(self, age_segment: Optional[int] = None):
        """
        Parameters
        ----------
        age_segment : 1 = 50~64세 | 2 = 65~74세 | 3 = 75세+ | None = 자동 감지
        """
        self.client      = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.messages: list[dict] = []
        self.context: dict        = {}   # 누적 사용자 파라미터
        self.age_segment          = age_segment

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _system(self) -> str:
        return build_system_prompt(self.age_segment)

    def _ctx_for(self, tool_name: str) -> dict:
        """context 에서 해당 Tool에 유효한 파라미터만 추출한다."""
        valid = _TOOL_PARAMS.get(tool_name, set())
        result = {k: v for k, v in self.context.items() if k in valid}

        # age ↔ current_age 교차 매핑
        if 'current_age' in result and 'age' not in result and 'age' in valid:
            result['age'] = result['current_age']
        if 'age' in result and 'current_age' not in result and 'current_age' in valid:
            result['current_age'] = result['age']

        return result

    def _update_ctx(self, params: dict):
        """사용된 파라미터를 context에 병합하고 age alias를 동기화한다."""
        self.context.update(params)
        if 'current_age' in params:
            self.context['age'] = params['current_age']
        if 'age' in params:
            self.context['current_age'] = params['age']

    def _execute(self, tool_name: str, tool_input: dict) -> str:
        """
        이전 context를 기반으로 파라미터를 보충한 뒤 Tool을 실행한다.
        새로 받은 tool_input이 context보다 항상 우선한다.
        """
        merged = {**self._ctx_for(tool_name), **tool_input}
        try:
            result = call_tool(tool_name, merged)
            self._update_ctx(merged)
            return json.dumps(result, ensure_ascii=False, default=float)
        except TypeError as e:
            # 필수 파라미터 누락 — Claude가 되물을 수 있도록 오류 반환
            return json.dumps({'error': f'파라미터 부족: {e}'}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'error': str(e)}, ensure_ascii=False)

    def _process_tool_use(self, response) -> list[dict]:
        """response의 tool_use 블록을 모두 실행하고 tool_result 리스트를 반환한다."""
        results = []
        for block in response.content:
            if block.type != 'tool_use':
                continue
            content = self._execute(block.name, dict(block.input))
            results.append({
                'type':        'tool_result',
                'tool_use_id': block.id,
                'content':     content,
            })
        return results

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        사용자 메시지 한 턴을 처리하고 Agent 응답 텍스트를 반환한다.
        내부 messages 리스트에 대화가 누적된다.
        """
        self.messages.append({'role': 'user', 'content': user_message})

        for _ in range(self.MAX_ROUNDS):
            response = self.client.messages.create(
                model      = self.MODEL,
                max_tokens = self.MAX_TOKENS,
                system     = self._system(),
                tools      = TOOLS,
                messages   = self.messages,
            )

            if response.stop_reason == 'tool_use':
                # Tool 호출 → 결과를 messages에 추가 후 다음 라운드
                self.messages.append({'role': 'assistant', 'content': response.content})
                tool_results = self._process_tool_use(response)
                self.messages.append({'role': 'user', 'content': tool_results})

            elif response.stop_reason == 'end_turn':
                reply = ''.join(
                    b.text for b in response.content if hasattr(b, 'text')
                )
                self.messages.append({'role': 'assistant', 'content': response.content})
                return reply

            else:
                # max_tokens 등 예외 상황
                break

        return '죄송합니다, 응답 처리 중 오류가 발생했습니다.'

    def reset(self):
        """대화 기록과 파라미터 맥락을 모두 초기화한다."""
        self.messages.clear()
        self.context.clear()

    @property
    def turn_count(self) -> int:
        """사용자 발화 수."""
        return sum(1 for m in self.messages if m['role'] == 'user'
                   and isinstance(m['content'], str))

    # ── CLI 테스트 ────────────────────────────────────────────────────────────

    def run_cli(self):
        """터미널 대화 테스트 루프."""
        print('LifeLong WM AI Agent (종료: exit 또는 Ctrl+C)')
        print('─' * 50)
        while True:
            try:
                user_input = input('나: ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\n종료합니다.')
                break
            if not user_input or user_input.lower() in ('exit', 'quit', '종료'):
                break
            reply = self.chat(user_input)
            print(f'\nAgent: {reply}\n')


if __name__ == '__main__':
    WMAgent().run_cli()
