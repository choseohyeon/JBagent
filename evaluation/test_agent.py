"""
agent.py 오프라인 테스트 — API 호출 없이 구조·로직 검증.
실제 대화 테스트는 ANTHROPIC_API_KEY 설정 후 agent.py 직접 실행.
"""
import sys, json
sys.path.insert(0, '.')

PASS = "PASS ✅"
FAIL = "FAIL ❌"

def chk(cond, label, detail=""):
    s = PASS if cond else FAIL
    print(f"  {s}  {label}" + (f"  [{detail}]" if detail else ""))
    return cond


print("=" * 55)
print("1. 모듈 임포트")
print("=" * 55)
try:
    from agent.agent import WMAgent, _TOOL_PARAMS
    from agent.prompts import build_system_prompt
    chk(True, "agent.py, prompts.py 임포트")
except Exception as e:
    chk(False, f"임포트 실패: {e}")
    sys.exit(1)


print()
print("=" * 55)
print("2. 시스템 프롬프트 생성")
print("=" * 55)
for seg, label in [(None, "자동"), (1, "50~64세"), (2, "65~74세"), (3, "75세+")]:
    p = build_system_prompt(seg)
    chk(len(p) > 500, f"segment={label} 프롬프트 생성", f"{len(p)}자")

# 필수 키워드 포함 확인
p_all = build_system_prompt(2)
for kw in ["run_survival", "run_monte_carlo", "수치", "보이스피싱", "맥락"]:
    chk(kw in p_all, f"필수 키워드 포함: '{kw}'")


print()
print("=" * 55)
print("3. context 로직 — 파라미터 연속성")
print("=" * 55)
agent = WMAgent(age_segment=2)

# 첫 번째 Tool 호출 후 context 저장
agent._update_ctx({
    'sex': 'M', 'current_age': 65,
    'initial_assets': 30000, 'monthly_expense': 200,
    'pension_monthly': 80, 'pension_start_age': 65,
    'stock_ratio': 0.4,
})

# run_monte_carlo context 필터링
mc_ctx = agent._ctx_for('run_monte_carlo')
chk('initial_assets' in mc_ctx and mc_ctx['initial_assets'] == 30000,
    "initial_assets context 유지", str(mc_ctx.get('initial_assets')))
chk('current_age' in mc_ctx and mc_ctx['current_age'] == 65,
    "current_age context 유지")

# "지출만 바꾸면" 시나리오
new_input = {'monthly_expense': 150}
merged    = {**agent._ctx_for('run_monte_carlo'), **new_input}
chk(merged['monthly_expense'] == 150,
    "새 지출값 150 반영", str(merged['monthly_expense']))
chk(merged['initial_assets'] == 30000,
    "initial_assets는 이전 값 유지", str(merged['initial_assets']))

# age ↔ current_age 교차 매핑
chk(agent.context.get('age') == 65, "age alias 동기화 (current_age→age)")
cl_ctx = agent._ctx_for('run_clustering')
chk(cl_ctx.get('age') == 65, "run_clustering에서 age 활용")

# run_portfolio는 risk_level만 사용해야 함
pf_ctx = agent._ctx_for('run_portfolio')
chk(list(pf_ctx.keys()) == [] or all(k == 'risk_level' for k in pf_ctx),
    "run_portfolio context는 risk_level만", str(list(pf_ctx.keys())))


print()
print("=" * 55)
print("4. Tool 실행 — 실제 모델 호출 (API 키 불필요)")
print("=" * 55)
agent2 = WMAgent()

# survival
r = json.loads(agent2._execute('run_survival', {'sex': 'M', 'current_age': 65}))
chk('expected_remaining_life' in r and r.get('error') is None,
    "run_survival 실행", str(r.get('expected_remaining_life')))

# portfolio
r = json.loads(agent2._execute('run_portfolio', {'risk_level': 'moderate'}))
chk('weights' in r and r.get('error') is None,
    "run_portfolio 실행", str(r.get('sharpe_ratio')))

# context 누적 확인
chk('sex' in agent2.context and 'current_age' in agent2.context,
    "Tool 실행 후 context 누적")

# 파라미터 누락 시 오류 반환 (crash 없이)
r_err = json.loads(agent2._execute('run_monte_carlo', {}))
chk('error' in r_err, "필수 파라미터 누락 → error 반환 (crash 없음)")


print()
print("=" * 55)
print("5. _TOOL_PARAMS 완결성")
print("=" * 55)
from agent.tools import TOOLS
tool_names_spec  = {t['name'] for t in TOOLS}
tool_names_param = set(_TOOL_PARAMS.keys())
chk(tool_names_spec == tool_names_param,
    "TOOLS 스펙 ↔ _TOOL_PARAMS 이름 일치",
    str(tool_names_spec.symmetric_difference(tool_names_param) or "없음"))


print()
print("=" * 55)
print("6. reset() 동작")
print("=" * 55)
agent2.reset()
chk(len(agent2.messages) == 0 and len(agent2.context) == 0,
    "reset() 후 messages·context 비워짐")


print()
print("=" * 55)
print("완료 — ANTHROPIC_API_KEY 설정 후 실제 대화 테스트:")
print("  python agent/agent.py")
print("=" * 55)
