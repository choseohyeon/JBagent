import sys, json
sys.path.insert(0, '.')
from agent.tools import call_tool, TOOLS

print('=== Tool 호출 테스트 ===\n')

r = call_tool('run_survival', {'sex':'M','current_age':65,'smoking':True})
print('run_survival:', json.dumps(r, ensure_ascii=False))

r = call_tool('run_portfolio', {'risk_level':'conservative'})
print('run_portfolio:', json.dumps(r, ensure_ascii=False))

r = call_tool('run_clustering', {'age':60,'net_assets':20000,'income':4000,'members':2})
print('run_clustering:', json.dumps(r, ensure_ascii=False))

print()
print(f'총 {len(TOOLS)}개 Tool 정의 완료')
names = [t['name'] for t in TOOLS]
print('Tool 이름:', names)
