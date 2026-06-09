"""
Fix 1 (연금 물가연동) + Fix 2 (생존편향 제거) 검증
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from models.monte_carlo import DataLoader, MonteCarloSimulator, SimulationInput
from models.survival import SurvivalModel, PersonalProfile

np.random.seed(42)
loader = DataLoader()
params = loader.fit_distributions()
sim    = MonteCarloSimulator(params)
surv   = SurvivalModel()

# ─────────────────────────────────────────────────────────────
# FIX 1: 연금 물가연동 검증
# ─────────────────────────────────────────────────────────────
print('=' * 60)
print('[ FIX 1 ] 연금 물가연동 검증')
print('=' * 60)

# 테스트 1: 지출=0, 자산=0, 연금만 있을 때 고갈확률 = 0%여야 함
si_zero = SimulationInput(
    profile=PersonalProfile('M', 65),
    initial_assets=0,
    monthly_expense=0,
    pension_monthly=100,
    pension_start_age=65,
    stock_ratio=0.0,
    n_simulations=5_000,
)
r_zero = sim.run(si_zero)
p = r_zero.depletion_prob
print(f'\n[테스트 1] 자산 0 + 지출 0 + 연금 100만')
print(f'  고갈 확률: {p:.1%}  (기대값: 0.0%)')
print(f'  판정: {"PASS ✅" if p == 0.0 else "FAIL ❌ — 연금이 지출보다 적거나 연동 오류"}')

# 테스트 2: 20년 후 연금 누적액이 명목 고정보다 커야 함
si_big = SimulationInput(
    profile=PersonalProfile('M', 60),
    initial_assets=500_000,   # 충분히 커서 고갈 없음
    monthly_expense=0,
    pension_monthly=100,
    pension_start_age=60,
    stock_ratio=0.0,          # 수익률 변동 제거
    n_simulations=5_000,
)
r_big = sim.run(si_big)
asset_60 = r_big.percentile_paths.loc[60, 'p50']
asset_80 = r_big.percentile_paths.loc[80, 'p50']
accumulated = asset_80 - asset_60
nominal_fixed = 100 * 12 * 20          # 물가연동 없으면 24,000만
expected_low  = nominal_fixed * 1.10   # 최소 10% 이상 커야 연동 의미 있음

print(f'\n[테스트 2] 20년 누적 연금 (수익률 0, 지출 0)')
print(f'  실제 누적액 (p50): {accumulated:,.0f}만원')
print(f'  명목 고정 기준:    {nominal_fixed:,.0f}만원')
print(f'  증가율: {(accumulated / nominal_fixed - 1):.1%}')
print(f'  판정: {"PASS ✅" if accumulated > expected_low else "FAIL ❌"}')

# 테스트 3: 단조성 — 연금 많을수록 고갈 확률 감소
pensions  = [0, 50, 100, 200]
depletions = []
for pm in pensions:
    si = SimulationInput(
        profile=PersonalProfile('M', 65),
        initial_assets=20_000,
        monthly_expense=200,
        pension_monthly=pm,
        pension_start_age=65,
        stock_ratio=0.4,
        n_simulations=3_000,
    )
    depletions.append(sim.run(si).depletion_prob)

print(f'\n[테스트 3] 연금 증가 → 고갈 확률 단조 감소 여부')
for pm, dp in zip(pensions, depletions):
    print(f'  연금 월 {pm:>3}만원 → 고갈 확률: {dp:.1%}')
monotone = all(depletions[i] >= depletions[i+1] for i in range(len(depletions)-1))
print(f'  단조 감소: {"PASS ✅" if monotone else "FAIL ❌"}')


# ─────────────────────────────────────────────────────────────
# FIX 2: 생존편향 제거 검증
# ─────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('[ FIX 2 ] 생존편향 제거 검증')
print('=' * 60)

si_surv = SimulationInput(
    profile=PersonalProfile('M', 60),
    initial_assets=100_000,   # 충분해서 고갈 없음 → 0 = 사망만
    monthly_expense=0,
    pension_monthly=0,
    pension_start_age=65,
    stock_ratio=0.0,
    n_simulations=10_000,
)
r_surv = sim.run(si_surv)

print(f'\n[테스트 4] 고갈 없는 케이스: 0자산 비율 = 사망률과 일치 여부')
print(f'  (자산 충분, 지출=0 → 자산=0인 경로는 순수하게 사망한 경로)')
print()
print(f'  {"연령":>5}  {"0자산 비율(시뮬)":>16}  {"사망률(생명표)":>14}  {"차이":>8}')
print(f'  {'-'*50}')

model_profile = PersonalProfile('M', 60)
for age in [70, 75, 80, 85, 90]:
    pp = r_surv.percentile_paths
    if age in pp.index:
        # p50이 0이면 절반 이상 사망
        p10_val = pp.loc[age, 'p10']
        # 0자산 비율 추정: p10=0이면 10%이상 사망, p50=0이면 50%이상 사망
        # 더 정확히: 생명표 생존확률과 비교
        sf_val = surv.get_survival_function(model_profile).get(age, 0)
        death_rate_table = 1 - sf_val
        # 시뮬에서 0자산 비율: p10>0이면 <10% 사망
        sim_zero_pct = '~0~10%' if p10_val > 0 else (
            '~10~50%' if pp.loc[age,'p50'] > 0 else '>50%'
        )
        print(f'  {age:>5}세  {sim_zero_pct:>16}  {death_rate_table:>13.1%}  ', end='')
        # 대략적 일관성 판정
        if death_rate_table < 0.10 and sim_zero_pct == '~0~10%':
            print('일치 ✅')
        elif death_rate_table > 0.50 and sim_zero_pct == '>50%':
            print('일치 ✅')
        elif 0.10 <= death_rate_table <= 0.50 and sim_zero_pct == '~10~50%':
            print('일치 ✅')
        else:
            print('확인 필요 ⚠️')

# 테스트 5: 연령 0 시점(현재)에서 모든 경로 생존 → p10 = 초기자산
p10_now = r_surv.percentile_paths.loc[60, 'p10']
print(f'\n[테스트 5] 시작 연령(60세) p10 = 초기자산 여부')
print(f'  p10(60세): {p10_now:,.0f}만원  (기대값: 100,000만원)')
print(f'  판정: {"PASS ✅" if abs(p10_now - 100_000) < 1 else "FAIL ❌"}')

print()
print('=' * 60)
print('검증 완료')
print('=' * 60)
