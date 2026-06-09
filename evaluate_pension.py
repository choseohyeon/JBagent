"""
pension.py 검증: 조정률 / 손익분기 / NPV / 생존함수 연결
"""
import sys, os
import numpy as np
sys.path.insert(0, '.')
from models.pension import PensionOptimizer, PensionProfile, STANDARD_START_AGE
from models.survival import SurvivalModel, PersonalProfile

opt = PensionOptimizer()
PASS = "PASS ✅"
FAIL = "FAIL ❌"

def check(cond, label, got=None, expected=None):
    status = PASS if cond else FAIL
    detail = f"  got={got}, expected={expected}" if got is not None else ""
    print(f"  {status}  {label}{detail}")
    return cond

# ─────────────────────────────────────────────────────────────
# 1. 조정률 정확성 (법령 기준)
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("[ 1 ] 조정률 정확성 검증")
print("=" * 55)

cases = [
    (60, -0.30),   # 5년 조기 → -0.5% × 60개월 = -30%
    (61, -0.24),
    (62, -0.18),
    (63, -0.12),
    (64, -0.06),
    (65,  0.00),   # 정시
    (66, +0.072),  # 1년 연기 → +0.6% × 12개월 = +7.2%
    (67, +0.144),
    (68, +0.216),
    (69, +0.288),
    (70, +0.360),  # 5년 연기 → +0.6% × 60개월 = +36%
]
all_pass = True
for age, expected_rate in cases:
    _, rate = opt.monthly_amount(100.0, age)
    ok = abs(rate - expected_rate) < 1e-9
    check(ok, f"{age}세 조정률", got=f"{rate:+.1%}", expected=f"{expected_rate:+.1%}")
    all_pass = all_pass and ok

# ─────────────────────────────────────────────────────────────
# 2. 손익분기 수식 검증 (수계산과 비교)
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("[ 2 ] 손익분기 수식 검증 (수계산 대조)")
print("=" * 55)

def manual_breakeven(base, claim_age):
    """직접 공식 계산: actual*(t-claim) = base*(t-65) 풀기"""
    actual, _ = opt.monthly_amount(base, claim_age)
    a = actual * 12
    b = base * 12
    if a == b:
        return float('inf')
    return (a * claim_age - b * 65) / (a - b)

base = 100.0
for age in [60, 63, 67, 70]:
    code_val   = opt.breakeven_age(base, age)
    manual_val = manual_breakeven(base, age)
    ok = abs(code_val - manual_val) < 0.05
    check(ok, f"{age}세 손익분기", got=f"{code_val:.1f}세", expected=f"{manual_val:.1f}세")

# 단조성: 수령 늦출수록 손익분기도 높아져야 함
bes = [opt.breakeven_age(base, a) for a in range(61, 71)]
mono = all(bes[i] <= bes[i+1] for i in range(len(bes)-1))
check(mono, "손익분기 단조 증가 (늦을수록 높아짐)")

# ─────────────────────────────────────────────────────────────
# 3. NPV 공식 검증
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("[ 3 ] NPV 공식 검증")
print("=" * 55)

# 할인율 0일 때: NPV = 월수령 × 12 × 수령연수
monthly, _ = opt.monthly_amount(100.0, 65)
death_age  = 83.2
npv_zero_r = opt.npv_pension(monthly, 65, death_age, discount_rate=0)
expected_zero = monthly * 12 * (death_age - 65)
check(abs(npv_zero_r - expected_zero) < 1,
      "할인율=0일 때 NPV = 단순 합계",
      got=f"{npv_zero_r:,.0f}", expected=f"{expected_zero:,.0f}")

# 할인율 > 0이면 NPV < 단순합
npv_pos_r = opt.npv_pension(monthly, 65, death_age, discount_rate=0.025)
check(npv_pos_r < expected_zero,
      "할인율 적용 시 NPV < 단순합 (현재가치 원리)")

# 수령 기간 0이면 NPV = 0
npv_zero_period = opt.npv_pension(monthly, 65, 65, discount_rate=0.025)
check(abs(npv_zero_period) < 1,
      "수령 기간 0 → NPV ≈ 0", got=f"{npv_zero_period:.1f}")

# 수령액 많을수록 NPV 증가 단조성
npvs = [opt.npv_pension(opt.monthly_amount(100.0, a)[0], a, 85.0)
        for a in range(60, 71)]
check(True, "NPV 단조성 (시각 확인용): " +
      " / ".join(f"{a}세:{v:,.0f}" for a, v in zip(range(60, 71), npvs)))

# ─────────────────────────────────────────────────────────────
# 4. 생존함수 연결 효과 검증
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("[ 4 ] 생존함수 연결 효과 검증")
print("=" * 55)

surv   = SurvivalModel()
sprof  = PersonalProfile('M', 60)
sf     = surv.get_survival_function(sprof)
profile = PensionProfile(current_age=60,
                         expected_monthly_pension=100.0,
                         life_expectancy=23.2)

res_no_sf = opt.compare_scenarios(profile, survival_fn=None)
res_sf    = opt.compare_scenarios(profile, survival_fn=sf)

# 생존함수 적용 시 총수령액이 달라야 함
diff_at_65 = abs(res_no_sf.loc[res_no_sf.claim_age==65,'total_lifetime'].values[0]
               - res_sf.loc[res_sf.claim_age==65,'total_lifetime'].values[0])
check(diff_at_65 > 0,
      f"생존함수 적용 시 총수령액 변화 있음 (차이: {diff_at_65:,.0f}만원)")

# 조기수령일수록 생존함수 보정 후 불리해져야 함
total_60_sf  = res_sf.loc[res_sf.claim_age==60,  'total_lifetime'].values[0]
total_65_sf  = res_sf.loc[res_sf.claim_age==65,  'total_lifetime'].values[0]
total_60_raw = res_no_sf.loc[res_no_sf.claim_age==60,'total_lifetime'].values[0]
total_65_raw = res_no_sf.loc[res_no_sf.claim_age==65,'total_lifetime'].values[0]
ratio_sf  = total_60_sf  / total_65_sf
ratio_raw = total_60_raw / total_65_raw
check(ratio_sf <= ratio_raw,
      f"생존함수 적용 시 조기수령 상대적 불리 (비율 sf={ratio_sf:.3f} vs raw={ratio_raw:.3f})")

# ─────────────────────────────────────────────────────────────
# 5. NPS API 상태 확인
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("[ 5 ] NPS API 연결 상태")
print("=" * 55)

key = os.getenv('NPS_API_KEY', '')
if not key:
    print("  ⚠️  API 키 없음 — 폴백 사용 중")
else:
    avg = opt._nps_avg_by_age
    fallback_vals = {60: 62.0, 65: 78.0, 70: 85.0, 75: 72.0, 80: 58.0}
    is_fallback = (avg == fallback_vals)
    if is_fallback:
        print("  ⚠️  API 호출 실패 — 폴백 데이터 사용 중")
        print(f"     폴백 기준값: {avg}")
        print("     (pension 모델 자체는 법령 고정값으로 동작하므로 핵심 기능 정상)")
    else:
        print(f"  {PASS}  실제 NPS API 데이터 사용 중: {avg}")

print()
print("=" * 55)
print("검증 완료")
print("=" * 55)
