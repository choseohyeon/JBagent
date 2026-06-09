"""
portfolio.py 검증: CVaR 공식 / 비중 제약 / 단조성 / 효율적 프론티어
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from models.portfolio import PortfolioOptimizer

opt = PortfolioOptimizer()
R   = opt.returns.values   # (204, 2)
S   = opt.n_scenarios

PASS = "PASS ✅"
FAIL = "FAIL ❌"

def check(cond, label, got=None, expected=None):
    status = PASS if cond else FAIL
    detail = f"  got={got}  expected≈{expected}" if got is not None else ""
    print(f"  {status}  {label}{detail}")
    return cond

# ─────────────────────────────────────────────────────────────
# 1. 비중 제약 검증
# ─────────────────────────────────────────────────────────────
print("=" * 58)
print("[ 1 ] 비중 제약 검증 (합=1, 음수 없음)")
print("=" * 58)

for level in ('conservative', 'moderate', 'aggressive'):
    res = opt.optimize_for_profile(level)
    w   = res.weights.values
    sum_ok  = abs(w.sum() - 1.0) < 1e-4
    nonneg  = (w >= -1e-6).all()
    label   = opt.PROFILE_KO[level]
    check(sum_ok,  f"{label}: 비중 합 = {w.sum():.6f}")
    check(nonneg,  f"{label}: 비중 >= 0 (최소값 {w.min():.6f})")

# ─────────────────────────────────────────────────────────────
# 2. CVaR 수식 직접 검증 (Rockafellar-Uryasev)
# ─────────────────────────────────────────────────────────────
print()
print("=" * 58)
print("[ 2 ] CVaR 수식 직접 검증 (alpha=0.95)")
print("=" * 58)

alpha = 0.95
res_check = opt.optimize(alpha=alpha)
w_vec = res_check.weights.values

# 포트폴리오 손실 시나리오
port_losses = -(R @ w_vec)

# 직접 계산: VaR + E[초과 손실]
var_threshold  = np.percentile(port_losses, alpha * 100)
exceed_losses  = port_losses[port_losses >= var_threshold]
cvar_direct    = float(exceed_losses.mean())
cvar_code      = res_check.cvar_annual / 12   # 월간으로 환산

check(abs(cvar_direct - cvar_code) < 0.005,
      "CVaR 수식 직접 계산 대조",
      got=f"{cvar_code:.4f}", expected=f"{cvar_direct:.4f}")

# CVaR >= VaR (정의상 반드시)
var_monthly = float(np.percentile(port_losses, alpha * 100))
check(cvar_code >= var_monthly,
      f"CVaR({cvar_code:.4f}) >= VaR({var_monthly:.4f})")

# CVaR >= 기대 손실 (정의상 반드시)
mean_loss = float(port_losses.mean())
check(cvar_code >= mean_loss,
      f"CVaR({cvar_code:.4f}) >= E[손실]({mean_loss:.4f})")

# ─────────────────────────────────────────────────────────────
# 3. 효율적 프론티어 단조성
# ─────────────────────────────────────────────────────────────
print()
print("=" * 58)
print("[ 3 ] 효율적 프론티어 단조성")
print("=" * 58)

ef = opt.efficient_frontier(n_points=10)

# 기대수익 증가 → CVaR 증가 (단조)
returns_mono = ef['expected_return_annual'].is_monotonic_increasing
cvar_mono    = ef['cvar_annual'].is_monotonic_increasing
check(returns_mono, "기대수익률 단조 증가")
check(cvar_mono,    "CVaR 단조 증가 (수익 높을수록 꼬리 위험 증가)")

# KOSPI 비중 증가 → CVaR 증가
kospi_mono = ef['w_KOSPI'].is_monotonic_increasing
check(kospi_mono,   "KOSPI 비중 단조 증가 (공격적일수록 주식 비중 높아짐)")

# Sharpe 단조 감소 (채권 고비중이 Sharpe 높음 — 2009~2026 한국 시장)
sharpe_mono = ef['sharpe_ratio'].is_monotonic_decreasing
check(sharpe_mono,  "Sharpe 단조 감소 (채권 고비중 → 높은 Sharpe)")

# ─────────────────────────────────────────────────────────────
# 4. 극단값 테스트
# ─────────────────────────────────────────────────────────────
print()
print("=" * 58)
print("[ 4 ] 극단값 테스트")
print("=" * 58)

# 최소 CVaR 포트폴리오 (목표 수익률 없음) → 채권 비중 높아야 함
res_min_cvar = opt.optimize(alpha=0.95, target_return=None)
check(res_min_cvar.weights['KTB'] > 0.7,
      f"최소CVaR 포트폴리오: KTB 비중 > 70%  (실제 {res_min_cvar.weights['KTB']:.1%})")

# 최대 수익률 목표 → KOSPI 비중 100% 수렴
max_target = float(opt.mean_returns.max()) * 0.97
res_max_r  = opt.optimize(alpha=0.95, target_return=max_target)
check(res_max_r.weights['KOSPI'] > 0.9,
      f"최대수익 목표: KOSPI 비중 > 90%  (실제 {res_max_r.weights['KOSPI']:.1%})")

# ─────────────────────────────────────────────────────────────
# 5. Monte Carlo 연결값 범위
# ─────────────────────────────────────────────────────────────
print()
print("=" * 58)
print("[ 5 ] Monte Carlo stock_ratio 범위 검증")
print("=" * 58)

ratios = {}
for level in ('conservative', 'moderate', 'aggressive'):
    r = opt.stock_ratio_for_profile(level)
    ratios[level] = r
    check(0.0 <= r <= 1.0,
          f"{opt.PROFILE_KO[level]} stock_ratio = {r:.2f} (0~1 범위)")

# 단조성: 보수적 < 중립 < 공격적
mono = ratios['conservative'] < ratios['moderate'] < ratios['aggressive']
check(mono,
      f"단조 증가: 보수({ratios['conservative']:.2f}) < "
      f"중립({ratios['moderate']:.2f}) < "
      f"공격({ratios['aggressive']:.2f})")

print()
print("=" * 58)
print("검증 완료")
print("=" * 58)
