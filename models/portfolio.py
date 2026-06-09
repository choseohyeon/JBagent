import numpy as np
import pandas as pd
import cvxpy as cp
import xlrd
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path(__file__).parent.parent / "필요한 파일"


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    weights: pd.Series              # 자산별 비중 (합=1)
    cvar_annual: float              # 연간 CVaR (양수 = 손실 크기)
    var_annual: float               # 연간 VaR
    expected_return_annual: float   # 연간 기대수익률
    sharpe_ratio: float             # Sharpe ratio (무위험 수익률 = 0)
    status: str                     # 최적화 상태


# ── 포트폴리오 최적화 ──────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    CVaR 포트폴리오 최적화 (Rockafellar & Uryasev 2000).

    자산: KOSPI (주식) + KTB 채권지수 (채권)
    목적: CVaR_α 최소화 (하위 (1-α)% 시나리오의 평균 손실 최소화)
    제약: 비중 합 = 1, 비중 >= 0 (공매도 금지)
    """

    # 리스크 허용도 → 기대수익률 목표 (최대 기대수익률 대비 비율)
    RISK_PROFILES = {
        'conservative': 0.30,   # 보수적: 최대의 30% 목표
        'moderate':     0.60,   # 중립
        'aggressive':   0.85,   # 공격적
    }

    PROFILE_KO = {
        'conservative': '보수적',
        'moderate':     '중립',
        'aggressive':   '공격적',
    }

    def __init__(self):
        self.returns      = self._load_returns()
        self.n_assets     = self.returns.shape[1]
        self.n_scenarios  = self.returns.shape[0]
        self.mean_returns = self.returns.mean()
        self.cov_matrix   = self.returns.cov()

    # ── 데이터 로드 ───────────────────────────────────────────────────────────

    def _load_returns(self) -> pd.DataFrame:
        kospi = self._load_kospi()
        ktb   = self._load_ktb()
        df    = pd.concat([kospi, ktb], axis=1).dropna()
        df.columns = ['KOSPI', 'KTB']
        return df

    def _load_kospi(self) -> pd.Series:
        path = DATA_DIR / "KOSPI.xls"
        wb = xlrd.open_workbook(str(path))
        ws = wb.sheet_by_index(0)
        dates, prices = [], []
        for r in range(1, ws.nrows):
            row = ws.row_values(r)
            try:
                dt    = pd.to_datetime(str(row[0]).strip())
                price = float(str(row[1]).replace(',', ''))
                dates.append(dt); prices.append(price)
            except Exception:
                continue
        s = pd.Series(prices, index=dates).sort_index()
        return s.resample('ME').last().pct_change().dropna()

    def _load_ktb(self) -> pd.Series:
        ktb_dir = DATA_DIR / "KTB 지수"
        frames  = []
        for f in sorted(ktb_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f, encoding='utf-8', thousands=',')
                df = df.iloc[:, [0, 1]].copy()
                df.columns = ['date', 'total']
                df['date']  = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
                df['total'] = pd.to_numeric(df['total'], errors='coerce')
                frames.append(df.dropna())
            except Exception:
                continue
        combined = (pd.concat(frames)
                      .set_index('date').sort_index()
                      .pipe(lambda x: x[~x.index.duplicated(keep='last')]))
        return combined['total'].resample('ME').last().pct_change().dropna()

    # ── CVaR 최적화 ───────────────────────────────────────────────────────────

    def optimize(self, alpha: float = 0.95,
                 target_return: float = None) -> OptimizationResult:
        """
        CVaR_α 최소화.

        Rockafellar-Uryasev 선형 재공식화:
          min  ζ + (1/((1-α)·S)) · Σ z_s
          s.t. z_s >= -r_s'w - ζ   ∀s
               z_s >= 0             ∀s
               Σ w_i = 1
               w_i >= 0

        Parameters
        ----------
        alpha         : 신뢰 수준 (0.95 → 하위 5% 평균 손실 최소화)
        target_return : 월간 최소 기대수익률 제약 (None = 제약 없음)
        """
        R    = self.returns.values          # (S, N)
        S, N = R.shape

        w    = cp.Variable(N)               # 포트폴리오 비중
        zeta = cp.Variable()                # VaR 보조변수 (ζ)
        z    = cp.Variable(S)               # 초과 손실 보조변수

        losses = -R @ w
        cvar   = zeta + (1.0 / ((1 - alpha) * S)) * cp.sum(z)

        constraints = [
            z >= losses - zeta,
            z >= 0,
            cp.sum(w) == 1,
            w >= 0,
        ]
        if target_return is not None:
            constraints.append(self.mean_returns.values @ w >= target_return)

        prob = cp.Problem(cp.Minimize(cvar), constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise ValueError(f"최적화 실패: {prob.status}")

        weights = pd.Series(np.clip(w.value, 0, 1),
                            index=self.returns.columns)
        weights /= weights.sum()

        port_r   = R @ weights.values
        mu       = float(self.mean_returns.values @ weights.values)
        sigma    = float(np.std(port_r))
        var_m    = float(-np.percentile(port_r, (1 - alpha) * 100))

        return OptimizationResult(
            weights                = weights,
            cvar_annual            = float(cvar.value) * 12,
            var_annual             = var_m * 12,
            expected_return_annual = mu * 12,
            sharpe_ratio           = (mu / sigma) * (12 ** 0.5) if sigma > 0 else 0.0,
            status                 = prob.status,
        )

    # ── 리스크 프로파일별 포트폴리오 ──────────────────────────────────────────

    def optimize_for_profile(self, risk_level: str,
                              alpha: float = 0.95) -> OptimizationResult:
        """
        리스크 허용도 ('conservative' | 'moderate' | 'aggressive') 기반 최적화.
        기대수익률 목표 = min_return + mult × (max_return - min_return)
        """
        if risk_level not in self.RISK_PROFILES:
            raise ValueError(f"risk_level must be one of {list(self.RISK_PROFILES)}")

        mu_max = float(self.mean_returns.max())
        mu_min = float(self.mean_returns.min())
        mult   = self.RISK_PROFILES[risk_level]
        target = mu_min + mult * (mu_max - mu_min)

        return self.optimize(alpha=alpha, target_return=target)

    # ── 효율적 프론티어 ───────────────────────────────────────────────────────

    def efficient_frontier(self, n_points: int = 15,
                            alpha: float = 0.95) -> pd.DataFrame:
        """
        기대수익률 구간별 CVaR-최소 포트폴리오를 모아 효율적 프론티어 생성.
        """
        # 최소-CVaR 포트폴리오 기대수익률 (하한)
        min_cvar_res = self.optimize(alpha=alpha)
        mu_lo = float(self.mean_returns.values @ min_cvar_res.weights.values)
        mu_hi = float(self.mean_returns.max()) * 0.97

        targets = np.linspace(mu_lo, mu_hi, n_points)
        rows    = []
        for target in targets:
            try:
                res = self.optimize(alpha=alpha, target_return=target)
                rows.append({
                    'expected_return_annual': res.expected_return_annual,
                    'cvar_annual':            res.cvar_annual,
                    'var_annual':             res.var_annual,
                    'sharpe_ratio':           res.sharpe_ratio,
                    **{f'w_{c}': float(res.weights[c])
                       for c in self.returns.columns},
                })
            except Exception:
                continue

        return pd.DataFrame(rows)

    # ── Monte Carlo 연결 헬퍼 ─────────────────────────────────────────────────

    def stock_ratio_for_profile(self, risk_level: str) -> float:
        """
        Monte Carlo SimulationInput.stock_ratio 에 바로 넣을 수 있는
        리스크 프로파일별 KOSPI 비중을 반환한다.
        """
        res = self.optimize_for_profile(risk_level)
        return float(res.weights['KOSPI'])


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt = PortfolioOptimizer()

    print(f"데이터: {opt.returns.index[0].strftime('%Y-%m')} ~ "
          f"{opt.returns.index[-1].strftime('%Y-%m')}  "
          f"({opt.n_scenarios}개 월별 시나리오)\n")

    # ── 리스크 프로파일별 포트폴리오 ──────────────────────────────────────────
    print('=' * 58)
    print('리스크 프로파일별 최적 포트폴리오 (CVaR α=0.95)')
    print('=' * 58)
    print(f"{'프로파일':>8}  {'KOSPI':>7}  {'KTB':>7}  "
          f"{'기대수익(연)':>10}  {'CVaR(연)':>9}  {'Sharpe':>7}")
    print('-' * 58)

    for level in ('conservative', 'moderate', 'aggressive'):
        res = opt.optimize_for_profile(level)
        print(f"{opt.PROFILE_KO[level]:>8}  "
              f"{res.weights['KOSPI']:>6.1%}  "
              f"{res.weights['KTB']:>6.1%}  "
              f"{res.expected_return_annual:>9.1%}  "
              f"{res.cvar_annual:>8.1%}  "
              f"{res.sharpe_ratio:>7.3f}")

    # ── 효율적 프론티어 ───────────────────────────────────────────────────────
    print()
    print('=' * 58)
    print('효율적 프론티어 (기대수익률 순)')
    print('=' * 58)
    ef = opt.efficient_frontier(n_points=8)
    print(f"{'기대수익(연)':>11}  {'CVaR(연)':>9}  "
          f"{'w_KOSPI':>8}  {'w_KTB':>7}  {'Sharpe':>7}")
    print('-' * 58)
    for _, row in ef.iterrows():
        print(f"{row['expected_return_annual']:>10.1%}  "
              f"{row['cvar_annual']:>8.1%}  "
              f"{row['w_KOSPI']:>7.1%}  "
              f"{row['w_KTB']:>6.1%}  "
              f"{row['sharpe_ratio']:>7.3f}")

    # ── Monte Carlo 연결 ──────────────────────────────────────────────────────
    print()
    print('=' * 58)
    print('Monte Carlo stock_ratio 연결값')
    print('=' * 58)
    for level in ('conservative', 'moderate', 'aggressive'):
        ratio = opt.stock_ratio_for_profile(level)
        print(f"  {opt.PROFILE_KO[level]:>4}: stock_ratio = {ratio:.2f}  "
              f"(KOSPI {ratio:.0%} / KTB {1-ratio:.0%})")
