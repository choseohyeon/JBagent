import numpy as np
import pandas as pd
import xlrd
from pathlib import Path
from dataclasses import dataclass
from typing import NamedTuple

from models.survival import SurvivalModel, PersonalProfile

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# ── 데이터 클래스 ────────────────────────────────────────────────────────────

@dataclass
class SimulationInput:
    profile: PersonalProfile
    initial_assets: float      # 초기 자산 (만원)
    monthly_expense: float     # 월 기본 지출 (만원)
    pension_monthly: float     # 월 연금 수령액 (만원)
    pension_start_age: int     # 연금 수령 시작 연령
    stock_ratio: float = 0.4   # 주식 비중 (0~1), 나머지는 채권
    n_simulations: int = 10_000
    monthly_income: float = 0.0    # 월 근로소득 (만원) — 0이면 없음
    income_until_age: int = 60     # 근로소득 종료 연령


class SimulationResult(NamedTuple):
    depletion_prob: float            # 자산 고갈 확률
    percentile_paths: pd.DataFrame   # p10/p50/p90 연령별 자산 경로
    depletion_age_dist: np.ndarray   # 고갈 발생 연령 배열
    median_final_assets: float       # 중앙값 최종 자산 (만원)


@dataclass
class DistributionParams:
    stock_mean: float
    stock_std:  float
    bond_mean:  float
    bond_std:   float
    corr:       float
    infl_mean:  float
    infl_std:   float


# ── 데이터 로더 ──────────────────────────────────────────────────────────────

class DataLoader:
    """KRX·ECOS 파일에서 연간 수익률/물가상승률 분포 추정"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    # ── CPI ──────────────────────────────────────────────────────────────────

    def _load_cpi(self) -> pd.Series:
        path = next(self.data_dir.glob("소비자물가지수*.csv"))
        df = pd.read_csv(path, encoding='utf-8', header=0)
        row = df.iloc[0]  # 총지수 행
        date_cols = [c for c in df.columns if '/' in str(c)]
        s = pd.to_numeric(row[date_cols], errors='coerce').dropna()
        s.index = pd.to_datetime(s.index, format='%Y/%m')
        return s.sort_index()

    # ── 시장금리 (국고채 3년) ─────────────────────────────────────────────────

    def _load_bond_rate(self) -> pd.Series:
        path = next(self.data_dir.glob("시장금리*.csv"))
        df = pd.read_csv(path, encoding='utf-8', header=0)
        mask = (df.iloc[:, 1].str.contains('국고채', na=False) &
                df.iloc[:, 1].str.contains('3년', na=False))
        row = df[mask].iloc[0]
        date_cols = [c for c in df.columns if '/' in str(c)]
        s = pd.to_numeric(row[date_cols], errors='coerce').dropna()
        s.index = pd.to_datetime(s.index, format='%Y/%m')
        return (s / 100).sort_index()  # % → 소수

    # ── KOSPI 주가지수 ────────────────────────────────────────────────────────

    def _load_stock_returns(self) -> pd.Series:
        path = self.data_dir / "KOSPI.xls"
        wb = xlrd.open_workbook(str(path))
        ws = wb.sheet_by_index(0)
        dates, prices = [], []
        for r in range(1, ws.nrows):
            row = ws.row_values(r)
            try:
                dt    = pd.to_datetime(str(row[0]).strip())
                price = float(str(row[1]).replace(',', ''))
                dates.append(dt)
                prices.append(price)
            except Exception:
                continue
        s = pd.Series(prices, index=dates).sort_index()
        monthly = s.resample('ME').last()
        return monthly.pct_change().dropna()

    # ── KTB 채권지수 ──────────────────────────────────────────────────────────

    def _load_ktb_returns(self) -> pd.Series:
        ktb_dir = self.data_dir / "KTB 지수"
        frames = []
        for f in sorted(ktb_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f, encoding='utf-8', thousands=',')
                # 컬럼 0: 날짜, 컬럼 1: 총수익지수
                df = df.iloc[:, [0, 1]].copy()
                df.columns = ['date', 'total']
                df['date']  = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
                df['total'] = pd.to_numeric(df['total'], errors='coerce')
                frames.append(df.dropna())
            except Exception:
                continue
        combined = (pd.concat(frames)
                      .set_index('date')
                      .sort_index()
                      .pipe(lambda x: x[~x.index.duplicated(keep='last')]))
        monthly = combined['total'].resample('ME').last()
        return monthly.pct_change().dropna()

    # ── 분포 추정 ─────────────────────────────────────────────────────────────

    def fit_distributions(self) -> DistributionParams:
        stock_m = self._load_stock_returns()
        bond_m  = self._load_ktb_returns()
        cpi     = self._load_cpi()

        # 연간 수익률 — 현재 진행 중인 연도는 월 데이터가 불완전해
        # resample 시 극단값이 발생하므로 완성된 연도만 사용
        current_year = pd.Timestamp.today().year
        stock_a = (1 + stock_m).resample('YE').prod() - 1
        bond_a  = (1 + bond_m).resample('YE').prod() - 1
        stock_a = stock_a[stock_a.index.year < current_year]
        bond_a  = bond_a[bond_a.index.year < current_year]

        # 물가상승률 (전년동월비)
        infl = cpi.pct_change(12).dropna()

        # 공통 기간 상관계수
        common = stock_a.index.intersection(bond_a.index)
        corr = float(np.corrcoef(stock_a.loc[common], bond_a.loc[common])[0, 1])

        return DistributionParams(
            stock_mean = float(stock_a.mean()),
            stock_std  = float(stock_a.std()),
            bond_mean  = float(bond_a.mean()),
            bond_std   = float(bond_a.std()),
            corr       = corr,
            infl_mean  = float(infl.mean()),
            infl_std   = float(infl.std()),
        )


# ── Monte Carlo 시뮬레이터 ────────────────────────────────────────────────────

class MonteCarloSimulator:
    """
    생애 현금흐름 Monte Carlo 시뮬레이션.

    확률 변수:
      - 포트폴리오 수익률: 다변량 정규분포 (주식·채권 상관 반영)
      - 물가상승률: 정규분포 (ECOS CPI 역사적 분포)
      - 의료비 충격: Poisson 발생 × 로그정규 크기
      - 수명: SurvivalModel 샘플링
    """

    # Age-stratified Poisson rate (events/year)
    # <70: 0.05 (1 event per 20 yrs), 70-79: 0.10, 80+: 0.20
    # Rationale: NHIS data shows ~2x medical cost increase per decade after 65
    MEDICAL_LAMBDA = 0.08   # kept for backward compatibility (not used in run())
    MEDICAL_MEAN   = 500.0  # 평균 충격 500만원
    MEDICAL_STD    = 300.0  # 표준편차

    def __init__(self, params: DistributionParams,
                 survival_model: SurvivalModel = None):
        self.p = params
        self.survival = survival_model or SurvivalModel()

    # ── 내부 샘플러 ───────────────────────────────────────────────────────────

    def _sample_portfolio_returns(self, n_years: int, n_sims: int,
                                   stock_ratio: float) -> np.ndarray:
        """다변량 정규분포로 포트폴리오 수익률 샘플링 → shape (n_sims, n_years)"""
        p = self.p
        cov = np.array([
            [p.stock_std**2,
             p.corr * p.stock_std * p.bond_std],
            [p.corr * p.stock_std * p.bond_std,
             p.bond_std**2],
        ])
        raw = np.random.multivariate_normal(
            [p.stock_mean, p.bond_mean], cov,
            size=n_sims * n_years
        ).reshape(n_sims, n_years, 2)
        return stock_ratio * raw[:, :, 0] + (1 - stock_ratio) * raw[:, :, 1]

    def _sample_inflation(self, n_years: int, n_sims: int) -> np.ndarray:
        raw = np.random.normal(self.p.infl_mean, self.p.infl_std,
                               size=(n_sims, n_years))
        return np.clip(raw, -0.02, 0.15)

    def _sample_medical_shocks(self, n_years: int, n_sims: int,
                               base_age: int = 65) -> np.ndarray:
        """Age-varying Poisson(lambda) x LogNormal magnitude -> annual shock (10k KRW).
        Lambda schedule: age<70 -> 0.05, 70-79 -> 0.10, 80+ -> 0.20
        """
        ages    = base_age + np.arange(n_years)
        lambdas = np.where(ages < 70, 0.05, np.where(ages < 80, 0.10, 0.20))
        sigma   = np.sqrt(np.log(1 + (self.MEDICAL_STD / self.MEDICAL_MEAN) ** 2))
        mu      = np.log(self.MEDICAL_MEAN) - sigma ** 2 / 2
        counts  = np.column_stack([np.random.poisson(lam, n_sims) for lam in lambdas])
        sizes   = np.random.lognormal(mu, sigma, (n_sims, n_years))
        return counts * sizes

    # ── 핵심 시뮬레이션 ────────────────────────────────────────────────────────

    def run(self, sim_input: SimulationInput) -> SimulationResult:
        inp      = sim_input
        N        = inp.n_simulations
        MAX_AGE  = 100
        base_age = inp.profile.current_age
        max_yrs  = MAX_AGE - base_age

        # 1. 수명 샘플링
        death_ages = self.survival.sample_death_age(inp.profile, N)
        sim_yrs    = np.clip(death_ages - base_age, 1, max_yrs).astype(int)

        # 2. 전체 경로 랜덤 변수 일괄 샘플링
        port_r  = self._sample_portfolio_returns(max_yrs, N, inp.stock_ratio)
        infl    = self._sample_inflation(max_yrs, N)
        medical = self._sample_medical_shocks(max_yrs, N, base_age)

        # 3. 연령별 자산 경로 추적
        ages         = np.arange(base_age, MAX_AGE + 1)
        n_ages       = len(ages)
        asset_paths  = np.full((N, n_ages), np.nan)
        asset_paths[:, 0] = inp.initial_assets

        base_annual_exp  = inp.monthly_expense * 12
        depletion_ages   = np.full(N, MAX_AGE + 1, dtype=int)

        for yr in range(max_yrs):
            alive   = yr < sim_yrs                      # (N,) bool
            current = asset_paths[:, yr]

            # 누적 물가 (연금·지출 모두에 적용)
            cum_infl = np.prod(1 + infl[:, :yr], axis=1) if yr > 0 \
                       else np.ones(N)

            # 연금: 국민연금은 물가 연동 (실질 구매력 유지)
            pension_yr = np.where(
                base_age + yr >= inp.pension_start_age,
                inp.pension_monthly * 12 * cum_infl, 0.0
            )

            # 근로소득: income_until_age 이전까지만
            income_yr = np.where(
                base_age + yr < inp.income_until_age,
                inp.monthly_income * 12, 0.0
            )

            # 지출: 물가 반영
            expense_yr = base_annual_exp * cum_infl

            net_draw   = expense_yr + medical[:, yr] - pension_yr - income_yr
            new_assets = current * (1 + port_r[:, yr]) - net_draw

            # 최초 고갈 기록
            first_depletion = alive & (current > 0) & (new_assets <= 0)
            depletion_ages[first_depletion] = base_age + yr + 1

            asset_paths[:, yr + 1] = np.where(
                alive,
                np.maximum(new_assets, 0.0),
                np.nan
            )

        # 4. 결과 집계
        depleted      = depletion_ages <= MAX_AGE
        depletion_prob = float(depleted.mean())

        # 사망 경로(NaN) → 0 처리: 모든 경로 포함해 퍼센타일 계산
        # (생존 경로만 쓰면 일찍 사망한 경로가 빠져 낙관적 편향 발생)
        asset_paths_filled = np.where(np.isnan(asset_paths), 0.0, asset_paths)

        pct_data = {}
        for pct in [10, 50, 90]:
            vals = [float(np.percentile(asset_paths_filled[:, col], pct))
                    for col in range(n_ages)]
            pct_data[f'p{pct}'] = vals

        percentile_df = pd.DataFrame(pct_data, index=ages)

        # 최종 자산: 각 경로의 사망 연령 시점 자산 중앙값
        # (age 100 기준은 대부분 사망 후라 중앙값=0이 되는 문제 방지)
        final_assets_at_death = np.array([
            asset_paths_filled[i, min(sim_yrs[i], max_yrs)]
            for i in range(N)
        ])
        median_final = float(np.median(final_assets_at_death))

        return SimulationResult(
            depletion_prob       = depletion_prob,
            percentile_paths     = percentile_df,
            depletion_age_dist   = depletion_ages[depleted],
            median_final_assets  = median_final,
        )

    # ── Sequence-of-Returns Risk Analysis ────────────────────────────────────

    def sequence_risk_report(self, sim_input: 'SimulationInput',
                             front_years: int = 5,
                             n_sims: int = 3_000) -> pd.DataFrame:
        """
        Quantifies sequence-of-returns risk by stratifying paths on
        their first `front_years` average portfolio return.

        Returns DataFrame with columns:
            quartile | avg_early_return | depletion_prob | n_paths

        Key insight: paths with the same long-run average return can have
        very different depletion outcomes depending on when bad returns occur.
        Early bad returns deplete capital before it can recover → much higher
        depletion probability for Q1 vs Q4 paths.
        """
        inp      = sim_input
        N        = n_sims
        MAX_AGE  = 100
        base_age = inp.profile.current_age
        max_yrs  = MAX_AGE - base_age

        # Sample all random variables
        death_ages = self.survival.sample_death_age(inp.profile, N)
        sim_yrs    = np.clip(death_ages - base_age, 1, max_yrs).astype(int)
        port_r     = self._sample_portfolio_returns(max_yrs, N, inp.stock_ratio)
        infl       = self._sample_inflation(max_yrs, N)
        medical    = self._sample_medical_shocks(max_yrs, N, base_age)

        # Run asset-path simulation (same logic as run())
        assets          = np.full(N, inp.initial_assets, dtype=float)
        depletion_ages  = np.full(N, MAX_AGE + 1, dtype=int)
        base_annual_exp = inp.monthly_expense * 12

        for yr in range(max_yrs):
            alive    = yr < sim_yrs
            cum_infl = np.prod(1 + infl[:, :yr], axis=1) if yr > 0 else np.ones(N)
            pension  = np.where(base_age + yr >= inp.pension_start_age,
                                inp.pension_monthly * 12 * cum_infl, 0.0)
            income   = np.where(base_age + yr < inp.income_until_age,
                                inp.monthly_income * 12, 0.0)
            net_draw = base_annual_exp * cum_infl + medical[:, yr] - pension - income
            new_a    = assets * (1 + port_r[:, yr]) - net_draw
            first_dep = alive & (assets > 0) & (new_a <= 0)
            depletion_ages[first_dep] = base_age + yr + 1
            assets = np.where(alive, np.maximum(new_a, 0.0), assets)

        depleted     = depletion_ages <= MAX_AGE
        early_return = port_r[:, :front_years].mean(axis=1)  # avg first-N-year return

        # Stratify into quartiles
        q_bounds = np.percentile(early_return, [0, 25, 50, 75, 100])
        labels   = [f'Q1 (worst {front_years}y)', f'Q2 (below avg)',
                    f'Q3 (above avg)',              f'Q4 (best {front_years}y)']
        rows = []
        for i, label in enumerate(labels):
            lo, hi = q_bounds[i], q_bounds[i + 1]
            mask   = (early_return >= lo) & (early_return <= hi if i == 3 else early_return < hi)
            n      = int(mask.sum())
            rows.append({
                'quartile':          label,
                'avg_early_return':  float(early_return[mask].mean()) if n else 0.0,
                'depletion_prob':    float(depleted[mask].mean())      if n else 0.0,
                'n_paths':           n,
            })
        return pd.DataFrame(rows)


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("데이터 로드 및 분포 추정 중...")
    loader = DataLoader()
    params = loader.fit_distributions()

    print(f"  주식 연수익률: {params.stock_mean:.1%} ± {params.stock_std:.1%}")
    print(f"  채권 연수익률: {params.bond_mean:.1%}  ± {params.bond_std:.1%}")
    print(f"  물가상승률:    {params.infl_mean:.1%}  ± {params.infl_std:.1%}")
    print(f"  주식-채권 상관: {params.corr:.2f}")

    simulator = MonteCarloSimulator(params)

    cases = [
        ("자산3억/지출250/연금100", SimulationInput(
            profile=PersonalProfile(sex='M', current_age=60),
            initial_assets=30_000, monthly_expense=250,
            pension_monthly=100,   pension_start_age=65,
            stock_ratio=0.4,
        )),
        ("자산5억/지출300/연금150", SimulationInput(
            profile=PersonalProfile(sex='F', current_age=60),
            initial_assets=50_000, monthly_expense=300,
            pension_monthly=150,   pension_start_age=65,
            stock_ratio=0.3,
        )),
    ]

    for label, sim_input in cases:
        print(f"\n[{label}] 시뮬레이션 실행 중...")
        result = simulator.run(sim_input)

        print(f"  자산 고갈 확률:   {result.depletion_prob:.1%}")
        print(f"  중앙값 최종자산:  {result.median_final_assets:,.0f}만원")
        print(f"  {'연령':>5}  {'p10':>10}  {'p50':>10}  {'p90':>10}")
        for age in [65, 70, 75, 80, 85, 90]:
            if age in result.percentile_paths.index:
                r = result.percentile_paths.loc[age]
                print(f"  {age}세  {r['p10']:>9,.0f}만  {r['p50']:>9,.0f}만  {r['p90']:>9,.0f}만")
