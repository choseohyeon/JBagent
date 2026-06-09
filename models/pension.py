import numpy as np
import pandas as pd
import requests
import os
from dataclasses import dataclass
from typing import NamedTuple
from dotenv import load_dotenv

load_dotenv()

# ── 국민연금 수령 규칙 (법령 고정값) ──────────────────────────────────────────
# 조기수령: 1개월 앞당길수록 0.5% 감액, 최대 5년(60개월) = 30% 감액
# 연기수령: 1개월 늦출수록 0.6% 증액, 최대 5년(60개월) = 36% 증액
EARLY_REDUCTION_PER_MONTH  = 0.005   # 0.5% / 월
DEFER_INCREASE_PER_MONTH   = 0.006   # 0.6% / 월
STANDARD_START_AGE         = 65      # 2033년 이후 출생자 기준
MAX_EARLY_YEARS            = 5
MAX_DEFER_YEARS            = 5


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class PensionProfile:
    current_age: int
    expected_monthly_pension: float   # 65세 기준 예상 월 수령액 (만원)
    life_expectancy: float            # 기대 잔여 수명 (년) — SurvivalModel 연결


class ScenarioResult(NamedTuple):
    claim_age: int                    # 수령 시작 연령
    monthly_amount: float             # 월 수령액 (만원)
    adjustment_rate: float            # 감액/증액률
    total_lifetime: float             # 기대 총수령액 (만원)
    breakeven_age: float              # 손익분기 연령 (65세 수령 대비)
    npv: float                        # 할인율 반영 NPV (만원)


# ── 핵심 계산 ─────────────────────────────────────────────────────────────────

class PensionOptimizer:
    """
    국민연금 수령 시기 최적화.
    조기/정시/연기 수령 시나리오를 비교하고 생존 확률을 결합해 최적 시기를 권고한다.
    """

    DISCOUNT_RATE = 0.025   # 물가상승률 수준의 할인율 (실질 기준)

    def __init__(self):
        self._nps_avg_by_age = self._fetch_nps_avg()

    # ── NPS API ───────────────────────────────────────────────────────────────

    def _fetch_nps_avg(self) -> dict:
        """
        공공데이터포털 국민연금 API에서 연령대별 평균 월 수령액을 가져온다.
        API 호출 실패 시 통계청 공표 기준값으로 폴백.
        """
        key = os.getenv('NPS_API_KEY', '')
        if not key:
            return self._fallback_avg()
        try:
            url = ('https://apis.data.go.kr/B552015/NpsBeneficiaryInfoService'
                   '/getBeneficiaryByAgeList')
            params = {
                'serviceKey': key,
                'numOfRows': '20',
                'pageNo': '1',
                'givKndCd': '10',    # 노령연금
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return self._fallback_avg()
            # XML 또는 JSON 파싱
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            result = {}
            for item in root.iter('item'):
                age_tag  = item.find('ageGrpCd')
                avg_tag  = item.find('avgMmAmt')
                if age_tag is not None and avg_tag is not None:
                    try:
                        age = int(age_tag.text)
                        avg = float(avg_tag.text) / 10_000  # 원 → 만원
                        result[age] = avg
                    except (ValueError, TypeError):
                        continue
            return result if result else self._fallback_avg()
        except Exception:
            return self._fallback_avg()

    @staticmethod
    def _fallback_avg() -> dict:
        """국민연금공단 2024년 통계 기준 연령대별 평균 월 수령액 (만원)"""
        return {
            60: 62.0,
            65: 78.0,
            70: 85.0,
            75: 72.0,
            80: 58.0,
        }

    # ── 수령액 계산 ───────────────────────────────────────────────────────────

    @staticmethod
    def monthly_amount(base_monthly: float, claim_age: int) -> tuple[float, float]:
        """
        수령 시작 연령에 따른 월 수령액과 조정률을 반환한다.

        Parameters
        ----------
        base_monthly : 65세 기준 월 수령액 (만원)
        claim_age    : 실제 수령 시작 연령 (60~70)

        Returns
        -------
        (월 수령액, 조정률)  조정률: 양수=증액, 음수=감액
        """
        months_diff = (claim_age - STANDARD_START_AGE) * 12

        if months_diff < 0:    # 조기수령
            months_early  = min(abs(months_diff), MAX_EARLY_YEARS * 12)
            adj_rate      = -EARLY_REDUCTION_PER_MONTH * months_early
        elif months_diff > 0:  # 연기수령
            months_defer  = min(months_diff, MAX_DEFER_YEARS * 12)
            adj_rate      = DEFER_INCREASE_PER_MONTH * months_defer
        else:
            adj_rate = 0.0

        return base_monthly * (1 + adj_rate), adj_rate

    def npv_pension(self, monthly: float, claim_age: int,
                    death_age: float, discount_rate: float = None,
                    current_age: int = None) -> float:
        """
        연금 수령 기간의 현재가치(NPV)를 계산한다.
        현재 시점 = current_age (미지정 시 claim_age).
        시나리오 간 공정 비교를 위해 반드시 current_age를 통일해야 한다.
        """
        if discount_rate is None:
            discount_rate = self.DISCOUNT_RATE
        base_age = current_age if current_age is not None else claim_age
        annual   = monthly * 12
        years    = max(death_age - claim_age, 0)
        if discount_rate == 0:
            pv_at_claim = annual * years
        else:
            pv_at_claim = annual * (1 - (1 + discount_rate) ** -years) / discount_rate
        # claim_age 시점의 PV → current_age 시점으로 할인
        wait_years = max(claim_age - base_age, 0)
        return pv_at_claim / (1 + discount_rate) ** wait_years

    # ── 손익분기 연령 ─────────────────────────────────────────────────────────

    @staticmethod
    def breakeven_age(base_monthly: float, claim_age: int) -> float:
        """
        65세 정시 수령 대비 누적 수령액이 같아지는 손익분기 연령.
        """
        base_annual   = base_monthly * 12
        actual, _     = PensionOptimizer.monthly_amount(base_monthly, claim_age)
        actual_annual = actual * 12

        if claim_age == STANDARD_START_AGE:
            return float(STANDARD_START_AGE)

        if claim_age < STANDARD_START_AGE:
            # 조기수령: 더 일찍 받지만 금액 적음
            # base starts at 65, early starts at claim_age
            # 65세 시점 이후: early_total(t) vs base_total(t)
            # early_total(t) = actual_annual*(t - claim_age)
            # base_total(t)  = base_annual*(t - 65)
            # break even: actual*(t-claim) = base*(t-65)
            # t*(actual - base) = actual*claim - base*65
            if actual_annual == base_annual:
                return float('inf')
            t = (actual_annual * claim_age - base_annual * STANDARD_START_AGE) \
                / (actual_annual - base_annual)
            return round(t, 1)
        else:
            # 연기수령: 늦게 시작하지만 금액 큼
            # actual*(t - claim) = base*(t - 65)
            if actual_annual == base_annual:
                return float('inf')
            t = (actual_annual * claim_age - base_annual * STANDARD_START_AGE) \
                / (actual_annual - base_annual)
            return round(t, 1)

    # ── 시나리오 비교 ─────────────────────────────────────────────────────────

    def compare_scenarios(self, profile: PensionProfile,
                          survival_fn=None) -> pd.DataFrame:
        """
        수령 시작 연령 60~70세 시나리오별 결과를 비교한다.

        Parameters
        ----------
        profile     : PensionProfile
        survival_fn : pd.Series (index=연령, values=생존확률) — 선택적
                      제공 시 총수령액을 생존확률 가중 기댓값으로 보정
        """
        base     = profile.expected_monthly_pension
        death_age = profile.current_age + profile.life_expectancy
        rows = []

        for claim_age in range(60, 71):
            monthly, adj_rate = self.monthly_amount(base, claim_age)

            # 기대 수령 기간 (생존함수 제공 시 가중 계산)
            if survival_fn is not None:
                # E[수령기간] = Σ S(t) for t = claim_age+1 ~ 100
                sf_from_claim = survival_fn[survival_fn.index >= claim_age]
                expected_years = float(sf_from_claim.values[1:].sum()) \
                                 if len(sf_from_claim) > 1 else 0.0
            else:
                expected_years = max(death_age - claim_age, 0)

            total = monthly * 12 * expected_years
            be    = self.breakeven_age(base, claim_age)
            npv   = self.npv_pension(monthly, claim_age, death_age,
                                     current_age=profile.current_age)

            rows.append(ScenarioResult(
                claim_age      = claim_age,
                monthly_amount = round(monthly, 1),
                adjustment_rate= adj_rate,
                total_lifetime = round(total, 0),
                breakeven_age  = be,
                npv            = round(npv, 0),
            ))

        return pd.DataFrame(rows)

    # ── 최적 시기 권고 ────────────────────────────────────────────────────────

    def recommend(self, profile: PensionProfile,
                  survival_fn=None) -> dict:
        """
        NPV 최대화 기준 최적 수령 시기와 근거를 반환한다.
        """
        df = self.compare_scenarios(profile, survival_fn)

        best_npv   = df.loc[df['npv'].idxmax()]
        best_total = df.loc[df['total_lifetime'].idxmax()]
        standard   = df[df['claim_age'] == STANDARD_START_AGE].iloc[0]

        return {
            'scenarios':          df,
            'best_by_npv':        best_npv,
            'best_by_total':      best_total,
            'standard':           standard,
            'npv_gain_vs_65':     round(best_npv['npv'] - standard['npv'], 0),
            'total_gain_vs_65':   round(best_total['total_lifetime']
                                        - standard['total_lifetime'], 0),
        }


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from models.survival import SurvivalModel, PersonalProfile

    optimizer = PensionOptimizer()
    model     = SurvivalModel()

    profile = PensionProfile(
        current_age=60,
        expected_monthly_pension=100.0,   # 월 100만원
        life_expectancy=23.2,             # SurvivalModel 기대여명
    )

    # 생존함수 연결
    surv_profile = PersonalProfile(sex='M', current_age=60)
    sf = model.get_survival_function(surv_profile)

    result = optimizer.recommend(profile, survival_fn=sf)
    df = result['scenarios']

    print('=' * 68)
    print('국민연금 수령 시기 시나리오 비교 (기준: 65세 월 100만원)')
    print('=' * 68)
    print(f"{'수령시작':>6} {'월수령액':>8} {'조정률':>7} "
          f"{'기대총수령':>12} {'손익분기':>8} {'NPV':>12}")
    print('-' * 68)
    for _, row in df.iterrows():
        marker = ' ◀ 최적(NPV)' if row['claim_age'] == result['best_by_npv']['claim_age'] \
                 else (' ◀ 최적(총액)' if row['claim_age'] == result['best_by_total']['claim_age']
                       and row['claim_age'] != result['best_by_npv']['claim_age'] else '')
        print(f"{row['claim_age']:>5}세  "
              f"{row['monthly_amount']:>7.1f}만  "
              f"{row['adjustment_rate']:>+6.1%}  "
              f"{row['total_lifetime']:>10,.0f}만  "
              f"{row['breakeven_age']:>7.1f}세  "
              f"{row['npv']:>10,.0f}만{marker}")

    print('=' * 68)
    best = result['best_by_npv']
    print(f"NPV 최적 수령 시작: {best['claim_age']}세  "
          f"(65세 대비 NPV +{result['npv_gain_vs_65']:,.0f}만원)")
    print(f"총액 최적 수령 시작: {result['best_by_total']['claim_age']}세  "
          f"(65세 대비 총액 +{result['total_gain_vs_65']:,.0f}만원)")
