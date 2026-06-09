import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PersonalProfile:
    sex: Literal['M', 'F']
    current_age: int
    smoking: bool = False
    chronic_disease: bool = False
    bmi: float = 22.0


class SurvivalModel:
    """
    2024 국민생명표 기반 생존 분석 모델.
    Cox PH 위험비로 개인 건강 변수를 반영해 S(t)를 개인화한다.

    출처: 통계청 2024년 생명표 보도자료 (완전생명표, 남자/여자)
    Cox HR 출처: 국내 역학 연구 메타분석 기반 추정치
    """

    # 2024 완전생명표 qx (사망확률), 연령 0~100 (index = 연령)
    # 100세+는 qx = 1.0 처리
    _QX_MALE = [
        0.00261, 0.00024, 0.00018, 0.00013, 0.00010, 0.00009, 0.00008, 0.00007,
        0.00006, 0.00006, 0.00008, 0.00009, 0.00012, 0.00014, 0.00016, 0.00018,
        0.00022, 0.00025, 0.00028, 0.00030, 0.00032, 0.00035, 0.00038, 0.00043,
        0.00047, 0.00051, 0.00055, 0.00059, 0.00062, 0.00065, 0.00068, 0.00071,
        0.00074, 0.00077, 0.00079, 0.00083, 0.00089, 0.00097, 0.00106, 0.00115,
        0.00123, 0.00133, 0.00143, 0.00154, 0.00167, 0.00181, 0.00196, 0.00214,
        0.00236, 0.00259, 0.00284, 0.00309, 0.00335, 0.00365, 0.00400, 0.00439,
        0.00477, 0.00513, 0.00553, 0.00596, 0.00645, 0.00699, 0.00759, 0.00819,
        0.00878, 0.00941, 0.01020, 0.01117, 0.01229, 0.01364, 0.01523, 0.01705,
        0.01910, 0.02120, 0.02360, 0.02609, 0.02895, 0.03224, 0.03632, 0.04188,
        0.04870, 0.05645, 0.06476, 0.07352, 0.08356, 0.09463, 0.10678, 0.12003,
        0.13441, 0.14993, 0.16661, 0.18444, 0.20340, 0.22346, 0.24435, 0.26662,
        0.28957, 0.31330, 0.33768, 0.36256, 1.00000,
    ]

    _QX_FEMALE = [
        0.00214, 0.00024, 0.00016, 0.00010, 0.00008, 0.00008, 0.00008, 0.00008,
        0.00006, 0.00005, 0.00005, 0.00006, 0.00008, 0.00011, 0.00014, 0.00016,
        0.00019, 0.00022, 0.00025, 0.00026, 0.00027, 0.00028, 0.00030, 0.00031,
        0.00031, 0.00031, 0.00033, 0.00035, 0.00038, 0.00040, 0.00041, 0.00041,
        0.00042, 0.00043, 0.00045, 0.00048, 0.00052, 0.00058, 0.00064, 0.00072,
        0.00078, 0.00083, 0.00086, 0.00088, 0.00090, 0.00094, 0.00102, 0.00111,
        0.00120, 0.00128, 0.00136, 0.00146, 0.00155, 0.00162, 0.00168, 0.00174,
        0.00183, 0.00194, 0.00207, 0.00218, 0.00230, 0.00247, 0.00270, 0.00299,
        0.00328, 0.00355, 0.00386, 0.00423, 0.00471, 0.00531, 0.00599, 0.00678,
        0.00770, 0.00876, 0.01008, 0.01156, 0.01335, 0.01547, 0.01812, 0.02152,
        0.02558, 0.03030, 0.03585, 0.04217, 0.04959, 0.05787, 0.06720, 0.07764,
        0.08924, 0.10206, 0.11613, 0.13147, 0.14809, 0.16596, 0.18505, 0.20529,
        0.22660, 0.24885, 0.27190, 0.29559, 1.00000,
    ]

    # Cox PH 위험비 (Hazard Ratio)
    _HR = {
        'smoking':       1.50,
        'chronic':       1.35,
        'bmi_obese':     1.25,   # BMI >= 30
        'bmi_under':     1.40,   # BMI < 18.5
    }

    def __init__(self):
        self._lx_male   = self._qx_to_lx(self._QX_MALE)
        self._lx_female = self._qx_to_lx(self._QX_FEMALE)

    # ── 내부 유틸 ──────────────────────────────────────────────────────────

    @staticmethod
    def _qx_to_lx(qx: list) -> np.ndarray:
        """qx 리스트 → lx (출생 기준 생존율, 0~100세)"""
        lx = np.ones(len(qx))
        for i in range(1, len(qx)):
            lx[i] = lx[i - 1] * (1.0 - qx[i - 1])
        return lx

    def _combined_hr(self, profile: PersonalProfile) -> float:
        hr = 1.0
        if profile.smoking:
            hr *= self._HR['smoking']
        if profile.chronic_disease:
            hr *= self._HR['chronic']
        if profile.bmi >= 30.0:
            hr *= self._HR['bmi_obese']
        elif profile.bmi < 18.5:
            hr *= self._HR['bmi_under']
        return hr

    def _lx_for(self, sex: Literal['M', 'F']) -> np.ndarray:
        return self._lx_male if sex == 'M' else self._lx_female

    # ── 공개 API ───────────────────────────────────────────────────────────

    def get_survival_function(self, profile: PersonalProfile) -> pd.Series:
        """
        개인화된 조건부 생존 함수 S(t | current_age 생존).
        Cox PH 조정: S_adj(t) = S_base(t) ^ HR

        Returns
        -------
        pd.Series  index=연령(current_age~100), values=생존확률
        """
        lx  = self._lx_for(profile.sex)
        hr  = self._combined_hr(profile)
        ages = np.arange(profile.current_age, 101)

        s_base     = lx[ages] / lx[profile.current_age]
        s_adjusted = np.power(np.clip(s_base, 0, 1), hr)

        return pd.Series(s_adjusted, index=ages, name='survival_prob')

    def sample_death_age(self, profile: PersonalProfile,
                         n_samples: int = 10_000) -> np.ndarray:
        """
        Monte Carlo용 사망 연령 샘플링 (역변환 샘플링).

        Returns
        -------
        np.ndarray  shape=(n_samples,), 사망 연령 정수 배열
        """
        sf   = self.get_survival_function(profile)
        ages = sf.index.values

        # 연령별 사망 확률 P(age = t) = S(t) - S(t+1)
        death_probs = np.diff(sf.values, append=0) * -1
        death_probs = np.maximum(death_probs, 0)
        death_probs /= death_probs.sum()

        return np.random.choice(ages, size=n_samples, p=death_probs)

    def death_prob_before(self, profile: PersonalProfile,
                          target_age: int) -> float:
        """P(target_age 이전 사망 | current_age 생존)"""
        if target_age <= profile.current_age:
            return 0.0
        sf = self.get_survival_function(profile)
        return float(1.0 - sf.get(min(target_age, 100), 0.0))

    def expected_remaining_life(self, profile: PersonalProfile) -> float:
        """현재 연령 기준 기대 잔여 수명 (년)"""
        sf   = self.get_survival_function(profile)
        ages = sf.index.values
        # E[T] = sum S(t) for t = current_age+1 ~ 100
        return float(sf.values[1:].sum())


# ── 동작 확인 ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = SurvivalModel()

    profiles = [
        PersonalProfile(sex='M', current_age=60, smoking=False, chronic_disease=False, bmi=23.0),
        PersonalProfile(sex='M', current_age=60, smoking=True,  chronic_disease=True,  bmi=28.0),
        PersonalProfile(sex='F', current_age=60, smoking=False, chronic_disease=False, bmi=22.0),
    ]
    labels = ['60세 남성 건강', '60세 남성 흡연+만성질환', '60세 여성 건강']

    print('=' * 55)
    print(f'{"프로필":<22} {"기대여명":>6} {"83세前사망확률":>12} {"90세前사망확률":>12}')
    print('-' * 55)
    for p, label in zip(profiles, labels):
        exp = model.expected_remaining_life(p)
        p83 = model.death_prob_before(p, 83)
        p90 = model.death_prob_before(p, 90)
        print(f'{label:<22} {exp:>6.1f}년  {p83:>11.1%}  {p90:>11.1%}')
    print('=' * 55)

    # Monte Carlo 샘플 분포 확인
    base = PersonalProfile(sex='M', current_age=60)
    samples = model.sample_death_age(base, n_samples=10_000)
    print(f'\n[Monte Carlo 10,000회] 60세 남성 기본')
    print(f'  중앙값 사망연령: {np.median(samples):.0f}세')
    print(f'  10/50/90 퍼센타일: {np.percentile(samples, [10,50,90])}')
