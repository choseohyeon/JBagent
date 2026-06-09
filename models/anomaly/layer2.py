"""
Layer 2: 소비 패턴 이상 탐지 (재무 계획 이탈 경보)
CUSUM + PELT Change Point Detection
가계동향조사 Prior 기반 정상 범위 설정
"""
import numpy as np
import pandas as pd
import ruptures as rpt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent.parent / "필요한 파일"

# 가계동향조사 지출 카테고리 매핑
CATEGORIES = ['식료품', '주거광열', '의료보건', '교통', '통신', '교육', '기타']
CAT_KO     = {
    'food':      '식료품',
    'housing':   '주거광열',
    'medical':   '의료보건',
    'transport': '교통',
    'telecom':   '통신',
    'education': '교육',
    'other':     '기타',
}


@dataclass
class SpendingAlert:
    category: str
    alert_type: str          # 'spike' | 'structural_change' | 'normal'
    current: float           # 이번 달 지출 (만원)
    baseline_mean: float     # 개인 기저선 평균
    national_mean: float     # 전국 평균 (가계동향조사)
    z_score: float
    change_points: List[int] # PELT 탐지된 변화점 인덱스
    message: str


class SpendingAnomalyDetector:
    """
    월별 카테고리별 지출 이상 탐지.
    - CUSUM: 기저선 대비 지속적 이탈 감지
    - PELT: 지출 구조 변화 시점 탐지
    - 가계동향조사 Prior: 국가 평균 대비 과다 지출 경보
    """

    CUSUM_K_RATIO = 0.5    # CUSUM 슬랙 = mean × 0.5
    CUSUM_H_RATIO = 3.0    # CUSUM 임계 = mean × 3.0
    SPIKE_Z_THRESHOLD = 2.5

    def __init__(self):
        self.national_stats = self._load_national_stats()
        self._cusum_states: Dict[str, float] = {c: 0.0 for c in CAT_KO}

    # ── 가계동향조사 로드 ──────────────────────────────────────────────────────

    def _load_national_stats(self) -> Dict[str, Dict]:
        """
        가계동향조사 2025년 연간 지출에서 카테고리별 월평균/표준편차 추출.
        파일 없으면 2024년 통계청 공표 기준값 사용.
        """
        try:
            import openpyxl
            path = next(DATA_DIR.glob("*연간*가계동향*.xlsx"))
            wb   = openpyxl.load_workbook(str(path), read_only=True)
            ws   = wb.worksheets[0]
            rows = list(ws.iter_rows(values_only=True))

            # 2025년 행 찾기 (엑셀에서 숫자가 float으로 읽힐 수 있으므로 int 변환)
            target_row = None
            for row in rows:
                try:
                    if int(row[0]) == 2025:
                        target_row = row
                        break
                except (TypeError, ValueError):
                    continue

            if target_row and len(target_row) >= 15:
                # 컬럼 순서 (가계동향조사 연간 지출 통계표):
                # 0:연도 1:총지출 2:소비지출계 3:식료품 4:주류담배 5:의류신발
                # 6:주거광열 7:가정용품 8:보건 9:교통 10:오락문화
                # 11:통신 12:교육 13:음식숙박 14:기타상품서비스 15:비소비지출
                # 단위: 천원/월 → /10 하면 만원/월
                def to_man(val):
                    return float(val or 0) / 10   # 천원 → 만원

                monthly = {
                    'food':      to_man(target_row[3]),
                    'housing':   to_man(target_row[6]),
                    'medical':   to_man(target_row[8]),
                    'transport': to_man(target_row[9]),
                    'telecom':   to_man(target_row[11]),
                    'education': to_man(target_row[12]),
                    'other':     to_man(target_row[14]),
                }
                return {k: {'mean': v, 'std': v * 0.3} for k, v in monthly.items()}
        except Exception:
            pass
        return self._fallback_stats()

    @staticmethod
    def _fallback_stats() -> Dict[str, Dict]:
        """통계청 2024년 가계동향조사 기준 월평균 지출 (만원)"""
        return {
            'food':      {'mean': 36.8, 'std': 11.0},
            'housing':   {'mean': 29.3, 'std':  9.5},
            'medical':   {'mean': 18.6, 'std':  8.0},
            'transport': {'mean': 27.9, 'std': 10.5},
            'telecom':   {'mean': 13.8, 'std':  4.0},
            'education': {'mean': 17.9, 'std': 12.0},
            'other':     {'mean': 38.1, 'std': 15.0},
        }

    # ── 개인 기저선 통계 ──────────────────────────────────────────────────────

    @staticmethod
    def _baseline_stats(series: np.ndarray):
        if len(series) < 3:
            return float(series.mean()), float(series.std()) + 1e-6
        return float(series.mean()), float(series.std()) + 1e-6

    # ── CUSUM ────────────────────────────────────────────────────────────────

    def _cusum_detect(self, cat: str, history: np.ndarray,
                      new_val: float) -> tuple[bool, float]:
        mu, _ = self._baseline_stats(history)
        k = mu * self.CUSUM_K_RATIO
        h = mu * self.CUSUM_H_RATIO
        self._cusum_states[cat] = max(
            0, self._cusum_states[cat] + new_val - mu - k
        )
        return self._cusum_states[cat] > h, self._cusum_states[cat]

    # ── PELT (Change Point Detection) ────────────────────────────────────────

    @staticmethod
    def _pelt_detect(series: np.ndarray, min_len: int = 3) -> List[int]:
        if len(series) < min_len * 2:
            return []
        try:
            algo   = rpt.Pelt(model='rbf', min_size=min_len, jump=1)
            result = algo.fit_predict(series.reshape(-1, 1), pen=3.0)
            return [r for r in result if r < len(series)]
        except Exception:
            return []

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def analyze(self,
                spending_history: Dict[str, List[float]],
                new_month: Dict[str, float]) -> List[SpendingAlert]:
        """
        Parameters
        ----------
        spending_history : {category: [월별 지출 리스트 (만원)]}  최소 6개월
        new_month        : {category: 이번 달 지출 (만원)}

        Returns
        -------
        카테고리별 SpendingAlert 리스트
        """
        alerts = []

        for cat, ko in CAT_KO.items():
            history = np.array(spending_history.get(cat, []))
            current = new_month.get(cat, 0.0)
            national = self.national_stats.get(cat, {'mean': 30, 'std': 10})

            if len(history) < 3:
                continue

            mu, sigma = self._baseline_stats(history)
            z = (current - mu) / sigma if sigma > 0 else 0.0

            # CUSUM 탐지
            cusum_alert, _ = self._cusum_detect(cat, history, current)

            # PELT 변화점
            full_series    = np.append(history, current)
            change_points  = self._pelt_detect(full_series)
            structural_chg = len(change_points) > 0 and \
                             change_points[-1] >= len(history) - 2

            # 판정
            if structural_chg and cusum_alert:
                alert_type = 'structural_change'
                message = (f"{ko} 지출 구조적 변화 탐지 "
                           f"(이전 평균 {mu:.0f}만 → 현재 {current:.0f}만)")
            elif abs(z) > self.SPIKE_Z_THRESHOLD:
                alert_type = 'spike'
                direction  = '급증' if z > 0 else '급감'
                message    = (f"{ko} 지출 {direction} "
                              f"(Z={z:.1f}, 전국평균 {national['mean']:.0f}만)")
            else:
                alert_type = 'normal'
                message    = f"{ko} 정상 범위"

            alerts.append(SpendingAlert(
                category       = cat,
                alert_type     = alert_type,
                current        = round(current, 1),
                baseline_mean  = round(mu, 1),
                national_mean  = round(national['mean'], 1),
                z_score        = round(z, 2),
                change_points  = change_points,
                message        = message,
            ))

        return alerts


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # 12개월 정상 지출 이력
    history = {
        'food':      list(rng.normal(35, 5, 12)),
        'housing':   list(rng.normal(28, 4, 12)),
        'medical':   list(rng.normal(15, 3, 12)),
        'transport': list(rng.normal(25, 5, 12)),
        'telecom':   list(rng.normal(13, 2, 12)),
        'education': list(rng.normal(10, 3, 12)),
        'other':     list(rng.normal(35, 8, 12)),
    }

    test_months = [
        ("정상 달",    {'food': 34, 'housing': 27, 'medical': 14,
                        'transport': 24, 'telecom': 13, 'education': 11, 'other': 36}),
        ("의료비 급등", {'food': 34, 'housing': 27, 'medical': 80,
                        'transport': 24, 'telecom': 13, 'education': 11, 'other': 36}),
        ("지출 전반 증가", {'food': 55, 'housing': 45, 'medical': 30,
                           'transport': 40, 'telecom': 20, 'education': 25, 'other': 60}),
    ]

    detector = SpendingAnomalyDetector()
    print(f"전국 평균 기준 (가계동향조사): food={detector.national_stats['food']['mean']:.0f}만\n")

    for month_label, new_month in test_months:
        print(f"=== {month_label} ===")
        alerts = detector.analyze(history, new_month)
        for a in alerts:
            if a.alert_type != 'normal':
                print(f"  [{a.alert_type.upper()}] {a.message}")
        normal_count = sum(1 for a in alerts if a.alert_type == 'normal')
        if normal_count == len(alerts):
            print("  정상 범위 — 이상 없음")

        # 이번 달을 이력에 추가
        for cat in history:
            history[cat].append(new_month.get(cat, history[cat][-1]))
        print()
