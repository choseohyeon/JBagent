import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# 군집화에 사용할 컬럼 (가구마스터 EUC-KR 컬럼명)
FEATURE_COLS = {
    'age':        '가구주_만연령',
    'net_assets': '순자산',
    'income':     '경상소득(보완)',
    'members':    '가구원수',
}

PERCENTILE_LABELS = {
    10: '하위 10%',
    25: '하위 25%',
    50: '중앙값',
    75: '상위 25%',
    90: '상위 10%',
}


@dataclass
class UserProfile:
    age: int
    net_assets: float    # 순자산 (만원)
    income: float        # 연간 경상소득 (만원)
    members: int         # 가구원수


@dataclass
class ClusterResult:
    cluster_id: int
    cluster_size: int
    cluster_ratio: float          # 전체 대비 해당 군집 비율
    percentile_in_cluster: dict   # 군집 내 순자산/소득 퍼센타일
    user_percentile: dict         # 사용자의 군집 내 퍼센타일 위치
    top30_profile: dict           # 군집 내 상위 30% 평균 프로파일
    silhouette: float             # 군집 품질 지표


class HouseholdClusterModel:
    """
    통계청 가계금융복지조사 기반 K-means 군집 분석.
    사용자 가구와 유사한 또래 집단을 찾아 재무 상태 퍼센타일을 산출한다.
    """

    # K=5 verified via silhouette analysis (K=2..9, 2026-06-06):
    # K=2:0.3675  K=3:0.3591  K=4:0.3651  K=5:0.3746*  K=6:0.3165  K=7:0.3154  K=8:0.3133  K=9:0.2870
    OPTIMAL_K    = 5
    RANDOM_STATE = 42
    K_SILHOUETTE = {2: 0.3675, 3: 0.3591, 4: 0.3651, 5: 0.3746,
                    6: 0.3165, 7: 0.3154, 8: 0.3133, 9: 0.2870}

    def __init__(self):
        self.scaler  = StandardScaler()
        self.kmeans  = None
        self.df_raw  = None     # 원본 데이터 (만원 단위)
        self.df_feat = None     # 스케일된 피처 행렬
        self.labels_ = None
        self.silhouette_ = None
        self._fit()

    # ── K 선택 검증 ──────────────────────────────────────────────────────────

    @staticmethod
    def find_optimal_k(X_scaled: np.ndarray,
                       k_range=range(2, 10),
                       sample_size: int = 5_000,
                       random_state: int = 42) -> tuple[int, dict]:
        """
        Silhouette-score-based optimal K selection.
        Call this to re-verify OPTIMAL_K if the dataset changes.
        Returns (best_k, {k: silhouette_score}).
        """
        rng  = np.random.RandomState(random_state)
        idx  = rng.choice(len(X_scaled), min(sample_size, len(X_scaled)), replace=False)
        scores = {}
        for k in k_range:
            km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            scores[k] = float(silhouette_score(X_scaled[idx], labels[idx]))
        return max(scores, key=scores.get), scores

    # ── 데이터 로드 ───────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        path = next(DATA_DIR.glob("*가구마스터*.csv"))
        df = pd.read_csv(path, encoding='euc-kr', low_memory=False)

        # 필요 컬럼 추출 및 이름 정리
        rename = {v: k for k, v in FEATURE_COLS.items()}
        df = df[list(FEATURE_COLS.values())].rename(columns=rename)

        # 수치 변환 및 결측 제거
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        # 기본 이상치 제거: 연령 20~90, 소득 > 0, 가구원수 1~10
        df = df[
            (df['age'].between(20, 90)) &
            (df['income'] > 0) &
            (df['members'].between(1, 10))
        ]

        # 극단 이상치 윈저화 (99.9th percentile 상한 적용)
        # K-means는 이상치에 민감 — 단 1개 극단 가구가 군집을 왜곡할 수 있음
        for col in ('net_assets', 'income'):
            cap = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=cap)

        return df.reset_index(drop=True)

    # ── 모델 학습 ─────────────────────────────────────────────────────────────

    def _fit(self):
        self.df_raw  = self._load_data()
        X_raw        = self.df_raw[list(FEATURE_COLS.keys())].values

        # 표준화 (단위 차이 보정)
        X_scaled     = self.scaler.fit_transform(X_raw)
        self.df_feat = pd.DataFrame(X_scaled,
                                    columns=list(FEATURE_COLS.keys()))

        # K-means 학습
        self.kmeans  = KMeans(n_clusters=self.OPTIMAL_K,
                              random_state=self.RANDOM_STATE,
                              n_init=10)
        self.labels_ = self.kmeans.fit_predict(X_scaled)
        self.df_raw['cluster'] = self.labels_

        # 실루엣 점수 (샘플 최대 5,000개로 제한 — 속도)
        n_sample = min(5_000, len(X_scaled))
        idx      = np.random.RandomState(42).choice(
                       len(X_scaled), n_sample, replace=False)
        self.silhouette_ = silhouette_score(X_scaled[idx],
                                            self.labels_[idx])

    # ── 사용자 군집 배정 ──────────────────────────────────────────────────────

    def _assign_cluster(self, profile: UserProfile) -> int:
        x = np.array([[profile.age, profile.net_assets,
                        profile.income, profile.members]])
        return int(self.kmeans.predict(self.scaler.transform(x))[0])

    # ── 퍼센타일 계산 ─────────────────────────────────────────────────────────

    @staticmethod
    def _percentile_rank(series: pd.Series, value: float) -> float:
        """value가 series 내에서 몇 퍼센타일인지 반환 (0~100)"""
        return float((series < value).mean() * 100)

    # ── 분석 ─────────────────────────────────────────────────────────────────

    def analyze(self, profile: UserProfile) -> ClusterResult:
        """
        사용자 프로파일을 군집에 배정하고 또래 대비 재무 상태를 분석한다.
        """
        cid    = self._assign_cluster(profile)
        df_c   = self.df_raw[self.df_raw['cluster'] == cid]

        # 군집 내 퍼센타일 분포
        pct_in_cluster = {}
        for col in ('net_assets', 'income'):
            pct_in_cluster[col] = {
                p: round(float(np.percentile(df_c[col], p)), 0)
                for p in (10, 25, 50, 75, 90)
            }

        # 사용자의 군집 내 위치
        user_pct = {
            'net_assets': round(self._percentile_rank(
                df_c['net_assets'], profile.net_assets), 1),
            'income': round(self._percentile_rank(
                df_c['income'], profile.income), 1),
        }

        # 상위 30% 프로파일
        top30_mask  = df_c['net_assets'] >= df_c['net_assets'].quantile(0.70)
        top30       = df_c[top30_mask]
        top30_profile = {
            'avg_net_assets': round(float(top30['net_assets'].mean()), 0),
            'avg_income':     round(float(top30['income'].mean()), 0),
            'avg_age':        round(float(top30['age'].mean()), 1),
            'avg_members':    round(float(top30['members'].mean()), 1),
        }

        return ClusterResult(
            cluster_id         = cid,
            cluster_size       = len(df_c),
            cluster_ratio      = round(len(df_c) / len(self.df_raw), 3),
            percentile_in_cluster = pct_in_cluster,
            user_percentile    = user_pct,
            top30_profile      = top30_profile,
            silhouette         = round(self.silhouette_, 3),
        )

    def cluster_summary(self) -> pd.DataFrame:
        """전체 군집별 요약 통계"""
        rows = []
        for cid in range(self.OPTIMAL_K):
            df_c = self.df_raw[self.df_raw['cluster'] == cid]
            rows.append({
                'cluster':     cid,
                'size':        len(df_c),
                'age_mean':    round(df_c['age'].mean(), 1),
                'assets_med':  round(df_c['net_assets'].median(), 0),
                'income_med':  round(df_c['income'].median(), 0),
                'members_med': round(df_c['members'].median(), 1),
            })
        return pd.DataFrame(rows).sort_values('assets_med', ascending=False)


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("모델 학습 중 (K-means K=5)...")
    model = HouseholdClusterModel()

    print(f"데이터: {len(model.df_raw):,}가구  |  "
          f"실루엣 점수: {model.silhouette_:.3f}\n")

    # 군집 요약
    print("=" * 60)
    print("군집별 요약 (순자산 중앙값 내림차순)")
    print("=" * 60)
    summary = model.cluster_summary()
    print(f"{'군집':>4}  {'가구수':>6}  {'평균연령':>6}  "
          f"{'순자산중앙값':>10}  {'소득중앙값':>8}  {'가구원수':>6}")
    print('-' * 60)
    for _, row in summary.iterrows():
        print(f"  {int(row['cluster']):>2}   {int(row['size']):>5}가구  "
              f"{row['age_mean']:>6.1f}세  "
              f"{row['assets_med']:>9,.0f}만  "
              f"{row['income_med']:>7,.0f}만  "
              f"{row['members_med']:>5.1f}명")

    # 사용자 분석 예시 3가지
    profiles = [
        ('60세 자산상위', UserProfile(age=60, net_assets=50_000,
                                      income=6_000, members=2)),
        ('60세 중간층',   UserProfile(age=60, net_assets=20_000,
                                      income=4_000, members=2)),
        ('45세 4인가구',  UserProfile(age=45, net_assets=10_000,
                                      income=5_500, members=4)),
    ]

    for label, profile in profiles:
        result = model.analyze(profile)
        print()
        print(f"{'='*60}")
        print(f"사용자: {label}  →  군집 {result.cluster_id} "
              f"({result.cluster_size:,}가구, 전체의 {result.cluster_ratio:.1%})")
        print(f"{'='*60}")
        print(f"  순자산 퍼센타일: {result.user_percentile['net_assets']:.0f}%ile")
        print(f"  소득 퍼센타일:   {result.user_percentile['income']:.0f}%ile")
        print(f"\n  [군집 내 순자산 분포]")
        for p, v in result.percentile_in_cluster['net_assets'].items():
            marker = " ← 본인" if abs(v - profile.net_assets) == \
                min(abs(vv - profile.net_assets)
                    for vv in result.percentile_in_cluster['net_assets'].values()) \
                else ""
            print(f"    {PERCENTILE_LABELS[p]:>8}: {v:>8,.0f}만원{marker}")
        print(f"\n  [상위 30% 벤치마크]")
        tp = result.top30_profile
        print(f"    평균 순자산: {tp['avg_net_assets']:>8,.0f}만원  "
              f"(본인 대비 {tp['avg_net_assets']/max(profile.net_assets,1):.1f}배)")
        print(f"    평균 소득:   {tp['avg_income']:>8,.0f}만원")
        print(f"    평균 연령:   {tp['avg_age']:>6.1f}세  |  "
              f"평균 가구원수: {tp['avg_members']:.1f}명")
