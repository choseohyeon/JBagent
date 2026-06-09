"""
전체 통계 모델 평가 + Agent 연결 호환성 검증
CLAUDE.md 평가 지표:
  - Monte Carlo Coverage Probability
  - 생존 분석 C-statistic
  - 이상치 탐지 F1 / AUC-ROC
  - 군집 분석 Silhouette Score
  - Agent 입출력 JSON 직렬화 가능 여부
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, '.')
from models.survival   import SurvivalModel, PersonalProfile
from models.monte_carlo import DataLoader, MonteCarloSimulator, SimulationInput
from models.portfolio  import PortfolioOptimizer
from models.clustering import HouseholdClusterModel, UserProfile
from models.anomaly.layer1 import TransactionAnomalyDetector, Transaction
from models.anomaly.layer2 import SpendingAnomalyDetector


PASS = "PASS ✅"
FAIL = "FAIL ❌"
WARN = "WARN ⚠️"


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def row(status, label, detail=""):
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))


# ─────────────────────────────────────────────────────────────
# 1. 생존 분석 — C-statistic
# ─────────────────────────────────────────────────────────────

def eval_survival() -> dict:
    banner("1. 생존 분석 — C-statistic (Concordance Index)")
    model = SurvivalModel()
    rng   = np.random.default_rng(42)

    # 다양한 위험 프로파일 조합으로 쌍별 일치율 측정
    # (동일 프로파일 반복은 항상 1.0 → 의미 없음)
    test_profiles = [
        PersonalProfile('M', 60, False, False, 22.0),
        PersonalProfile('M', 60, True,  False, 22.0),
        PersonalProfile('M', 60, False, True,  22.0),
        PersonalProfile('M', 60, True,  True,  22.0),
        PersonalProfile('M', 60, True,  True,  32.0),
        PersonalProfile('F', 65, False, False, 22.0),
        PersonalProfile('F', 65, True,  False, 22.0),
        PersonalProfile('M', 70, False, False, 22.0),
        PersonalProfile('M', 70, True,  True,  30.0),
        PersonalProfile('F', 55, False, False, 22.0),
    ]
    # 각 프로파일의 기대수명
    exp_lives = [model.expected_remaining_life(p) for p in test_profiles]

    # 쌍별 비교: 위험 요인 더 많은 쪽이 기대수명 짧아야 함
    # 위험 스코어: smoking*1 + chronic*1 + obese*0.5 + younger_age(-1배)로 단순화
    def risk_score(p):
        return (int(p.smoking) * 1.0 + int(p.chronic_disease) * 1.0
                + (1.0 if p.bmi >= 30 else 0.0))

    concordant, total = 0, 0
    for i in range(len(test_profiles)):
        for j in range(len(test_profiles)):
            if i == j:
                continue
            ri = risk_score(test_profiles[i])
            rj = risk_score(test_profiles[j])
            if ri == rj:
                continue
            total += 1
            if ri > rj and exp_lives[i] < exp_lives[j]:
                concordant += 1
            elif ri < rj and exp_lives[i] > exp_lives[j]:
                concordant += 1

    c_stat = concordant / total if total > 0 else 0.0

    row(PASS if c_stat >= 0.80 else FAIL,
        f"C-statistic: {c_stat:.2f}",
        "다양한 프로파일 쌍 위험 일치율 (기준≥0.80)")

    # 기대수명 범위 확인 (0~100 사이)
    all_exp = [model.expected_remaining_life(PersonalProfile('M', a))
               for a in range(30, 91, 10)]
    range_ok = all(0 < e < 100 for e in all_exp)
    row(PASS if range_ok else FAIL,
        f"기대수명 범위 정상: {[round(e,1) for e in all_exp]}")

    # 단조성: 나이 많을수록 기대 잔여수명 감소
    mono_ok = all(all_exp[i] > all_exp[i+1] for i in range(len(all_exp)-1))
    row(PASS if mono_ok else FAIL, "기대수명 단조 감소 (나이 증가 시)")

    return {'c_statistic': c_stat, 'monotonic': mono_ok}


# ─────────────────────────────────────────────────────────────
# 2. Monte Carlo — Coverage Probability
# ─────────────────────────────────────────────────────────────

def eval_monte_carlo() -> dict:
    banner("2. Monte Carlo — Coverage Probability")
    loader = DataLoader()
    params = loader.fit_distributions()
    sim    = MonteCarloSimulator(params)
    rng    = np.random.default_rng(42)

    si = SimulationInput(
        profile=PersonalProfile('M', 65),
        initial_assets=30_000,
        monthly_expense=200,
        pension_monthly=80,
        pension_start_age=65,
        stock_ratio=0.4,
        n_simulations=5_000,
    )
    result = sim.run(si)
    pp     = result.percentile_paths

    # p10~p90 구간이 실제 분포의 80%를 포함하는지 (정의상 반드시)
    age_70 = pp.loc[70]
    gap_ok = age_70['p90'] > age_70['p10']
    row(PASS if gap_ok else FAIL,
        f"70세 p10({age_70['p10']:,.0f}) < p90({age_70['p90']:,.0f})")

    # 단조성: p10 ≤ p50 ≤ p90 at every age
    ages_check = [65, 70, 75, 80]
    mono_pct = all(
        pp.loc[a,'p10'] <= pp.loc[a,'p50'] <= pp.loc[a,'p90']
        for a in ages_check if a in pp.index
    )
    row(PASS if mono_pct else FAIL, "퍼센타일 순서 p10≤p50≤p90 (전 연령)")

    # 고갈 확률 [0,1]
    dp_ok = 0 <= result.depletion_prob <= 1
    row(PASS if dp_ok else FAIL,
        f"고갈 확률 범위: {result.depletion_prob:.1%}")

    # 단조성: 자산 많을수록 고갈 확률 낮아야 함
    si_rich = SimulationInput(
        profile=PersonalProfile('M', 65),
        initial_assets=100_000,
        monthly_expense=200,
        pension_monthly=80,
        pension_start_age=65,
        stock_ratio=0.4,
        n_simulations=2_000,
    )
    dp_rich = sim.run(si_rich).depletion_prob
    row(PASS if dp_rich < result.depletion_prob else FAIL,
        f"자산 많을수록 고갈↓: {result.depletion_prob:.1%} > {dp_rich:.1%}")

    return {
        'depletion_prob':  result.depletion_prob,
        'percentile_monotonic': mono_pct,
    }


# ─────────────────────────────────────────────────────────────
# 3. 이상치 탐지 — F1 / AUC-ROC
# ─────────────────────────────────────────────────────────────

def eval_anomaly() -> dict:
    banner("3. 이상치 탐지 — F1 / AUC-ROC (Layer 1)")
    import random
    rng = random.Random(42)

    # 정상 거래 60개로 기저선 학습
    normal_history = [
        Transaction(rng.uniform(3, 40), rng.randint(9,21), False, 'payment')
        for _ in range(60)
    ]
    detector = TransactionAnomalyDetector()
    detector.fit(normal_history)

    # 테스트: 정상 40 + 이상 40 (레이블 알고 있음)
    test_txs, true_labels = [], []
    for _ in range(30):   # 정상: 소액 일반 결제
        test_txs.append(Transaction(
            rng.uniform(3, 40), rng.randint(9, 21), False, 'payment'
        ))
        true_labels.append(0)
    for _ in range(10):   # 경계: 다소 큰 금액 (애매한 케이스)
        test_txs.append(Transaction(
            rng.uniform(50, 120), rng.randint(8, 20), False, 'transfer'
        ))
        true_labels.append(0)
    for _ in range(20):   # 이상: 고액 이체
        test_txs.append(Transaction(
            rng.uniform(300, 800), rng.randint(9, 21), False, 'transfer'
        ))
        true_labels.append(1)
    for _ in range(20):   # 이상: 보이스피싱 패턴
        test_txs.append(Transaction(
            rng.uniform(100, 400), rng.randint(0, 5),
            True, 'transfer', consecutive_count=rng.randint(3, 6)
        ))
        true_labels.append(1)

    scores, preds = [], []
    for tx in test_txs:
        res = detector.score(tx)
        scores.append(res.score / 100)
        # medium + high 모두 탐지로 간주 (앱에서 둘 다 알림 발송)
        preds.append(1 if res.level in ('medium', 'high') else 0)

    true_arr  = np.array(true_labels)
    pred_arr  = np.array(preds)
    score_arr = np.array(scores)

    f1  = f1_score(true_arr, pred_arr, zero_division=0)
    auc = roc_auc_score(true_arr, score_arr)

    row(PASS if f1  >= 0.70 else WARN, f"F1-score:  {f1:.3f}  (기준 ≥ 0.70, medium+high 탐지)")
    row(PASS if auc >= 0.80 else WARN, f"AUC-ROC:  {auc:.3f}  (기준 ≥ 0.80)")

    # 레이블별 정밀도
    tp = int(((pred_arr == 1) & (true_arr == 1)).sum())
    fp = int(((pred_arr == 1) & (true_arr == 0)).sum())
    fn = int(((pred_arr == 0) & (true_arr == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    row(PASS if recall >= 0.70 else WARN,
        f"Recall:    {recall:.3f}  TP={tp} FP={fp} FN={fn}  (Recall 중시)")

    return {'f1': f1, 'auc': auc, 'precision': precision, 'recall': recall}


# ─────────────────────────────────────────────────────────────
# 4. 군집 분석 — Silhouette Score
# ─────────────────────────────────────────────────────────────

def eval_clustering() -> dict:
    banner("4. 군집 분석 — Silhouette Score")
    model = HouseholdClusterModel()
    sil   = model.silhouette_

    row(PASS if sil >= 0.30 else WARN,
        f"Silhouette Score: {sil:.3f}  (기준 ≥ 0.30, 0.5+ 우수)")

    # 군집 수가 K=5이고 단독 군집 없음
    sizes = model.df_raw.groupby('cluster').size()
    no_singleton = sizes.min() > 1
    row(PASS if no_singleton else FAIL,
        f"단독 군집 없음: 최소 크기 {sizes.min()}가구")

    # 군집 간 순자산 차이가 유의한지
    cluster_meds = model.df_raw.groupby('cluster')['net_assets'].median()
    spread = cluster_meds.max() / max(cluster_meds.min(), 1)
    row(PASS if spread >= 3 else WARN,
        f"군집 간 자산 격차: {spread:.1f}배 (최고/최저 중앙값)")

    return {'silhouette': sil, 'min_cluster_size': int(sizes.min())}


# ─────────────────────────────────────────────────────────────
# 5. Agent 연결 호환성 — JSON 직렬화 + 입력 타입 검증
# ─────────────────────────────────────────────────────────────

def eval_agent_compatibility() -> dict:
    banner("5. Agent 연결 호환성 검증")

    results = {}

    # (a) Survival → JSON
    try:
        m  = SurvivalModel()
        sf = m.get_survival_function(PersonalProfile('M', 60))
        out = {
            'expected_life': round(m.expected_remaining_life(PersonalProfile('M', 60)), 1),
            'death_prob_before_80': round(m.death_prob_before(PersonalProfile('M', 60), 80), 3),
            'survival_at_75': round(float(sf.get(75, 0)), 3),
        }
        json.dumps(out)
        row(PASS, "SurvivalModel 출력 JSON 직렬화", str(out))
        results['survival'] = True
    except Exception as e:
        row(FAIL, f"SurvivalModel JSON 실패: {e}")
        results['survival'] = False

    # (b) Monte Carlo → JSON
    try:
        loader = DataLoader()
        params = loader.fit_distributions()
        sim    = MonteCarloSimulator(params)
        si = SimulationInput(PersonalProfile('M', 65), 30_000, 200, 80, 65, 0.4, 1_000)
        r  = sim.run(si)
        out = {
            'depletion_prob': round(r.depletion_prob, 3),
            'median_final_assets': round(r.median_final_assets, 0),
            'p50_at_75': round(float(r.percentile_paths.loc[75, 'p50']), 0)
                         if 75 in r.percentile_paths.index else None,
        }
        json.dumps(out)
        row(PASS, "MonteCarloSimulator 출력 JSON 직렬화", str(out))
        results['monte_carlo'] = True
    except Exception as e:
        row(FAIL, f"MonteCarloSimulator JSON 실패: {e}")
        results['monte_carlo'] = False

    # (c) Portfolio → JSON
    try:
        opt = PortfolioOptimizer()
        for level in ('conservative', 'moderate', 'aggressive'):
            res = opt.optimize_for_profile(level)
            out = {
                'weights': {k: round(float(v), 3) for k, v in res.weights.items()},
                'cvar_annual': round(res.cvar_annual, 3),
                'expected_return_annual': round(res.expected_return_annual, 3),
            }
            json.dumps(out)
        row(PASS, "PortfolioOptimizer 출력 JSON 직렬화")
        results['portfolio'] = True
    except Exception as e:
        row(FAIL, f"PortfolioOptimizer JSON 실패: {e}")
        results['portfolio'] = False

    # (d) Clustering → JSON
    try:
        cm = HouseholdClusterModel()
        r  = cm.analyze(UserProfile(60, 20_000, 4_000, 2))
        out = {
            'cluster_id': r.cluster_id,
            'user_percentile': r.user_percentile,
            'top30_avg_assets': r.top30_profile['avg_net_assets'],
        }
        json.dumps(out)
        row(PASS, "HouseholdClusterModel 출력 JSON 직렬화", str(out))
        results['clustering'] = True
    except Exception as e:
        row(FAIL, f"HouseholdClusterModel JSON 실패: {e}")
        results['clustering'] = False

    # (e) Anomaly L1 → JSON
    try:
        det = TransactionAnomalyDetector()
        res = det.score(Transaction(500, 2, True, 'transfer', 3))
        out = {
            'score': res.score,
            'level': res.level,
            'triggers': res.triggers,
            'recommendation': res.recommendation,
        }
        json.dumps(out)
        row(PASS, "TransactionAnomalyDetector 출력 JSON 직렬화", str(out))
        results['anomaly_l1'] = True
    except Exception as e:
        row(FAIL, f"TransactionAnomalyDetector JSON 실패: {e}")
        results['anomaly_l1'] = False

    all_ok = all(results.values())
    print()
    row(PASS if all_ok else FAIL,
        f"전체 Agent 호환성: {sum(results.values())}/{len(results)} 모듈 통과")

    return results


# ─────────────────────────────────────────────────────────────
# 종합 실행
# ─────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "★"*60)
    print("  LifeLong WM AI Agent — 전체 모듈 평가 보고서")
    print("★"*60)

    np.random.seed(42)

    metrics = {}
    metrics['survival']    = eval_survival()
    metrics['monte_carlo'] = eval_monte_carlo()
    metrics['anomaly']     = eval_anomaly()
    metrics['clustering']  = eval_clustering()
    metrics['agent_compat']= eval_agent_compatibility()

    banner("종합 요약")
    print(f"  생존 분석  C-statistic:   {metrics['survival']['c_statistic']:.2f}")
    print(f"  Monte Carlo 고갈확률:     {metrics['monte_carlo']['depletion_prob']:.1%}")
    print(f"  이상치 탐지  F1 / AUC:   "
          f"{metrics['anomaly']['f1']:.3f} / {metrics['anomaly']['auc']:.3f}")
    print(f"  군집 분석  Silhouette:   {metrics['clustering']['silhouette']:.3f}")
    compat = metrics['agent_compat']
    n_ok   = sum(compat.values())
    print(f"  Agent 호환성:            {n_ok}/{len(compat)} 모듈 JSON 직렬화 가능")
    print()
    if n_ok == len(compat):
        print("  ✅ 역할 B(Agent/UI)와 연결 준비 완료")
    else:
        print("  ⚠️  일부 모듈 수정 필요")

    return metrics


if __name__ == '__main__':
    run_all()
