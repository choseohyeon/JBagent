"""
Claude API Tool Use 시그니처 정의 — 역할 B 참고용
역할 A가 확정한 함수 인터페이스를 Claude Tool 형식으로 래핑.

사용법 (역할 B):
    from agent.tools import TOOLS, call_tool
    response = client.messages.create(..., tools=TOOLS)
    result = call_tool(tool_name, tool_input)
"""
import sys
sys.path.insert(0, '.')

from models.survival    import SurvivalModel, PersonalProfile
from models.monte_carlo import DataLoader, MonteCarloSimulator, SimulationInput
from models.pension     import PensionOptimizer, PensionProfile
from models.portfolio   import PortfolioOptimizer
from models.clustering  import HouseholdClusterModel, UserProfile
from models.anomaly.layer1 import TransactionAnomalyDetector, Transaction

# ── 싱글턴 (모듈 로드 시 1회 초기화) ────────────────────────────────────────
_survival  = SurvivalModel()
_loader    = DataLoader()
_mc_params = _loader.fit_distributions()
_mc_sim    = MonteCarloSimulator(_mc_params)
_pension   = PensionOptimizer()
_portfolio = PortfolioOptimizer()
_cluster   = HouseholdClusterModel()
_anomaly   = TransactionAnomalyDetector()


# ── 개별 Tool 함수 ────────────────────────────────────────────────────────────

def run_survival(sex: str, current_age: int,
                 smoking: bool = False,
                 chronic_disease: bool = False,
                 bmi: float = 22.0) -> dict:
    """개인화 생존 분석 실행"""
    p   = PersonalProfile(sex, current_age, smoking, chronic_disease, bmi)
    sf  = _survival.get_survival_function(p)
    exp = _survival.expected_remaining_life(p)
    return {
        'expected_remaining_life': round(exp, 1),
        'death_prob_before_80':    round(_survival.death_prob_before(p, 80), 3),
        'death_prob_before_90':    round(_survival.death_prob_before(p, 90), 3),
        'survival_at_70':          round(float(sf.get(70, 0)), 3),
        'survival_at_80':          round(float(sf.get(80, 0)), 3),
    }


def run_monte_carlo(sex: str, current_age: int,
                    initial_assets: float, monthly_expense: float,
                    pension_monthly: float, pension_start_age: int,
                    stock_ratio: float = 0.4,
                    smoking: bool = False,
                    chronic_disease: bool = False,
                    n_simulations: int = 5_000) -> dict:
    """생애 현금흐름 Monte Carlo 시뮬레이션"""
    profile = PersonalProfile(sex, current_age, smoking, chronic_disease)
    si = SimulationInput(
        profile=profile,
        initial_assets=initial_assets,
        monthly_expense=monthly_expense,
        pension_monthly=pension_monthly,
        pension_start_age=pension_start_age,
        stock_ratio=stock_ratio,
        n_simulations=n_simulations,
    )
    result = _mc_sim.run(si)
    pp = result.percentile_paths
    ages = [70, 75, 80, 85, 90]
    paths = {}
    for age in ages:
        if age in pp.index:
            paths[str(age)] = {
                'p10': round(float(pp.loc[age,'p10']), 0),
                'p50': round(float(pp.loc[age,'p50']), 0),
                'p90': round(float(pp.loc[age,'p90']), 0),
            }
    return {
        'depletion_prob': round(result.depletion_prob, 3),
        'median_final_assets': round(result.median_final_assets, 0),
        'asset_paths': paths,
    }


def run_pension(current_age: int,
                expected_monthly_pension: float,
                life_expectancy: float,
                sex: str = 'M',
                smoking: bool = False) -> dict:
    """국민연금 수령 시기 최적화"""
    p   = PersonalProfile(sex, current_age, smoking)
    sf  = _survival.get_survival_function(p)
    exp = _survival.expected_remaining_life(p)
    prof = PensionProfile(current_age, expected_monthly_pension,
                          exp if life_expectancy <= 0 else life_expectancy)
    result = _pension.recommend(prof, survival_fn=sf)
    best   = result['best_by_npv']
    return {
        'optimal_claim_age': int(best['claim_age']),
        'monthly_at_optimal': round(float(best['monthly_amount']), 1),
        'adjustment_rate':    round(float(best['adjustment_rate']), 3),
        'npv_gain_vs_65':     round(float(result['npv_gain_vs_65']), 0),
        'breakeven_age':      round(float(best['breakeven_age']), 1),
    }


def run_portfolio(risk_level: str = 'moderate') -> dict:
    """CVaR 포트폴리오 최적화"""
    res = _portfolio.optimize_for_profile(risk_level)
    return {
        'weights':   {k: round(float(v), 3) for k, v in res.weights.items()},
        'stock_ratio_for_mc': round(float(res.weights.get('KOSPI', 0)), 3),
        'cvar_annual': round(res.cvar_annual, 3),
        'expected_return_annual': round(res.expected_return_annual, 3),
        'sharpe_ratio': round(res.sharpe_ratio, 3),
    }


def run_clustering(age: int, net_assets: float,
                   income: float, members: int) -> dict:
    """또래 집단 대비 재무 상태 분석"""
    result = _cluster.analyze(UserProfile(age, net_assets, income, members))
    return {
        'cluster_id': result.cluster_id,
        'cluster_size': result.cluster_size,
        'net_assets_percentile': result.user_percentile['net_assets'],
        'income_percentile':     result.user_percentile['income'],
        'top30_avg_assets':      result.top30_profile['avg_net_assets'],
        'top30_avg_income':      result.top30_profile['avg_income'],
        'silhouette':            result.silhouette,
    }


def run_anomaly_score(amount: float, hour: int,
                      is_new_account: bool, tx_type: str,
                      consecutive_count: int = 1) -> dict:
    """거래 이상도 점수 계산"""
    tx  = Transaction(amount, hour, is_new_account, tx_type, consecutive_count)
    res = _anomaly.score(tx)
    return {
        'score':          res.score,
        'level':          res.level,
        'triggers':       res.triggers,
        'recommendation': res.recommendation,
    }


# ── Claude Tool 스펙 (역할 B가 client.messages.create에 전달) ─────────────────

TOOLS = [
    {
        "name": "run_survival",
        "description": "성별·나이·건강 상태 기반 개인화 생존 분석. 기대여명과 특정 연령 이전 사망 확률을 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sex":             {"type": "string",  "enum": ["M","F"]},
                "current_age":     {"type": "integer"},
                "smoking":         {"type": "boolean", "default": False},
                "chronic_disease": {"type": "boolean", "default": False},
                "bmi":             {"type": "number",  "default": 22.0},
            },
            "required": ["sex", "current_age"],
        },
    },
    {
        "name": "run_monte_carlo",
        "description": "초기 자산·월 지출·연금을 입력받아 10,000회 생애 시뮬레이션을 실행. 자산 고갈 확률과 연령별 자산 퍼센타일 경로를 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sex":               {"type": "string",  "enum": ["M","F"]},
                "current_age":       {"type": "integer"},
                "initial_assets":    {"type": "number",  "description": "초기 자산 (만원)"},
                "monthly_expense":   {"type": "number",  "description": "월 지출 (만원)"},
                "pension_monthly":   {"type": "number",  "description": "월 연금 (만원)"},
                "pension_start_age": {"type": "integer"},
                "stock_ratio":       {"type": "number",  "default": 0.4},
                "smoking":           {"type": "boolean", "default": False},
                "chronic_disease":   {"type": "boolean", "default": False},
            },
            "required": ["sex","current_age","initial_assets",
                         "monthly_expense","pension_monthly","pension_start_age"],
        },
    },
    {
        "name": "run_pension",
        "description": "국민연금 수령 시기(60~70세) 시나리오를 비교해 NPV 최적 수령 시기와 월 수령액을 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_age":               {"type": "integer"},
                "expected_monthly_pension":  {"type": "number", "description": "65세 기준 예상 월 수령액 (만원)"},
                "life_expectancy":           {"type": "number", "description": "기대 잔여 수명 (년), 0이면 자동 계산"},
                "sex":                       {"type": "string", "enum": ["M","F"], "default": "M"},
            },
            "required": ["current_age","expected_monthly_pension","life_expectancy"],
        },
    },
    {
        "name": "run_portfolio",
        "description": "리스크 허용도에 맞는 CVaR 최적 자산 배분(KOSPI/KTB 비중)을 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_level": {
                    "type": "string",
                    "enum": ["conservative","moderate","aggressive"],
                    "description": "보수적/중립/공격적",
                },
            },
            "required": ["risk_level"],
        },
    },
    {
        "name": "run_clustering",
        "description": "또래 가구 대비 순자산·소득 퍼센타일과 상위 30% 벤치마크를 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "age":        {"type": "integer"},
                "net_assets": {"type": "number", "description": "순자산 (만원)"},
                "income":     {"type": "number", "description": "연간 경상소득 (만원)"},
                "members":    {"type": "integer", "description": "가구원수"},
            },
            "required": ["age","net_assets","income","members"],
        },
    },
    {
        "name": "run_anomaly_score",
        "description": "거래 이상도 점수(0~100)와 등급(low/medium/high)을 반환합니다. 보이스피싱·이상 이체 탐지.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount":            {"type": "number",  "description": "거래 금액 (만원)"},
                "hour":              {"type": "integer", "description": "거래 시각 (0~23)"},
                "is_new_account":    {"type": "boolean"},
                "tx_type":           {"type": "string",  "enum": ["payment","transfer","withdrawal"]},
                "consecutive_count": {"type": "integer", "default": 1},
            },
            "required": ["amount","hour","is_new_account","tx_type"],
        },
    },
]


def call_tool(name: str, inputs: dict) -> dict:
    """Agent에서 Tool 호출 시 사용하는 디스패처"""
    dispatch = {
        'run_survival':      run_survival,
        'run_monte_carlo':   run_monte_carlo,
        'run_pension':       run_pension,
        'run_portfolio':     run_portfolio,
        'run_clustering':    run_clustering,
        'run_anomaly_score': run_anomaly_score,
    }
    fn = dispatch.get(name)
    if fn is None:
        raise ValueError(f"Unknown tool: {name}")
    return fn(**inputs)
