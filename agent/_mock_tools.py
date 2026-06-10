"""
Role A 통계 모델이 완성되기 전까지 사용하는 Mock 함수들.
Role A가 각 models/*.py를 완성하면 이 파일의 dispatch를 실제 함수로 교체한다.

교체 방법:
    from models.monte_carlo import run_monte_carlo
    MOCK_DISPATCH["run_monte_carlo"] = lambda inp: run_monte_carlo(**inp)
"""


def mock_monte_carlo(inp: dict) -> dict:
    assets = inp.get("assets", 300_000_000)
    monthly_expense = inp.get("monthly_expense", 2_000_000)
    pension = inp.get("pension", 800_000)

    net_monthly = pension - monthly_expense
    depletion_prob = max(0.05, min(0.95, round(0.50 - net_monthly / 10_000_000, 3)))

    return {
        "depletion_probability": depletion_prob,
        "percentile_10_safe_years": 12,
        "percentile_50_safe_years": 22,
        "percentile_90_safe_years": 35,
        "safe_withdrawal_monthly": round(assets * 0.003),
        "simulation_count": 10000,
        "note": "[MOCK] Role A monte_carlo 연결 전 임시 값",
    }


def mock_estimate_survival(inp: dict) -> dict:
    age = inp.get("age", 65)
    sex = inp.get("sex", "M")
    smoking = inp.get("smoking", False)

    base = 85 if sex == "F" else 80
    if smoking:
        base -= 5
    remaining = max(0, base - age)

    return {
        "survival_prob_5yr": round(max(0, 0.95 - age * 0.003), 3),
        "survival_prob_10yr": round(max(0, 0.85 - age * 0.005), 3),
        "survival_prob_20yr": round(max(0, 0.50 - age * 0.007), 3),
        "expected_remaining_years": remaining,
        "p10_death_age": base - 10,
        "p50_death_age": base,
        "p90_death_age": base + 8,
        "note": "[MOCK] Role A survival 연결 전 임시 값",
    }


def mock_optimize_pension(inp: dict) -> dict:
    current_age = inp.get("current_age", 60)
    base_monthly = inp.get("expected_monthly_pension", 1_000_000)

    scenarios = []
    for claim_age in [60, 62, 65, 68, 70]:
        if claim_age < current_age:
            continue
        # 조기수령 -7.2%/년, 연기 +7.2%/년
        adj = 1.0 + (claim_age - 65) * 0.072
        adj_monthly = round(base_monthly * adj)
        total_to_85 = round(adj_monthly * 12 * (85 - claim_age))
        breakeven = 65 + max(0, (claim_age - 65) * 3)
        scenarios.append({
            "claim_age": claim_age,
            "adjusted_monthly_pension": adj_monthly,
            "lifetime_total_to_85": total_to_85,
            "breakeven_age": breakeven,
        })

    return {
        "scenarios": scenarios,
        "recommendation": "65세 수령이 기대수명 기준 최적입니다.",
        "note": "[MOCK] Role A pension 연결 전 임시 값",
    }


def mock_optimize_portfolio(inp: dict) -> dict:
    risk = inp.get("risk_tolerance", "medium")

    alloc = {
        "low":    {"국내채권": 0.60, "해외채권": 0.20, "국내주식": 0.10, "해외주식": 0.05, "현금": 0.05},
        "medium": {"국내채권": 0.40, "해외채권": 0.15, "국내주식": 0.25, "해외주식": 0.15, "현금": 0.05},
        "high":   {"국내채권": 0.20, "해외채권": 0.10, "국내주식": 0.40, "해외주식": 0.25, "현금": 0.05},
    }
    ret = {"low": 0.035, "medium": 0.055, "high": 0.075}
    cvar = {"low": -0.05, "medium": -0.10, "high": -0.18}

    return {
        "optimal_allocation": alloc.get(risk, alloc["medium"]),
        "expected_annual_return": ret.get(risk, 0.055),
        "cvar_95": cvar.get(risk, -0.10),
        "note": "[MOCK] Role A portfolio 연결 전 임시 값",
    }


def mock_get_percentile_rank(inp: dict) -> dict:
    age = inp.get("age", 65)
    net_assets = inp.get("net_assets", 300_000_000)

    # 가계금융복지조사 기반 연령대별 중위 순자산 (단순화)
    medians = {60: 280_000_000, 65: 260_000_000, 70: 230_000_000, 75: 200_000_000}
    age_key = min(medians, key=lambda x: abs(x - age))
    median = medians[age_key]

    percentile = min(99, max(1, int(50 + (net_assets - median) / median * 30)))

    return {
        "percentile_rank": percentile,
        "peer_median_assets": median,
        "top30_threshold": round(median * 1.5),
        "cluster_label": "중산층 은퇴 준비층",
        "peer_count": 1240,
        "note": "[MOCK] Role A clustering 연결 전 임시 값",
    }


def mock_detect_transaction_anomaly(inp: dict) -> dict:
    amount = inp.get("amount", 100_000)
    is_new_account = inp.get("is_new_account", False)
    hour = inp.get("hour", 14)

    score = 0.0
    flags = []

    if amount > 5_000_000:
        score += 0.4
        flags.append("고액 거래")
    if is_new_account:
        score += 0.4
        flags.append("처음 거래하는 계좌")
    if hour < 6 or hour > 22:
        score += 0.2
        flags.append("비정상 시간대")

    level = "낮음" if score < 0.3 else "중간" if score < 0.6 else "높음"
    rec = (
        "거래를 보류하고 본인 확인을 권장합니다." if level == "높음"
        else "주의가 필요합니다. 본인 거래가 맞는지 확인해 주세요." if level == "중간"
        else "정상 처리 가능합니다."
    )

    return {
        "anomaly_score": round(score, 2),
        "risk_level": level,
        "flags": flags,
        "recommendation": rec,
        "note": "[MOCK] Role A anomaly 연결 전 임시 값",
    }


MOCK_DISPATCH = {
    "run_monte_carlo": mock_monte_carlo,
    "estimate_survival": mock_estimate_survival,
    "optimize_pension": mock_optimize_pension,
    "optimize_portfolio": mock_optimize_portfolio,
    "get_percentile_rank": mock_get_percentile_rank,
    "detect_transaction_anomaly": mock_detect_transaction_anomaly,
}
