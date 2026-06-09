"""
Layer 1: 거래 이상 탐지 (금융사기 / 보이스피싱 방어)
Z-score + IQR + Isolation Forest + CUSUM 앙상블
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.ensemble import IsolationForest


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class Transaction:
    amount: float                  # 금액 (만원)
    hour: int                      # 거래 시각 (0~23)
    is_new_account: bool           # 처음 거래 계좌 여부
    tx_type: str                   # 'transfer' | 'payment' | 'withdrawal'
    consecutive_count: int = 1     # 단기 연속 이체 횟수


@dataclass
class AnomalyScore:
    score: float                   # 0~100 (높을수록 이상)
    level: str                     # 'low' | 'medium' | 'high'
    triggers: List[str]            # 이상 감지 근거
    recommendation: str            # 권고 조치


# ── 탐지기 ───────────────────────────────────────────────────────────────────

class TransactionAnomalyDetector:
    """
    개인 거래 기저선 자동 학습 후 신규 거래의 이상도를 0~100으로 평가.
    사전 레이블 데이터 불필요 (비지도 학습).
    """

    THRESHOLDS = {'low': 30, 'medium': 60, 'high': 80}

    # 가중치: 각 탐지 방법의 최종 점수 기여도
    # history 충분 시 (30+ 거래): 통계적 탐지 위주
    # history 부족 시: context 가중치를 동적으로 상향 조정 (score() 참조)
    WEIGHTS = {
        'zscore':          0.25,
        'iqr':             0.20,
        'isolation':       0.25,
        'cusum':           0.15,
        'context':         0.15,
    }

    def __init__(self, min_history: int = 30):
        self.min_history   = min_history   # 기저선 학습 최소 거래 수
        self.history: List[Transaction] = []
        self._iforest: Optional[IsolationForest] = None
        self._cusum_state: float = 0.0     # CUSUM 누적값

    # ── 기저선 학습 ───────────────────────────────────────────────────────────

    def fit(self, transactions: List[Transaction]):
        self.history = list(transactions)
        if len(self.history) >= self.min_history:
            X = self._to_features(self.history)
            self._iforest = IsolationForest(
                contamination=0.05, random_state=42, n_estimators=100
            )
            self._iforest.fit(X)
        self._cusum_state = 0.0

    def _to_features(self, txs: List[Transaction]) -> np.ndarray:
        return np.array([
            [t.amount, t.hour, int(t.is_new_account),
             self._type_code(t.tx_type), t.consecutive_count]
            for t in txs
        ])

    @staticmethod
    def _type_code(tx_type: str) -> int:
        return {'payment': 0, 'withdrawal': 1, 'transfer': 2}.get(tx_type, 0)

    # ── 점수 계산 ─────────────────────────────────────────────────────────────

    def _zscore_score(self, tx: Transaction) -> tuple[float, list]:
        if len(self.history) < 5:
            return 0.0, []
        amounts = [t.amount for t in self.history]
        mu, sigma = np.mean(amounts), np.std(amounts)
        if sigma == 0:
            return 0.0, []
        z = abs(tx.amount - mu) / sigma
        score = min(z / 4.0, 1.0) * 100
        triggers = [f"금액 Z-score {z:.1f}σ"] if z > 2.5 else []
        return score, triggers

    def _iqr_score(self, tx: Transaction) -> tuple[float, list]:
        if len(self.history) < 5:
            return 0.0, []
        amounts = [t.amount for t in self.history]
        q1, q3 = np.percentile(amounts, 25), np.percentile(amounts, 75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0, []
        fence = q3 + 3.0 * iqr
        if tx.amount > fence:
            score = min((tx.amount - fence) / fence, 1.0) * 100
            return score, [f"IQR 상한({fence:,.0f}만) {(tx.amount/fence):.1f}배 초과"]
        return 0.0, []

    def _isolation_score(self, tx: Transaction) -> tuple[float, list]:
        if self._iforest is None:
            return 0.0, []
        x = self._to_features([tx])
        raw = self._iforest.score_samples(x)[0]      # 음수: 이상할수록 낮음
        # score_samples 범위 대략 [-0.5, 0.5] → [0, 100]
        score = max(0.0, min((-raw - 0.0) / 0.5, 1.0)) * 100
        triggers = ["Isolation Forest 이상 패턴"] if score > 60 else []
        return score, triggers

    def _cusum_score(self, tx: Transaction) -> tuple[float, list]:
        if len(self.history) < 5:
            return 0.0, []
        mu = np.mean([t.amount for t in self.history])
        k  = mu * 0.5   # 허용 슬랙
        self._cusum_state = max(0, self._cusum_state + tx.amount - mu - k)
        threshold = mu * 5
        if self._cusum_state > threshold:
            score = min(self._cusum_state / threshold, 2.0) * 50
            return score, [f"CUSUM 누적 이상 ({self._cusum_state:,.0f})"]
        return 0.0, []

    def _context_score(self, tx: Transaction) -> tuple[float, list]:
        score, triggers = 0.0, []

        # 새 계좌 이체
        if tx.is_new_account and tx.tx_type == 'transfer':
            score += 40
            triggers.append("처음 거래 계좌 이체")

        # 심야/새벽 거래 (23시~5시)
        if tx.hour >= 23 or tx.hour <= 5:
            score += 20
            triggers.append(f"심야 거래 ({tx.hour}시)")

        # 단기 연속 이체 (3회 이상)
        if tx.consecutive_count >= 3:
            score += 30
            triggers.append(f"단기 연속 이체 {tx.consecutive_count}회")

        # 보이스피싱 패턴: 새 계좌 + 큰 금액 + 연속
        if tx.is_new_account and tx.consecutive_count >= 2 and tx.amount > 100:
            score += 30
            triggers.append("보이스피싱 복합 패턴")

        return min(score, 100.0), triggers

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def score(self, tx: Transaction) -> AnomalyScore:
        """신규 거래의 이상도 점수를 계산한다."""
        s_z,   t_z   = self._zscore_score(tx)
        s_iqr, t_iqr = self._iqr_score(tx)
        s_iso, t_iso = self._isolation_score(tx)
        s_cus, t_cus = self._cusum_score(tx)
        s_ctx, t_ctx = self._context_score(tx)

        has_history = len(self.history) >= self.min_history
        if has_history:
            w = self.WEIGHTS
            final = (w['zscore']    * s_z   +
                     w['iqr']       * s_iqr +
                     w['isolation'] * s_iso +
                     w['cusum']     * s_cus +
                     w['context']   * s_ctx)
        else:
            # history 부족: 통계 탐지 불가 → context 단독으로 판단
            # context_score는 보이스피싱 구조 신호 기반이므로 단독으로도 유효
            final = s_ctx
        final = min(final, 100.0)

        all_triggers = t_z + t_iqr + t_iso + t_cus + t_ctx

        if final >= self.THRESHOLDS['high']:
            level = 'high'
            rec   = "거래 즉시 보류 후 본인 확인 필요"
        elif final >= self.THRESHOLDS['medium']:
            level = 'medium'
            rec   = "거래 진행 전 확인 요청"
        else:
            level = 'low'
            rec   = "정상 기록"

        # 정상 거래만 기저선에 추가 (이상 거래가 기저선을 오염시키는 것 방지)
        # medium/high 거래를 history에 포함하면 Z-score 민감도가 급락함
        if level == 'low':
            self.history.append(tx)

        return AnomalyScore(
            score          = round(final, 1),
            level          = level,
            triggers       = all_triggers,
            recommendation = rec,
        )


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import random
    rng = random.Random(42)

    # 정상 거래 이력 생성
    normal_txs = [
        Transaction(amount=rng.uniform(5, 50),
                    hour=rng.randint(9, 21),
                    is_new_account=False,
                    tx_type='payment')
        for _ in range(60)
    ]

    detector = TransactionAnomalyDetector()
    detector.fit(normal_txs)

    test_cases = [
        ("정상 소액 결제",
         Transaction(20, 14, False, 'payment')),
        ("고액 이체",
         Transaction(500, 15, False, 'transfer')),
        ("새 계좌 심야 이체",
         Transaction(200, 2, True, 'transfer')),
        ("보이스피싱 패턴",
         Transaction(300, 1, True, 'transfer', consecutive_count=4)),
    ]

    print(f"{'케이스':<20} {'점수':>5}  {'등급':>7}  트리거")
    print('-' * 70)
    for label, tx in test_cases:
        result = detector.score(tx)
        triggers = ' | '.join(result.triggers) if result.triggers else '없음'
        print(f"{label:<20} {result.score:>5.1f}  {result.level:>7}  {triggers}")
