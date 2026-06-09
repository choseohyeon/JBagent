"""
Layer 3: 생활 패턴 이상 탐지 (독거 고령층 안부 확인)
복수 신호 점수 합산 — 합성 데이터 기반
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DailySignal:
    """하루치 생활 신호"""
    date: str                        # 'YYYY-MM-DD'
    card_transactions: int           # 카드 결제 건수
    conversation_count: int          # AI Agent 대화 횟수
    utility_paid: bool               # 정기 요금 (관리비/통신비) 납부 여부
    pharmacy_visit: bool             # 약국/병원 방문 여부


@dataclass
class WellnessScore:
    date: str
    score: float                     # 0~100 (낮을수록 이상)
    alert_level: str                 # 'normal' | 'watch' | 'alert' | 'urgent'
    inactive_days: int               # 연속 무활동 일수
    signals: Dict[str, float]        # 신호별 기여 점수
    action: str                      # 권고 조치


class LifePatternDetector:
    """
    독거 고령층 생활 패턴 모니터링.
    카드 결제 / 대화 빈도 / 정기 지출 / 약국 방문을 종합해 안부를 추론한다.
    개인 기저선을 자동 학습하며, 활동이 급감하면 알림을 발송한다.
    """

    ALERT_THRESHOLDS = {'normal': 60, 'watch': 40, 'alert': 20}

    # 신호별 가중치
    SIGNAL_WEIGHTS = {
        'card':         0.35,
        'conversation': 0.30,
        'utility':      0.20,
        'pharmacy':     0.15,
    }

    def __init__(self, baseline_days: int = 30):
        self.baseline_days = baseline_days
        self.history: List[DailySignal] = []
        self._inactive_streak: int = 0

    def fit(self, signals: List[DailySignal]):
        """기저선 학습"""
        self.history = list(signals[-self.baseline_days:])

    def _baseline(self) -> Dict[str, float]:
        if not self.history:
            return {'card': 3.0, 'conversation': 2.0}
        return {
            'card':         max(np.mean([s.card_transactions for s in self.history]), 0.1),
            'conversation': max(np.mean([s.conversation_count for s in self.history]), 0.1),
        }

    def score_day(self, signal: DailySignal) -> WellnessScore:
        base = self._baseline()

        # 1. 카드 결제 점수 (0~100)
        card_ratio  = signal.card_transactions / base['card']
        card_score  = min(card_ratio * 100, 100)

        # 2. 대화 빈도 점수
        conv_ratio  = signal.conversation_count / base['conversation']
        conv_score  = min(conv_ratio * 100, 100)

        # 3. 정기 요금 납부 (이진)
        util_score  = 100.0 if signal.utility_paid else 0.0

        # 4. 약국/병원 방문 (이진 — 있으면 활동 확인)
        pharma_score = 100.0 if signal.pharmacy_visit else 50.0

        w = self.SIGNAL_WEIGHTS
        total = (w['card']         * card_score  +
                 w['conversation'] * conv_score  +
                 w['utility']      * util_score  +
                 w['pharmacy']     * pharma_score)
        total = max(0.0, min(total, 100.0))

        # 연속 무활동일 추적
        is_inactive = (signal.card_transactions == 0 and
                       signal.conversation_count == 0)
        if is_inactive:
            self._inactive_streak += 1
        else:
            self._inactive_streak = 0

        # 연속 무활동이면 추가 패널티
        streak_penalty = min(self._inactive_streak * 10, 40)
        total = max(0.0, total - streak_penalty)

        # 알림 수준
        if total >= self.ALERT_THRESHOLDS['normal']:
            level  = 'normal'
            action = '정상 — 모니터링 지속'
        elif total >= self.ALERT_THRESHOLDS['watch']:
            level  = 'watch'
            action = 'Agent가 안부 인사 메시지 발송'
        elif total >= self.ALERT_THRESHOLDS['alert']:
            level  = 'alert'
            action = f'신뢰 연락처 문자 알림 (연속 무활동 {self._inactive_streak}일)'
        else:
            level  = 'urgent'
            action = f'긴급 알림 — 복지사/119 연락 필요 (연속 무활동 {self._inactive_streak}일)'

        self.history.append(signal)

        return WellnessScore(
            date           = signal.date,
            score          = round(total, 1),
            alert_level    = level,
            inactive_days  = self._inactive_streak,
            signals        = {
                'card':         round(card_score, 1),
                'conversation': round(conv_score, 1),
                'utility':      round(util_score, 1),
                'pharmacy':     round(pharma_score, 1),
            },
            action = action,
        )


# ── 합성 데이터 생성기 ────────────────────────────────────────────────────────

def generate_synthetic_signals(days: int = 60,
                                seed: int = 42) -> List[DailySignal]:
    """
    독거 고령층 30일 정상 패턴 + 30일 이상 패턴 합성 데이터.
    실제 로그 데이터 없으므로 통계적 분포 기반 생성.
    """
    rng = np.random.default_rng(seed)
    signals = []
    for d in range(days):
        date = f"2026-04-{d+1:02d}" if d < 30 else f"2026-05-{d-29:02d}"
        if d < 30:   # 정상 구간
            card  = int(rng.poisson(3))
            conv  = int(rng.poisson(2))
            util  = rng.random() > 0.05
            pharm = rng.random() > 0.85
        else:        # 이상 구간 (활동 감소)
            decay = (d - 30) / 30
            card  = int(rng.poisson(max(3 * (1 - decay), 0.1)))
            conv  = int(rng.poisson(max(2 * (1 - decay), 0.05)))
            util  = rng.random() > (0.05 + decay * 0.8)
            pharm = rng.random() > 0.9
        signals.append(DailySignal(date, card, conv, util, pharm))
    return signals


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    signals = generate_synthetic_signals(60)

    detector = LifePatternDetector()
    detector.fit(signals[:30])   # 첫 30일로 기저선 학습

    print(f"{'날짜':>12}  {'웰니스':>6}  {'등급':>7}  "
          f"{'카드':>5}  {'대화':>5}  {'무활동':>5}  조치")
    print('-' * 75)

    prev_level = 'normal'
    for sig in signals[30:]:     # 이후 30일 모니터링
        result = detector.score_day(sig)
        marker = ' ◀ 변화' if result.alert_level != prev_level else ''
        print(f"{result.date:>12}  {result.score:>6.1f}  "
              f"{result.alert_level:>7}  "
              f"{result.signals['card']:>5.0f}  "
              f"{result.signals['conversation']:>5.0f}  "
              f"{result.inactive_days:>5}일  "
              f"{result.action[:30]}{marker}")
        prev_level = result.alert_level
