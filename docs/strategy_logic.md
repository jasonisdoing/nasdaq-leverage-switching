# 전략 로직 (Strategy Logic)

## 1. 핵심 전략: 2-Tier Hysteresis Switching
이 프로젝트는 단순한 모멘텀 전략이 아닌, **히스테리시스(Hysteresis)** 개념을 도입한 스위칭 전략을 사용합니다. 이는 잦은 매매 신호 번복(Whipsaw)을 방지하고 추세의 안정성을 확보하기 위함입니다.

### 작동 원리
시그널 자산인 **QQQ(나스닥 100)**의 **고점 대비 하락률(Drawdown)**을 기준으로 공격 자산과 방어 자산을 교체합니다.

- **공격 자산 (Offense)**: `TQQQ` (나스닥 3배 레버리지)
- **방어 자산 (Defense)**: `GDX` (금광주), `GLDM` (금), `SPLV` (저변동성) 등 (튜닝을 통해 최적 자산 선정)

### 이중 임계값 (Dual Thresholds)
매수와 매도 기준을 다르게 설정하여 시장의 노이즈를 걸러냅니다.

1.  **매도 기준 (`drawdown_sell_cutoff`)**:
    - 공격 자산을 보유 중일 때, QQQ의 하락률이 이 기준보다 **더 낮아지면(악화되면)** 방어 자산으로 교체합니다.
    - 예: `sell_cutoff = 0.4%` → 하락률이 -0.4% 밑으로 내려가면 매도.

2.  **매수 기준 (`drawdown_buy_cutoff`)**:
    - 방어 자산을 보유 중일 때, QQQ의 하락률이 이 기준보다 **더 높아지면(회복되면)** 공격 자산으로 교체합니다.
    - 예: `buy_cutoff = 0.3%` → 하락률이 -0.3% 위로 올라오면 매수.

> **핵심**: 항상 `buy_cutoff < sell_cutoff` 조건을 유지하여, "확실히 떨어지면 팔고, 확실히 오르면 산다"는 원칙을 지킵니다. 그 사이 구간(Dead Zone)에서는 포지션을 변경하지 않고 유지(Hold)합니다.

## 2. 튜닝 알고리즘 (Tuning Algorithm)

### 전수 조사 (Exhaustive Search)
최적의 파라미터를 찾기 위해 가능한 모든 조합을 테스트합니다.

- **튜닝 대상 파라미터**:
    - `drawdown_buy_cutoff`: 0.1% ~ 3.0% (0.1% 단위)
    - `drawdown_sell_cutoff`: 0.1% ~ 3.0% (0.1% 단위)
    - `defense_ticker`: GDX, GLDM, SPLV 등 후보군

### 최적화 기준
- **CAGR (Compound Annual Growth Rate)**: 연평균 수익률이 가장 높은 조합을 최우선으로 선택합니다.
- **MDD (Maximum Drawdown)**: 참고 지표로 함께 출력됩니다.

### 데이터 기간
- 기본적으로 최근 **12개월** 데이터를 사용하여 시장의 최신 트렌드에 맞는 파라미터를 찾습니다.

## 3. 시그널 계산 상세

### Drawdown 계산
```python
Peak = cummax(Price)
Drawdown = (Price / Peak) - 1.0
```

### 포지션 결정 (`pick_target`)
```python
if prev_target == offense:
    if current_dd < sell_cut:
        return defense  # 매도
    return offense      # 유지
else:
    if current_dd > buy_cut:
        return offense  # 매수
    return defense      # 유지
```

## 4. 추천과 백테스트의 통합

> **중요**: `recommend.py`는 독립적인 추천 로직이 **없습니다**.

**설계 원칙**: 백테스트 결과의 마지막 날 = 오늘의 추천

이렇게 하면:
- ✅ 백테스트와 추천이 항상 **100% 일관된 결과**를 보장
- ✅ 로직 변경 시 **한 곳만 수정**하면 됨
- ✅ 유지보수 부담 감소
