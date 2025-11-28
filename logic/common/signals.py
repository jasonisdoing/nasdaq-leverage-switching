"""공통 시그널 계산 및 포지션 선택 로직."""

from typing import Dict

import numpy as np
import pandas as pd


def compute_signals(prices: pd.Series, settings: Dict) -> pd.DataFrame:
    """가격 시계열로 추세/변동성/드로다운 신호를 계산합니다."""
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    df["ma_short"] = prices.rolling(settings["ma_short"]).mean()
    df["ma_long"] = prices.rolling(settings["ma_long"]).mean()

    # 변동성 필터 제거: 변동성은 0으로 두고 사용하지 않음
    df["vol"] = 0.0

    peak = prices.cummax()
    df["drawdown"] = prices / peak - 1.0
    return df.dropna()


def pick_target(row, settings: Dict) -> str:
    """신호 행을 받아 매수 대상 티커를 결정합니다."""
    dd_cutoff_raw = settings["drawdown_cutoff"]
    dd_cutoff = dd_cutoff_raw / 100 if dd_cutoff_raw > 1 else dd_cutoff_raw
    offense = settings["trade_ticker"]
    defense = settings["defense_ticker"]

    # 방어/공격 두 자산 전환 (방어가 CASH거나 ETF여도 동일 로직)
    if row["drawdown"] <= -dd_cutoff:
        return defense
    if row["ma_short"] > row["ma_long"]:
        return offense
    return defense
