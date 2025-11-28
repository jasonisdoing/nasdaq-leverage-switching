"""추천(신호) 생성 로직."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from logic.common.signals import compute_signals, pick_target
from logic.common.data import compute_bounds, download_prices
from utils.report import render_table_eaw


def run_recommend(settings: Dict) -> Dict[str, object]:
    start_bound, warmup_start, end_bound = compute_bounds(settings)

    prices_full = download_prices(settings, warmup_start)
    signal_df_full = compute_signals(prices_full[settings["signal_symbol"]], settings)
    valid_index = prices_full.index[prices_full.index >= start_bound]
    prices = prices_full.loc[valid_index]
    signal_df = signal_df_full.loc[valid_index]
    if signal_df.empty:
        raise ValueError("시그널 계산에 필요한 데이터가 없습니다.")
    last_date = signal_df.index.max()
    target = pick_target(signal_df.loc[last_date], settings)

    # 상태 계산: 타깃을 BUY, 나머지 WAIT
    statuses = {}
    for sym in settings["trade_symbols"]:
        if sym == target:
            statuses[sym] = "BUY"
        else:
            statuses[sym] = "WAIT"

    # 일간 수익률은 전일 대비 종가 기준
    daily_rets = prices[settings["trade_symbols"]].pct_change()
    last_ret = daily_rets.loc[last_date] if last_date in daily_rets.index else pd.Series(dtype=float)

    # 테이블 구성
    headers = [
        "#",
        "티커",
        "상태",
        "보유일",
        "일간(%)",
        "현재가",
        "문구",
    ]
    aligns = ["center", "center", "center", "right", "right", "right", "left"]
    rows: List[List[str]] = []
    for idx, sym in enumerate(settings["trade_symbols"], start=1):
        price = prices.at[last_date, sym]
        ret = last_ret.get(sym, 0.0) if not last_ret.empty else 0.0
        rows.append(
            [
                str(idx),
                sym,
                statuses[sym],
                "-",
                f"{ret*100:+.2f}%",
                f"{price:,.2f}",
                "타깃" if sym == target else "",
            ]
        )

    table_lines = render_table_eaw(headers, rows, aligns)

    # 상태 요약
    status_counts = {}
    for st in statuses.values():
        status_counts[st] = status_counts.get(st, 0) + 1

    status_lines = ["=== 상태 요약 ==="]
    for st, cnt in status_counts.items():
        status_lines.append(f"  {st}: {cnt}개")

    return {
        "as_of": last_date.date().isoformat(),
        "target": target,
        "status_lines": status_lines,
        "table_lines": table_lines,
    }


def write_recommend_log(report: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"추천 로그 생성: {datetime.now().isoformat()}\n")
        f.write(f"기준일: {report['as_of']}\n\n")
        f.write("\n".join(report["status_lines"]))
        f.write("\n\n=== 추천 목록 ===\n\n")
        for line in report["table_lines"]:
            f.write(line + "\n")
