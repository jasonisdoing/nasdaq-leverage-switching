"""튜닝 핵심 로직 (백테스트 실행·집계)."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from os import cpu_count

import numpy as np
import pandas as pd
import yfinance as yf

from logic.common.settings import load_settings
from logic.common.data import (
    compute_bounds,
    download_fx,
    download_opens,
    download_prices,
    _extract_field,
)
from logic.backtest.runner import run_backtest
from utils.report import render_table_eaw


def _is_rate_limit_error(exc: Exception) -> bool:
    s = repr(exc).lower()
    return "yfratelimiterror" in s or "rate limit" in s


def _is_network_or_data_error(exc: Exception) -> bool:
    s = repr(exc).lower()
    keywords = [
        "dnserror",
        "could not resolve host",
        "timed out",
        "operation timed out",
        "시가 데이터가 비어 있습니다",
        "가격 데이터를 받아오지 못했습니다",
    ]
    return any(k in s for k in keywords)


def _run_single(
    args: Tuple[
        Dict, Dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Timestamp
    ]
) -> Dict:
    base_settings, overrides, pre_prices, pre_opens, pre_fx, pre_bench, start_bound = (
        args
    )
    tuned = dict(base_settings)
    tuned.update(overrides)
    report = run_backtest(
        tuned,
        pre_prices=pre_prices,
        pre_opens=pre_opens,
        pre_fx=pre_fx,
        pre_bench=pre_bench,
        start_bound_override=start_bound,
    )
    return {
        "params": tuned,
        "cagr": report["cagr"],
        "mdd": report["max_drawdown"],
        "sharpe": report["sharpe"],
        "vol": report["vol"],
        "period_return": report.get("period_return", 0.0),
    }


def run_tuning(
    tuning_config: Dict[str, np.ndarray],
    *,
    max_workers: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    partial_cb: Callable[[List[Dict], int, int], None] | None = None,
) -> Tuple[List[Dict], Dict]:
    start_ts = datetime.now()
    settings = load_settings(Path("settings.json"))  # 필수 키 없으면 예외
    start_bound, warmup_start, end_bound = compute_bounds(settings)

    try:
        # 프리패치: 방어자산 후보까지 모두 포함해 한 번에 다운로드
        all_defs = list(tuning_config.get("defense_ticker", []))
        tickers = list({settings["trade_ticker"], settings["signal_ticker"], *all_defs})
        price_raw = yf.download(
            tickers, start=warmup_start, auto_adjust=True, progress=False
        )
        if price_raw is None or len(price_raw) == 0:
            raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
        pre_prices = _extract_field(price_raw, "Close", tickers)
        pre_opens = _extract_field(price_raw, "Open", tickers)

        pre_fx = download_fx(warmup_start)
        bench_raw_entries = settings["benchmarks"]
        bench_tickers = []
        for b in bench_raw_entries:
            if isinstance(b, dict):
                ticker = b.get("ticker")
            else:
                ticker = str(b)
            if ticker:
                bench_tickers.append(ticker)
        bench_raw = yf.download(
            bench_tickers, start=warmup_start, auto_adjust=True, progress=False
        )
        if bench_raw is None or len(bench_raw) == 0:
            raise ValueError(
                f"벤치마크 데이터를 받아오지 못했습니다: {settings['benchmarks']}"
            )
        pre_bench = _extract_field(bench_raw, "Close", bench_tickers)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise SystemExit(
                "yfinance YFRateLimitError: 잠시 후 다시 실행하세요."
            ) from exc
        raise RuntimeError(
            f"프리패치 단계에서 데이터 로드에 실패했습니다: {exc}"
        ) from exc

    combos: List[Dict] = []
    combos: List[Dict] = []
    combos: List[Dict] = []
    for buy_cut in tuning_config["drawdown_buy_cutoff"]:
        for sell_cut in tuning_config["drawdown_sell_cutoff"]:
            # 히스테리시스 조건: buy_cutoff < sell_cutoff (절대값 기준)
            # 예: buy=1.0, sell=2.0 -> OK
            # 예: buy=2.0, sell=1.0 -> Invalid (이미 팔았는데 사야 함? 논리적 모순)
            if buy_cut >= sell_cut:
                continue

            for def_t in tuning_config["defense_ticker"]:
                combos.append(
                    {
                        "drawdown_buy_cutoff": float(buy_cut),
                        "drawdown_sell_cutoff": float(sell_cut),
                        "defense_ticker": str(def_t),
                    }
                )

    total_cases = len(combos)
    workers = max_workers or cpu_count() or 1
    results: List[Dict] = []
    completed = 0
    next_progress = 1

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(
                _run_single,
                (
                    settings,
                    overrides,
                    pre_prices,
                    pre_opens,
                    pre_fx,
                    pre_bench,
                    start_bound,
                ),
            ): overrides
            for overrides in combos
        }
        for fut in as_completed(future_map):
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                overrides = future_map[fut]
                if _is_rate_limit_error(exc) or _is_network_or_data_error(exc):
                    print(f"[튜닝 중단] 네트워크/데이터 오류 감지: {exc}")
                    raise SystemExit(1) from exc
                print(f"[튜닝 경고] 조합 {overrides} 실패: {exc}")
            completed += 1
            progress = int(completed / total_cases * 100)
            if progress_cb and progress >= next_progress:
                progress_cb(completed, total_cases)
                next_progress = progress + 1
            if partial_cb and progress >= next_progress - 1:
                partial_cb(results, completed, total_cases)

    # 정렬: CAGR 내림차순
    results.sort(key=lambda x: x["cagr"], reverse=True)
    months = round((end_bound - start_bound).days / 30.0, 1)
    return results, {
        "start_ts": start_ts,
        "total": total_cases,
        "period_start": start_bound.date(),
        "period_end": end_bound.date(),
        "period_months": months,
        "months_range": settings["months_range"],
    }


def render_top_table(
    results: List[Dict], top_n: int = 100, months_range: int | None = None
) -> List[str]:
    if not months_range:
        raise ValueError("months_range must be provided")
    pr_label = f"{months_range}개월 수익률(%)"
    headers = [
        "defense_ticker",
        "buy_cutoff",
        "sell_cutoff",
        pr_label,
        "CAGR(%)",
        "MDD(%)",
        "Sharpe",
        "Vol(%)",
    ]
    aligns = ["right"] * len(headers)
    rows: List[List[str]] = []
    for row in results[:top_n]:
        p = row["params"]
        rows.append(
            [
                str(p.get("defense_ticker", "")),
                f"{p['drawdown_buy_cutoff']:.2f}",
                f"{p['drawdown_sell_cutoff']:.2f}",
                f"{row.get('period_return', 0.0)*100:.2f}",
                f"{row['cagr']*100:.2f}",
                f"{row['mdd']*100:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['vol']*100:.2f}",
            ]
        )
    return render_table_eaw(headers, rows, aligns)
