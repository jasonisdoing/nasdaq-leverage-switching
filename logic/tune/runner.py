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
from logic.common.data import download_fx, download_prices, _extract_close
from logic.backtest.runner import run_backtest
from utils.report import render_table_eaw


def _is_rate_limit_error(exc: Exception) -> bool:
    s = repr(exc).lower()
    return "yfratelimiterror" in s or "rate limit" in s


def _run_single(args: Tuple[Dict, Dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Timestamp]) -> Dict:
    base_settings, overrides, pre_prices, pre_fx, pre_bench, start_bound = args
    tuned = dict(base_settings)
    tuned.update(overrides)
    report = run_backtest(
        tuned,
        pre_prices=pre_prices,
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
    end_bound = pd.Timestamp.today().normalize()
    start_bound = end_bound - pd.DateOffset(months=settings["months_range"])

    try:
        pre_prices = download_prices(settings, start_bound)
        pre_fx = download_fx(start_bound)
        bench_raw = yf.download(settings["benchmarks"], start=start_bound, auto_adjust=True, progress=False)
        if bench_raw is None or len(bench_raw) == 0:
            raise ValueError(f"벤치마크 데이터를 받아오지 못했습니다: {settings['benchmarks']}")
        pre_bench = _extract_close(bench_raw, settings["benchmarks"])
    except Exception as exc:
        if _is_rate_limit_error(exc):
            raise SystemExit("yfinance YFRateLimitError: 잠시 후 다시 실행하세요.") from exc
        raise RuntimeError(f"프리패치 단계에서 데이터 로드에 실패했습니다: {exc}") from exc

    combos: List[Dict] = []
    for ma_s in tuning_config["ma_short"]:
        for ma_l in tuning_config["ma_long"]:
            for vol_lb in tuning_config["vol_lookback"]:
                for vol_cut in tuning_config["vol_cutoff"]:
                    for dd_cut in tuning_config["drawdown_cutoff"]:
                        combos.append(
                            {
                                "ma_short": int(ma_s),
                                "ma_long": int(ma_l),
                                "vol_lookback": int(vol_lb),
                                "vol_cutoff": float(vol_cut),
                                "drawdown_cutoff": float(dd_cut),
                            }
                        )

    total_cases = len(combos)
    workers = max_workers or cpu_count() or 1
    results: List[Dict] = []
    completed = 0
    next_progress = 1

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(_run_single, (settings, overrides, pre_prices, pre_fx, pre_bench, start_bound)): overrides
            for overrides in combos
        }
        for fut in as_completed(future_map):
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                overrides = future_map[fut]
                if _is_rate_limit_error(exc):
                    print("YFRateLimitError 감지: 잠시 후 다시 실행하세요.")
                    raise SystemExit(1) from exc
                print(f"[튜닝 경고] 조합 {overrides} 실패: {exc}")
                continue
            completed += 1
            progress = int(completed / total_cases * 100)
            if progress_cb and progress >= next_progress:
                progress_cb(completed, total_cases)
                next_progress = progress + 1
            if partial_cb and progress >= next_progress - 1:
                partial_cb(results, completed, total_cases)

    # 정렬: CAGR 내림차순
    results.sort(key=lambda x: x["cagr"], reverse=True)
    return results, {"start_ts": start_ts, "total": total_cases}


def render_top_table(results: List[Dict], top_n: int = 20) -> List[str]:
    headers = [
        "ma_short",
        "ma_long",
        "vol_lookback",
        "vol_cutoff",
        "drawdown_cutoff",
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
                str(p["ma_short"]),
                str(p["ma_long"]),
                str(p["vol_lookback"]),
                f"{p['vol_cutoff']:.2f}",
                f"{p['drawdown_cutoff']:.2f}",
                f"{row['cagr']*100:.2f}",
                f"{row['mdd']*100:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['vol']*100:.2f}",
            ]
        )
    return render_table_eaw(headers, rows, aligns)
