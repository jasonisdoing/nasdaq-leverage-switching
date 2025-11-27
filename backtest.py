"""QQQM/QLD/TQQQ 레버리지 전환 기본 백테스트 스크립트."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from config import INITIAL_CAPITAL_KRW
from utils.report import format_kr_money, render_table_eaw


def load_settings(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    required_keys = [
        "signal_symbol",
        "trade_symbols",
        "ma_short",
        "ma_long",
        "vol_lookback",
        "vol_cutoff",
        "drawdown_cutoff",
        "benchmarks",
        "months_range",
    ]
    missing = [k for k in required_keys if k not in settings]
    if missing:
        raise ValueError(f"settings.json에 필수 키가 없습니다: {missing}")

    return settings


def _extract_close(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """yfinance 다운로드 결과에서 종가 영역을 안전하게 추출."""
    if isinstance(data.columns, pd.MultiIndex):
        candidates = ["close", "adj close"]
        close_level = None
        close_key = None
        for level in range(data.columns.nlevels):
            level_values = data.columns.get_level_values(level)
            for cand in candidates:
                matches = [v for v in level_values if str(v).lower() == cand]
                if matches:
                    close_level = level
                    close_key = matches[0]
                    break
            if close_level is not None:
                break
        if close_level is None:
            raise ValueError(
                f"종가 컬럼을 찾지 못했습니다. 사용 가능 컬럼: {list(data.columns)}"
            )
        prices = data.xs(close_key, axis=1, level=close_level)
    else:
        close_candidates = [c for c in ["Close", "Adj Close"] if c in data.columns]
        close_col = close_candidates[0] if close_candidates else data.columns[0]
        prices = data[[close_col]].rename(columns={close_col: tickers[0]})

    prices = prices.dropna(how="all")
    return prices


def download_prices(settings: Dict, start: str) -> pd.DataFrame:
    tickers = list(set(settings["trade_symbols"] + [settings["signal_symbol"]]))
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = _extract_close(data, tickers)
    prices = prices.dropna(subset=settings["trade_symbols"] + [settings["signal_symbol"]])
    return prices


def download_fx(start: str) -> pd.Series:
    """USD/KRW 환율(원/달러)을 받아온다."""
    fx_raw = yf.download("USDKRW=X", start=start, auto_adjust=True, progress=False)
    fx = _extract_close(fx_raw, ["USDKRW=X"]).squeeze()
    fx.name = "USDKRW"
    return fx


def compute_signals(prices: pd.Series, settings: Dict) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    df["ma_short"] = prices.rolling(settings["ma_short"]).mean()
    df["ma_long"] = prices.rolling(settings["ma_long"]).mean()
    df["vol"] = prices.pct_change().rolling(settings["vol_lookback"]).std() * np.sqrt(252)
    peak = prices.cummax()
    df["drawdown"] = prices / peak - 1.0
    return df.dropna()


def pick_target(row, settings: Dict) -> str:
    if row["drawdown"] <= -settings["drawdown_cutoff"]:
        return "QQQM"
    if row["ma_short"] > row["ma_long"] and row["vol"] < settings["vol_cutoff"]:
        return "TQQQ"
    if row["ma_short"] > row["ma_long"]:
        return "QLD"
    return "QQQM"


def run_backtest(settings: Dict) -> Dict[str, object]:
    end_bound = pd.Timestamp.today().normalize()
    start_bound = end_bound - pd.DateOffset(months=settings["months_range"])

    prices = download_prices(settings, start_bound)
    signal_df = compute_signals(prices[settings["signal_symbol"]], settings)
    common_index = signal_df.index.intersection(prices.index)
    prices = prices.loc[common_index]
    signal_df = signal_df.loc[common_index]

    returns = prices[settings["trade_symbols"]].pct_change().dropna()
    signal_df = signal_df.loc[returns.index]
    signal_df["target"] = signal_df.apply(lambda row: pick_target(row, settings), axis=1)

    # 환율 데이터(원/달러)
    fx = download_fx(start_bound)
    fx = fx.reindex(returns.index, method="ffill").dropna()

    # 초기 자본: 원화 -> 달러
    first_date = returns.index[0]
    init_fx = fx.loc[first_date]
    initial_capital_usd = INITIAL_CAPITAL_KRW / init_fx
    capital_usd = initial_capital_usd

    equity = []
    daily_rets = []
    daily_log: List[str] = []
    hold_days = {s: 0 for s in settings["trade_symbols"]}
    asset_pnl = {s: 0.0 for s in settings["trade_symbols"]}
    asset_exposure_days = {s: 0 for s in settings["trade_symbols"]}
    trade_counts = {s: 0 for s in settings["trade_symbols"]}
    win_days = {s: 0 for s in settings["trade_symbols"]}
    trade_days = {s: 0 for s in settings["trade_symbols"]}
    prev_target = None

    for date, row in returns.iterrows():
        target = signal_df.at[date, "target"]
        daily_ret = row[target]

        capital_before = capital_usd
        pnl = capital_before * daily_ret
        capital_usd *= 1 + daily_ret

        equity.append(capital_usd)
        daily_rets.append(daily_ret)

        # 보유일/노출일 및 티커별 기여도
        for sym in settings["trade_symbols"]:
            if sym == target:
                hold_days[sym] += 1
                asset_exposure_days[sym] += 1
                asset_pnl[sym] += pnl
                trade_days[sym] += 1
                if row[sym] > 0:
                    win_days[sym] += 1
                if prev_target != sym:
                    trade_counts[sym] += 1
            else:
                hold_days[sym] = 0

        weights = {s: (1.0 if s == target else 0.0) for s in settings["trade_symbols"]}
        cash_value = 0.0
        total_value = capital_usd + cash_value
        fx_today = fx.loc[date]
        krw_value = total_value * fx_today

        # 테이블 데이터 준비
        headers = [
            "#",
            "티커",
            "상태",
            "보유일",
            "현재가",
            "일간(%)",
            "수량",
            "금액",
            "평가손익",
            "평가(%)",
            "누적손익",
            "누적(%)",
            "비중",
            "문구",
        ]
        aligns = [
            "center",
            "center",
            "center",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "left",
        ]
        rows = []

        # 현금 행
        rows.append(
            [
                "1",
                "CASH",
                "HOLD" if cash_value > 0 else "WAIT",
                "0",
                "1",
                "+0.0%",
                f"{cash_value:,.2f}",
                f"{cash_value:,.2f}",
                "0",
                "+0.0%",
                "0",
                "+0.0%",
                f"{cash_value/total_value:0.1%}",
                "",
            ]
        )

        # 자산 행들
        for idx, sym in enumerate(settings["trade_symbols"], start=2):
            price = prices.at[date, sym]
            ret = row[sym]
            weight = weights[sym]
            position_value = total_value * weight
            qty = position_value / price if price > 0 else 0.0

            if sym == target and prev_target != sym:
                state = "BUY"
            elif sym == target and prev_target == sym:
                state = "HOLD"
            elif sym != target and prev_target == sym:
                state = "SELL"
            else:
                state = "WAIT"

            eval_pnl = pnl if sym == target else 0.0
            eval_pct = ret if weight > 0 else 0.0
            cumulative_pnl = total_value - initial_capital_usd if sym == target else 0.0
            cumulative_pct = (
                total_value / initial_capital_usd - 1 if sym == target else 0.0
            )

            rows.append(
                [
                    str(idx),
                    sym,
                    state,
                    str(hold_days[sym]),
                    f"{price:,.2f}",
                    f"{ret:+.2%}",
                    f"{qty:,.4f}",
                    f"{position_value:,.2f}",
                    f"{eval_pnl:,.2f}",
                    f"{eval_pct*100:+.2f}%",
                    f"{cumulative_pnl:,.2f}",
                    f"{cumulative_pct*100:+.2f}%",
                    f"{weight:0.0%}",
                    "타깃" if sym == target else "",
                ]
            )

        table_lines = render_table_eaw(headers, rows, aligns)

        header_line = (
            f"{date.date()} | 목표: {target} | 총자산: ${total_value:,.2f} "
            f"({format_kr_money(krw_value)}) | 일간수익률: {daily_ret:+.2%} | "
            f"누적수익률: {(total_value / initial_capital_usd - 1):+.2%}"
        )

        daily_log.append(header_line)
        daily_log.extend(table_lines)
        daily_log.append("")  # 빈 줄

        # 다음 일자를 위한 상태 업데이트
        prev_target = target

    equity_series = pd.Series(equity, index=returns.index)
    daily_rets_series = pd.Series(daily_rets, index=returns.index)

    cagr = (
        (equity_series.iloc[-1] / initial_capital_usd) ** (252 / len(equity_series))
        - 1
    )
    vol = daily_rets_series.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else np.nan
    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1
    max_dd = drawdown.min()
    win_rate = (daily_rets_series > 0).mean()
    last_fx = fx.iloc[-1]
    final_krw = equity_series.iloc[-1] * last_fx
    period_return = equity_series.iloc[-1] / initial_capital_usd - 1
    start_date = equity_series.index[0].date()
    end_date = equity_series.index[-1].date()
    total_pnl = equity_series.iloc[-1] - initial_capital_usd

    # 벤치마크(VOO, QQQM)의 기간 수익률 및 CAGR
    bench_tickers = settings["benchmarks"]
    bench_lines = []
    try:
        bench_raw = yf.download(bench_tickers, start=start_bound, auto_adjust=True, progress=False)
        bench_prices = _extract_close(bench_raw, bench_tickers)
        bench_prices = bench_prices.reindex(equity_series.index, method="ffill")
        for t in bench_tickers:
            if t not in bench_prices.columns:
                continue
            series = bench_prices[t].dropna()
            if series.empty:
                continue
            br = series.iloc[-1] / series.iloc[0] - 1
            cagr_b = (1 + br) ** (252 / len(series)) - 1
            bench_lines.append((t, br, cagr_b))
    except Exception:
        bench_lines = []

    # 요약 섹션
    months = max(1, round((end_date - start_date).days / 30.0, 1))
    summary_lines = [
        "8. ========= 백테스트 결과 요약 ==========",
        f"| 기간: {start_date} ~ {end_date} ({months} 개월)",
        f"| 초기 자본: {format_kr_money(INITIAL_CAPITAL_KRW)}",
        f"| 최종 자산: {format_kr_money(final_krw)}",
        f"| 기간수익률(%): {period_return*100:+.2f}%",
    ]
    if bench_lines:
        summary_lines.append("| 벤치마크 기간수익률(%)")
        for i, (t, br, _) in enumerate(bench_lines, start=1):
            summary_lines.append(f"| {i}. {t}: {br*100:+.2f}%")
        summary_lines.append("| 벤치마크 CAGR(%)")
        for i, (t, _, cagr_b) in enumerate(bench_lines, start=1):
            summary_lines.append(f"| {i}. {t}: {cagr_b*100:+.2f}%")
    summary_lines.extend(
        [
            f"| CAGR(%): {cagr*100:+.2f}%",
            f"| MDD(%): {max_dd*100:.2f}%",
        ]
    )

    # 종목별 성과 요약
    asset_rows = []
    for idx, sym in enumerate(["CASH"] + settings["trade_symbols"], start=1):
        if sym == "CASH":
            pnl_usd = 0.0
            days = 0
            trades = 0
            win_rate = 0.0
        else:
            pnl_usd = asset_pnl[sym]
            days = asset_exposure_days[sym]
            trades = trade_counts[sym]
            w_days = win_days[sym]
            t_days = trade_days[sym]
            win_rate = w_days / t_days * 100 if t_days > 0 else 0.0
        krw_pnl = pnl_usd * last_fx
        asset_rows.append(
            [
                str(idx),
                sym,
                f"{pnl_usd:,.2f}",
                f"{pnl_usd:,.2f}",
                format_kr_money(krw_pnl),
                str(days),
                str(trades),
                f"{win_rate:.1f}%",
                "주요 기여" if abs(pnl_usd) >= abs(total_pnl) * 0.1 and total_pnl != 0 else "",
            ]
        )

    asset_headers = ["#", "티커", "기여도(USD)", "총손익(USD)", "총손익(KRW)", "노출일수", "거래횟수", "승률", "비고"]
    asset_aligns = ["center", "center", "right", "right", "right", "right", "right", "right", "left"]
    asset_summary_lines = ["7. ========= 종목별 성과 요약 =========="]
    asset_summary_lines.extend(render_table_eaw(asset_headers, asset_rows, asset_aligns))

    return {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "final_equity_usd": round(equity_series.iloc[-1], 2),
        "final_equity_krw": round(final_krw, 0),
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "last_target": signal_df["target"].iloc[-1],
        "daily_log": daily_log,
        "summary_lines": summary_lines,
        "asset_summary_lines": asset_summary_lines,
    }


if __name__ == "__main__":
    settings_path = Path("settings.json")
    settings = load_settings(settings_path)
    report = run_backtest(settings)

    out_dir = Path("zresults")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_{datetime.now().date()}.log"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"백테스트 로그 생성: {datetime.now().isoformat()}\n")
        f.write(
            f"초기자본: {format_kr_money(INITIAL_CAPITAL_KRW)} | 시작일: {report['start']} | "
            f"종료일: {report['end']}\n\n"
        )
        f.write("2. ========= 일자별 성과 ==========\n\n")
        for line in report["daily_log"]:
            f.write(line + "\n")
        f.write("\n")
        for line in report["asset_summary_lines"]:
            f.write(line + "\n")
        f.write("\n")
        for line in report["summary_lines"]:
            f.write(line + "\n")

    print("=== Backtest 결과 ===")
    for k, v in report.items():
        if k == "daily_log":
            continue
        print(f"{k}: {v}")
