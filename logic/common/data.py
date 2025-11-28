"""데이터 다운로드/전처리 공용 함수."""

from typing import Dict, List

import pandas as pd
import yfinance as yf


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


def download_prices(settings: Dict, start) -> pd.DataFrame:
    tickers = list(set(settings["trade_symbols"] + [settings["signal_symbol"]]))
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
    prices = _extract_close(data, tickers)
    prices = prices.dropna(subset=settings["trade_symbols"] + [settings["signal_symbol"]])
    if prices.empty:
        raise ValueError(f"가격 데이터가 비어 있습니다: {tickers}")
    return prices


def download_fx(start) -> pd.Series:
    """USD/KRW 환율(원/달러)을 받아온다."""
    fx_raw = yf.download("USDKRW=X", start=start, auto_adjust=True, progress=False)
    if fx_raw is None or len(fx_raw) == 0:
        raise ValueError("환율(USDKRW) 데이터를 받아오지 못했습니다.")
    fx = _extract_close(fx_raw, ["USDKRW=X"]).squeeze()
    fx.name = "USDKRW"
    if fx.empty:
        raise ValueError("환율(USDKRW) 데이터가 비어 있습니다.")
    return fx
