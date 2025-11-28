"""데이터 다운로드/전처리 공용 함수."""

from typing import Dict, List

import pandas as pd
import yfinance as yf

def compute_bounds(settings: Dict, end_bound: pd.Timestamp | None = None):
    """백테스트/튜닝/추천 모두 동일한 기간 산정 로직을 사용하도록 범위를 계산."""
    end = end_bound or pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=settings["months_range"])
    warmup_bdays = max(settings["ma_long"] * 2 + 10, 300)
    warmup_start = start - pd.offsets.BDay(warmup_bdays)
    return start, warmup_start, end


def _extract_field(data: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
    """yfinance 다운로드 결과에서 특정 필드(Open/Close 등)를 안전하게 추출."""
    key = field.lower()
    if isinstance(data.columns, pd.MultiIndex):
        candidates = [key, f"adj {key}"]
        level_idx = None
        field_key = None
        for level in range(data.columns.nlevels):
            level_values = data.columns.get_level_values(level)
            for cand in candidates:
                matches = [v for v in level_values if str(v).lower() == cand]
                if matches:
                    level_idx = level
                    field_key = matches[0]
                    break
            if level_idx is not None:
                break
        if level_idx is None:
            raise ValueError(
                f"{field} 컬럼을 찾지 못했습니다. 사용 가능 컬럼: {list(data.columns)}"
            )
        out = data.xs(field_key, axis=1, level=level_idx)
    else:
        candidates = [c for c in [field, field.capitalize()] if c in data.columns]
        field_col = candidates[0] if candidates else data.columns[0]
        out = data[[field_col]].rename(columns={field_col: tickers[0]})

    out = out.dropna(how="all")
    return out


def download_prices(settings: Dict, start) -> pd.DataFrame:
    tickers = list(set(settings["trade_symbols"] + [settings["signal_symbol"]]))
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
    prices = _extract_field(data, "Close", tickers)
    prices = prices.dropna(subset=settings["trade_symbols"] + [settings["signal_symbol"]])
    if prices.empty:
        raise ValueError(f"가격 데이터가 비어 있습니다: {tickers}")
    return prices


def download_opens(settings: Dict, start) -> pd.DataFrame:
    tickers = list(set(settings["trade_symbols"] + [settings["signal_symbol"]]))
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"시가 데이터를 받아오지 못했습니다: {tickers}")
    opens = _extract_field(data, "Open", tickers)
    opens = opens.dropna(subset=settings["trade_symbols"] + [settings["signal_symbol"]])
    if opens.empty:
        raise ValueError(f"시가 데이터가 비어 있습니다: {tickers}")
    return opens


def download_fx(start) -> pd.Series:
    """USD/KRW 환율(원/달러)을 받아온다."""
    fx_raw = yf.download("USDKRW=X", start=start, auto_adjust=True, progress=False)
    if fx_raw is None or len(fx_raw) == 0:
        raise ValueError("환율(USDKRW) 데이터를 받아오지 못했습니다.")
    fx = _extract_field(fx_raw, "Close", ["USDKRW=X"]).squeeze()
    fx.name = "USDKRW"
    if fx.empty:
        raise ValueError("환율(USDKRW) 데이터가 비어 있습니다.")
    return fx
