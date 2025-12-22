"""데이터 다운로드/전처리 공용 함수."""

import warnings

# pykrx의 pkg_resources 사용 경고 억제
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pandas as pd
import yfinance as yf


def compute_bounds(settings: dict, end_bound: pd.Timestamp | None = None):
    """백테스트/튜닝/추천 모두 동일한 기간 산정 로직을 사용하도록 범위를 계산."""
    end = end_bound or pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=settings["months_range"])
    warmup_bdays = 252  # 12개월 영업일 (신호 고점 계산용)
    warmup_start = start - pd.offsets.BDay(warmup_bdays)
    return start, warmup_start, end


def _extract_field(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
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
            raise ValueError(f"{field} 컬럼을 찾지 못했습니다. 사용 가능 컬럼: {list(data.columns)}")
        out = data.xs(field_key, axis=1, level=level_idx)
    else:
        candidates = [c for c in [field, field.capitalize()] if c in data.columns]
        field_col = candidates[0] if candidates else data.columns[0]
        out = data[[field_col]].rename(columns={field_col: tickers[0]})

    out = out.dropna(how="all")
    return out


# =============================================================================
# 미국 시장 (yfinance)
# =============================================================================


def _download_prices_us(settings: dict, start) -> pd.DataFrame:
    """미국 시장 종가 데이터를 yfinance로 다운로드."""
    tickers = list(
        {
            settings["offense_ticker"],
            settings["signal_ticker"],
            settings["defense_ticker"],
        }
    )
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")
    prices = _extract_field(data, "Close", tickers)
    needed = [
        settings["offense_ticker"],
        settings["signal_ticker"],
        settings["defense_ticker"],
    ]
    prices = prices.dropna(subset=needed)
    if prices.empty:
        raise ValueError(f"가격 데이터가 비어 있습니다: {tickers}")
    return prices


def _download_opens_us(settings: dict, start) -> pd.DataFrame:
    """미국 시장 시가 데이터를 yfinance로 다운로드."""
    tickers = list(
        {
            settings["offense_ticker"],
            settings["signal_ticker"],
            settings["defense_ticker"],
        }
    )
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"시가 데이터를 받아오지 못했습니다: {tickers}")
    opens = _extract_field(data, "Open", tickers)
    needed = [
        settings["offense_ticker"],
        settings["signal_ticker"],
        settings["defense_ticker"],
    ]
    opens = opens.dropna(subset=needed)
    if opens.empty:
        raise ValueError(f"시가 데이터가 비어 있습니다: {tickers}")
    return opens


def _download_fx_us(start) -> pd.Series:
    """USD/KRW 환율(원/달러)을 받아온다."""
    fx_raw = yf.download("USDKRW=X", start=start, auto_adjust=True, progress=False)
    if fx_raw is None or len(fx_raw) == 0:
        raise ValueError("환율(USDKRW) 데이터를 받아오지 못했습니다.")
    fx = _extract_field(fx_raw, "Close", ["USDKRW=X"]).squeeze()
    fx.name = "USDKRW"
    if fx.empty:
        raise ValueError("환율(USDKRW) 데이터가 비어 있습니다.")
    return fx


# =============================================================================
# 한국 시장 (pykrx)
# =============================================================================


def _download_prices_kor(settings: dict, start) -> pd.DataFrame:
    """한국 시장 종가 데이터를 pykrx로 다운로드."""
    try:
        from pykrx import stock as pykrx_stock
    except ImportError as e:
        raise ImportError("pykrx 패키지가 설치되어 있지 않습니다. pip install pykrx") from e

    tickers = list(
        {
            settings["offense_ticker"],
            settings["signal_ticker"],
            settings["defense_ticker"],
        }
    )
    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp.today().strftime("%Y%m%d")

    dfs = {}
    for ticker in tickers:
        df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if df is not None and not df.empty:
            dfs[ticker] = df["종가"]

    if not dfs:
        raise ValueError(f"가격 데이터를 받아오지 못했습니다: {tickers}")

    prices = pd.DataFrame(dfs)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna()
    if prices.empty:
        raise ValueError(f"가격 데이터가 비어 있습니다: {tickers}")
    return prices


def _download_opens_kor(settings: dict, start) -> pd.DataFrame:
    """한국 시장 시가 데이터를 pykrx로 다운로드."""
    try:
        from pykrx import stock as pykrx_stock
    except ImportError as e:
        raise ImportError("pykrx 패키지가 설치되어 있지 않습니다. pip install pykrx") from e

    tickers = list(
        {
            settings["offense_ticker"],
            settings["signal_ticker"],
            settings["defense_ticker"],
        }
    )
    start_str = pd.Timestamp(start).strftime("%Y%m%d")
    end_str = pd.Timestamp.today().strftime("%Y%m%d")

    dfs = {}
    for ticker in tickers:
        df = pykrx_stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if df is not None and not df.empty:
            dfs[ticker] = df["시가"]

    if not dfs:
        raise ValueError(f"시가 데이터를 받아오지 못했습니다: {tickers}")

    opens = pd.DataFrame(dfs)
    opens.index = pd.to_datetime(opens.index)
    opens = opens.dropna()
    if opens.empty:
        raise ValueError(f"시가 데이터가 비어 있습니다: {tickers}")
    return opens


def _download_fx_kor(start) -> pd.Series:
    """한국 시장은 원화 기준이므로 환율 1.0 고정."""
    date_range = pd.date_range(start, pd.Timestamp.today(), freq="B")
    fx = pd.Series(1.0, index=date_range, name="KRW")
    return fx


# =============================================================================
# 공용 인터페이스 (market에 따라 분기)
# =============================================================================


def download_prices(settings: dict, start) -> pd.DataFrame:
    """시장에 따라 적절한 데이터 소스에서 종가를 다운로드."""
    market = settings.get("market", "us")
    if market == "kor":
        return _download_prices_kor(settings, start)
    else:
        return _download_prices_us(settings, start)


def download_opens(settings: dict, start) -> pd.DataFrame:
    """시장에 따라 적절한 데이터 소스에서 시가를 다운로드."""
    market = settings.get("market", "us")
    if market == "kor":
        return _download_opens_kor(settings, start)
    else:
        return _download_opens_us(settings, start)


def download_fx(settings: dict, start) -> pd.Series:
    """시장에 따라 적절한 환율 데이터를 반환."""
    market = settings.get("market", "us")
    if market == "kor":
        return _download_fx_kor(start)
    else:
        return _download_fx_us(start)
