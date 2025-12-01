"""설정 로딩 유틸리티 (기본값/자동 보정 없음)."""

import json
from pathlib import Path
from typing import Dict, List

REQUIRED_KEYS: List[str] = [
    "signal_ticker",
    "trade_ticker",
    "defense_ticker",
    "ma_short",
    "ma_long",
    "drawdown_cutoff",
    "benchmarks",
    "months_range",
    "slippage",
]


def load_settings(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in settings]
    if missing:
        raise ValueError(f"settings.json에 필수 키가 없습니다: {missing}")

    if settings.get("defense_ticker") == "CASH":
        raise ValueError("defense_ticker에 CASH는 더 이상 허용되지 않습니다.")

    return settings
