"""튜닝 실행 엔트리 포인트."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from logic.tune.runner import render_top_table, run_tuning
from utils.report import render_table_eaw

# 탐색 범위(필수 키만 명시, 기본값/자동 보정 없음)
# 최근 결과를 반영해 유효 구간을 좁힌 버전
TUNING_CONFIG: Dict[str, np.ndarray] = {
    "drawdown_buy_cutoff": np.arange(0.1, 3.1, 0.1),
    "drawdown_sell_cutoff": np.arange(0.1, 3.1, 0.1),
    "defense_ticker": [
        "SCHD",
        "SGOV",
        "SPLV",
        "DIVO",
        "JEPI",
        "GLDM",
    ],
}


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}시간 {m}분 {s}초"


def main() -> None:
    start_ts = datetime.now()

    out_dir = Path("zresults")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tune_{start_ts.date()}.log"

    with Path("settings.json").open(encoding="utf-8") as f:
        settings = json.load(f)
    months_range = settings.get("months_range", "N")

    def write_partial(results: List[Dict], completed: int, total: int) -> None:
        # 상위 10개만 중간 저장
        results.sort(key=lambda x: x["cagr"], reverse=True)
        table_lines = render_top_table(results, top_n=10, months_range=months_range)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"실행 시각: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"진행률: {completed}/{total} ({completed/total*100:.1f}%)\n\n")
            f.write("=== 중간 결과 - 상위 10개 ===\n")
            for line in table_lines:
                f.write(line + "\n")

    def progress_cb(completed: int, total: int) -> None:
        pct = int(completed / total * 100)
        print(f"[튜닝 진행률] {pct}% ({completed}/{total})")

    print(f"[튜닝 시작] {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    total_cases = 1
    for arr in TUNING_CONFIG.values():
        total_cases *= len(arr)
    print(f"[튜닝 설정] 총 조합: {total_cases}개, 워커: auto (CPU)")

    try:
        results, meta = run_tuning(
            TUNING_CONFIG,
            months_range=months_range,
            max_workers=None,  # None -> CPU 코어 수 자동
            progress_cb=progress_cb,
            partial_cb=write_partial,
        )
    except SystemExit as exc:
        msg = str(exc)
        if "YFRateLimitError" in msg or "rate" in msg.lower():
            print("YFRateLimitError: 요청이 너무 많습니다. 잠시 후 다시 실행하세요.")
            return
        raise

    # 정렬: CAGR 내림차순
    results.sort(key=lambda x: x["cagr"], reverse=True)
    top_n = results[:100]

    # 최적 파라미터로 settings.json 업데이트
    if not results:
        print("튜닝 결과가 없습니다. settings.json을 변경하지 않습니다.")
    else:
        best = results[0]["params"]
        # 부동소수 표기 방지를 위해 소수점 2자리로 반올림
        best["drawdown_buy_cutoff"] = round(float(best["drawdown_buy_cutoff"]), 2)
        best["drawdown_sell_cutoff"] = round(float(best["drawdown_sell_cutoff"]), 2)
        best["backtested_date"] = datetime.now().date().isoformat()
        settings_path = Path("settings.json")
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=4)
        print(
            f"settings.json을 최적 파라미터로 업데이트했습니다. (backtested_date={best['backtested_date']})"
        )

    table_lines = render_top_table(results, top_n=100, months_range=months_range)

    end_ts = datetime.now()
    elapsed = format_seconds((end_ts - start_ts).total_seconds())

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"실행 시각: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"종료 시각: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"걸린 시간: {elapsed}\n")
        f.write("\n")

        f.write("=== 튜닝 설정 ===\n")
        if meta and meta.get("period_start") and meta.get("period_end"):
            f.write(
                f"기간: {meta['period_start']} ~ {meta['period_end']} ({meta['period_months']} 개월)\n"
            )
        else:
            f.write(f"기간: {start_ts.date()} ~ {end_ts.date()}\n")
        f.write("탐색 공간: ")
        parts = [f"{k} {len(v)}개" for k, v in TUNING_CONFIG.items()]
        f.write(" × ".join(parts) + f" = {total_cases}개 조합\n")
        for k, v in TUNING_CONFIG.items():
            if isinstance(v[0], str):
                f.write(f"  {k}: {list(v)}\n")
            else:
                if len(v) == 1:
                    f.write(f"  {k}: {v[0]}\n")
                else:
                    f.write(f"  {k}: {v[0]}~{v[-1]}\n")
        f.write("\n")

        f.write(f"=== 결과 - 기간: {months_range} 개월 | 정렬 기준: CAGR ===\n")
        for line in table_lines[:200]:
            f.write(line + "\n")
        if len(results) > len(top_n):
            f.write(f"... (총 {total_cases}개 중 상위 {len(top_n)}개 표시)\n")

    print(f"튜닝 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
