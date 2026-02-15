"""튜닝 실행 엔트리 포인트."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from logic.tune.runner import render_top_table, run_tuning

# 국가별 튜닝 설정
TUNING_CONFIG: dict[str, dict] = {
    "us": {
        "drawdown_buy_cutoff": np.arange(0.1, 3.1, 0.1),
        "drawdown_sell_cutoff": np.arange(0.1, 3.1, 0.1),
        "defense": [
            {"ticker": "SCHD", "name": "슈왑 미국 배당주 ETF"},
            {"ticker": "SPLV", "name": "인베스코 S&P500 저변동 ETF"},
            {"ticker": "SPHD", "name": "인베스코 고배당 저변동 ETF"},
            # {"ticker": "GLDM", "name": "SPDR 금 미니 ETF"},
            # {"ticker": "GDX", "name": "반에크 금광 ETF"},
            # {"ticker": "BRK-B", "name": "버크셔 해서웨이 B"},
            {"ticker": "VYM", "name": "뱅가드 미국 고배당주 ETF"},
            {"ticker": "DJD", "name": "인베스코 다우 존스 산업 평균 배당주 ETF"},
        ],
    },
    "kor": {
        "drawdown_buy_cutoff": np.arange(0.1, 3.1, 0.1),
        "drawdown_sell_cutoff": np.arange(0.1, 3.1, 0.1),
        "defense": [
            # {"ticker": "379800", "name": "KODEX 미국배당커버드콜액티브"},
            # {"ticker": "379800", "name": "KODEX 미국S&P500"},
            # {"ticker": "379810", "name": "KODEX 미국나스닥100"},
            {"ticker": "161510", "name": "PLUS 고배당주"},
            # {"ticker": "475350", "name": "RISE 버크셔포트폴리오TOP10"},
            # {"ticker": "473640", "name": "HANARO 글로벌금채굴기업"},
            # {"ticker": "489250", "name": "KODEX 미국배당다우존스"},
        ],
    },
}


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}시간 {m}분 {s}초"


def main() -> None:
    # CLI 인자로 country 지정 (기본값: us)
    country = sys.argv[1] if len(sys.argv) > 1 else "us"
    config_path = Path(f"config/{country}.json")

    if not config_path.exists():
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return

    if country not in TUNING_CONFIG:
        print(f"지원하지 않는 국가입니다: {country}")
        print(f"지원 국가: {list(TUNING_CONFIG.keys())}")
        return

    tuning_config = TUNING_CONFIG[country]

    start_ts = datetime.now()

    # 결과 폴더: zresults/{country}/
    out_dir = Path(f"zresults/{country}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tune_{start_ts.date()}.log"

    with config_path.open(encoding="utf-8") as f:
        settings = json.load(f)
    months_range = settings.get("months_range", 12)

    def write_partial(results: list[dict], completed: int, total: int) -> None:
        # 상위 10개만 중간 저장
        results.sort(key=lambda x: x["cagr"], reverse=True)
        table_lines = render_top_table(results, top_n=10, months_range=months_range)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"실행 시각: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"마켓: {country.upper()}\n")
            f.write(f"진행률: {completed}/{total} ({completed / total * 100:.1f}%)\n\n")
            f.write("=== 중간 결과 - 상위 10개 ===\n")
            for line in table_lines:
                f.write(line + "\n")

    def progress_cb(completed: int, total: int) -> None:
        pct = int(completed / total * 100)
        print(f"[튜닝 진행률] {pct}% ({completed}/{total})")

    print(f"[튜닝 시작] {start_ts.strftime('%Y-%m-%d %H:%M:%S')} ({country.upper()})")
    total_cases = 1
    for arr in tuning_config.values():
        total_cases *= len(arr)
    print(f"[튜닝 설정] 총 조합: {total_cases}개, 워커: auto (CPU)")

    try:
        results, meta = run_tuning(
            tuning_config,
            config_path=config_path,
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

    # 최적 파라미터로 config 파일 업데이트
    if not results:
        print(f"튜닝 결과가 없습니다. {config_path}을 변경하지 않습니다.")
    else:
        best_params = results[0]["params"]

        # 기존 config 파일 로드
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)

        # 파라미터 업데이트
        config["drawdown_buy_cutoff"] = round(float(best_params["drawdown_buy_cutoff"]), 2)
        config["drawdown_sell_cutoff"] = round(float(best_params["drawdown_sell_cutoff"]), 2)

        # defense 필드 업데이트 (_defense_obj에서 가져오기)
        defense_obj = best_params.get("_defense_obj")
        if defense_obj and isinstance(defense_obj, dict):
            config["defense"] = {
                "ticker": defense_obj.get("ticker", ""),
                "name": defense_obj.get("name", ""),
            }

        # backtested_date를 맨 위로 배치하기 위해 순서 재정렬
        ordered_config = {"backtested_date": datetime.now().date().isoformat()}
        for key, value in config.items():
            if key != "backtested_date":
                ordered_config[key] = value

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(ordered_config, f, ensure_ascii=False, indent=4)
        print(
            f"{config_path}을 최적 파라미터로 업데이트했습니다. (backtested_date={ordered_config['backtested_date']})"
        )

    table_lines = render_top_table(results, top_n=100, months_range=months_range)

    end_ts = datetime.now()
    elapsed = format_seconds((end_ts - start_ts).total_seconds())

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"실행 시각: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"종료 시각: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"마켓: {country.upper()}\n")
        f.write(f"걸린 시간: {elapsed}\n")
        f.write("\n")

        f.write("=== 튜닝 설정 ===\n")
        if meta and meta.get("period_start") and meta.get("period_end"):
            f.write(f"기간: {meta['period_start']} ~ {meta['period_end']} ({meta['period_months']} 개월)\n")
        else:
            f.write(f"기간: {start_ts.date()} ~ {end_ts.date()}\n")
        f.write("탐색 공간: ")
        parts = [f"{k} {len(v)}개" for k, v in tuning_config.items()]
        f.write(" × ".join(parts) + f" = {total_cases}개 조합\n")
        for k, v in tuning_config.items():
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
