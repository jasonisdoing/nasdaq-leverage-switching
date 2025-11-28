"""백테스트 실행 엔트리 포인트."""

from datetime import datetime
from pathlib import Path

from logic.backtest.runner import run_backtest
from logic.common.settings import load_settings
from config import INITIAL_CAPITAL_KRW


def main() -> None:
    settings_path = Path("settings.json")
    settings = load_settings(settings_path)
    try:
        report = run_backtest(settings)
    except Exception as exc:
        if "YFRateLimitError" in repr(exc) or "rate limit" in repr(exc).lower():
            print("YFRateLimitError: 요청이 너무 많습니다. 잠시 후 다시 실행하세요.")
            return
        raise

    out_dir = Path("zresults")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_{datetime.now().date()}.log"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"백테스트 로그 생성: {datetime.now().isoformat()}\n")
        f.write(
            f"초기자본: {INITIAL_CAPITAL_KRW:,} | 시작일: {report['start']} | "
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
        f.write("\n")
        if report.get("bench_table_lines"):
            f.write("벤치마크/전략 기간 수익률 비교\n")
            for line in report["bench_table_lines"]:
                f.write(line + "\n")

    print("=== Backtest 결과 ===")
    for k, v in report.items():
        if k == "daily_log" or k.endswith("_lines"):
            continue
        print(f"{k}: {v}")
    # 요약 섹션 콘솔 출력
    print("\n".join(report["summary_lines"]))
    if report.get("bench_table_lines"):
        print("벤치마크/전략 기간 수익률 비교")
        for line in report["bench_table_lines"]:
            print(line)


if __name__ == "__main__":
    main()
