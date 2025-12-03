"""백테스트 실행 엔트리 포인트."""

from datetime import datetime
from pathlib import Path

from config import INITIAL_CAPITAL_KRW
from logic.backtest.runner import run_backtest
from logic.common.settings import load_settings


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
        f.write(f"초기자본: {INITIAL_CAPITAL_KRW:,} | 시작일: {report['start']} | 종료일: {report['end']}\n\n")
        f.write("2. ========= 일자별 성과 ==========\n\n")
        if report.get("segment_lines"):
            f.write("=== 구간별 보유 요약 ===\n")
            for line in report["segment_lines"]:
                f.write(line + "\n")
            f.write("\n=== 일자별 상세 ===\n")
        for line in report["daily_log"]:
            f.write(line + "\n")
        f.write("\n")
        for line in report.get("used_settings_lines", []):
            f.write(line + "\n")
        f.write("\n")
        if report.get("weekly_summary_lines"):
            for line in report["weekly_summary_lines"]:
                f.write(line + "\n")
            f.write("\n")
        if report.get("monthly_summary_lines"):
            for line in report["monthly_summary_lines"]:
                f.write(line + "\n")
            f.write("\n")
        for line in report["asset_summary_lines"]:
            f.write(line + "\n")
        f.write("\n")
        for line in report["summary_lines"]:
            f.write(line + "\n")
        if report.get("bench_table_lines"):
            for line in report["bench_table_lines"]:
                f.write(line + "\n")

    print("=== Backtest 결과 ===")
    for k, v in report.items():
        if k == "daily_log" or k.endswith("_lines"):
            continue
        print(f"{k}: {v}")
    if report.get("used_settings_lines"):
        print("\n".join(report["used_settings_lines"]))
    # 콘솔에 종목별 성과 요약
    print("\n".join(report["asset_summary_lines"]))
    # 요약 섹션 콘솔 출력
    print("\n".join(report["summary_lines"]))
    if report.get("bench_table_lines"):
        for line in report["bench_table_lines"]:
            print(line)

    print(f"백테스트 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
