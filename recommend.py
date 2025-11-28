"""추천 실행 엔트리 포인트."""

from datetime import datetime
from pathlib import Path

from logic.common.settings import load_settings
from logic.recommend.runner import run_recommend, write_recommend_log


def main() -> None:
    settings = load_settings(Path("settings.json"))
    report = run_recommend(settings)

    out_dir = Path("zresults")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"recommend_{datetime.now().date()}.log"
    write_recommend_log(report, out_path)

    # 콘솔 출력
    print("\n".join(report["status_lines"]))
    print("\n=== 추천 목록 ===")
    for line in report["table_lines"]:
        print(line)
    print(f"\n추천 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
