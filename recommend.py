import argparse
from datetime import datetime
from pathlib import Path

from logic.backtest.runner import run_backtest
from logic.backtest.settings import load_settings
from utils.slack import send_slack_recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="추천 실행 엔트리 포인트")
    parser.add_argument("country", nargs="?", default="us", help="대상 국가 (us/kor)")
    parser.add_argument("--slack", action="store_true", help="결과를 Slack으로 전송")
    args = parser.parse_args()

    country = args.country
    config_path = Path(f"config/{country}.json")

    if not config_path.exists():
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return

    settings = load_settings(config_path)

    try:
        result = run_backtest(settings)
    except Exception as exc:
        if "YFRateLimitError" in repr(exc) or "rate limit" in repr(exc).lower():
            print("YFRateLimitError: 요청이 너무 많습니다. 잠시 후 다시 실행하세요.")
            return
        raise

    # 마지막 날 추천 정보 추출
    last_target = result["last_target"]
    rec_data = result["recommendation_data"]
    end_date = rec_data["last_date"]

    # 티커와 이름 가져오기
    offense_ticker = settings["offense_ticker"]
    offense_name = settings.get("offense_name", offense_ticker)
    defense_ticker = settings["defense_ticker"]
    defense_name = settings.get("defense_name", defense_ticker)

    last_prices = rec_data["last_prices"]
    last_returns = rec_data["last_returns"]
    current_dd = rec_data["current_drawdown"]
    buy_cutoff = rec_data["buy_cutoff"]
    sell_cutoff = rec_data["sell_cutoff"]
    needed_recovery = rec_data["needed_recovery"]

    # 통화 기호 (한국은 원화, 미국은 달러)
    market = settings.get("market", "us")
    currency_symbol = "₩" if market == "kor" else "$"

    # 티커+이름 매핑
    ticker_names = {
        offense_ticker: offense_name,
        defense_ticker: defense_name,
    }

    # 추천 출력 생성
    table_lines = []
    assets = [offense_ticker, defense_ticker]
    for sym in assets:
        name = ticker_names.get(sym, sym)
        display_name = f"{sym}({name})" if name != sym else sym

        price = last_prices.get(sym, 0.0)
        ret = last_returns.get(sym, 0.0)

        if sym == last_target:
            status = "BUY ✅️"
            note = "타깃"
        elif sym == offense_ticker:
            status = "WAIT ⏳️"
            # 공격 자산이 타깃이 아닌 경우: DD 정보 표시
            note = f"DD {current_dd * 100:.2f}% (매수컷 -{buy_cutoff:.2f}%, 필요 {needed_recovery:+.2f}%)"
        else:
            status = "WAIT ⏳️"
            note = "방어"

        table_lines.append(f"📌 {display_name}")
        table_lines.append(f"  상태: {status}")
        table_lines.append(f"  일간: {ret * 100:+.2f}%")
        table_lines.append(f"  현재가: {currency_symbol}{price:,.2f}")
        if note:
            table_lines.append(f"  비고: {note}")
        table_lines.append("")

    # 타깃 이름
    target_name = ticker_names.get(last_target, last_target)
    target_display = f"{last_target}({target_name})" if target_name != last_target else last_target

    # 로그 파일 저장: zresults/{country}/
    out_dir = Path(f"zresults/{country}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"recommend_{datetime.now().date()}.log"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"추천 로그 생성: {datetime.now().isoformat()}\n")
        f.write(f"마켓: {country.upper()}\n\n")
        f.write("=== 추천 목록 ===\n")
        for line in table_lines:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"[INFO] 기준일: {end_date}\n")
        f.write(f"[INFO] 최종 타깃: {target_display}\n")
        f.write(f"[INFO] 적용 파라미터: {defense_ticker} / Buy {buy_cutoff}% / Sell {sell_cutoff}%\n")

    print(f"\n추천 결과 저장: {out_path}")

    # Slack 알림 전송
    if args.slack:
        tuning_meta = {
            "defense_ticker": settings["defense_ticker"],
            "buy_cutoff": buy_cutoff,
            "sell_cutoff": sell_cutoff,
            "cagr": result.get("cagr", 0.0),
        }
        send_slack_recommendation(
            country=country,
            as_of=end_date,
            target_display=target_display,
            table_lines=table_lines,
            tuning_meta=tuning_meta,
        )


if __name__ == "__main__":
    main()
