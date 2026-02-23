import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import MARKET_SCHEDULES
from logic.backtest.runner import run_backtest
from logic.backtest.settings import load_settings
from utils.slack import send_slack_recommendation


def is_market_open(country: str) -> bool:
    """현재 시간이 해당 국가의 거래 시간인지 확인합니다."""
    schedule = MARKET_SCHEDULES.get(country)
    if not schedule:
        return True  # 스케줄 정보가 없으면 항상 열려있다고 가정 (또는 에러 처리)

    tz = ZoneInfo(schedule["timezone"])
    now = datetime.now(tz)

    # 주말 체크 (월=0, ..., 일=6)
    if now.weekday() >= 5:
        return False

    current_time = now.time()
    return schedule["open"] <= current_time <= schedule["close"]


def load_previous_state(country: str) -> dict:
    """저장된 이전 추천 상태를 로드합니다."""
    state_path = Path(f"state/last_recommendation_{country}.json")
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_current_state(country: str, state: dict) -> None:
    """현재 추천 상태를 저장합니다."""
    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"last_recommendation_{country}.json"
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="추천 실행 엔트리 포인트")
    parser.add_argument("country", nargs="?", default="us", help="대상 국가 (us/kor)")
    parser.add_argument("--slack", action="store_true", help="결과를 Slack으로 전송")
    parser.add_argument("--auto", action="store_true", help="자동 실행 모드 (장 운영 시간 체크 수행)")
    args = parser.parse_args()

    country = args.country

    # 자동 실행 모드일 때만 장 운영 시간 체크
    if args.auto and not is_market_open(country):
        print(f"[{country.upper()}] 장 운영 시간이 아닙니다. 실행을 건너뜁니다.")
        return

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

    # 이전 상태 로드 및 변경 여부 확인
    prev_state = load_previous_state(country)
    prev_target = prev_state.get("target")
    is_changed = (prev_target is not None) and (prev_target != last_target)

    # 현재 상태 저장
    current_state = {"date": end_date, "target": last_target, "updated_at": datetime.now().isoformat()}
    save_current_state(country, current_state)

    # 티커와 이름 가져오기
    offense_ticker = settings["offense_ticker"]
    offense_name = settings.get("offense_name", offense_ticker)
    defense_ticker = settings["defense_ticker"]
    defense_name = settings.get("defense_name", defense_ticker)

    last_prices = rec_data["last_prices"]
    daily_returns = rec_data.get("daily_returns", {})
    cum_returns = rec_data.get("cum_returns", {})
    current_dd = rec_data["current_drawdown"]
    buy_cutoff = rec_data["buy_cutoff"]
    sell_cutoff = rec_data["sell_cutoff"]
    needed_recovery = rec_data["needed_recovery"]

    market = settings.get("market", "us")
    if market == "kor":
        currency_prefix = ""
        currency_suffix = "원"
        price_fmt = ",.0f"
    else:
        currency_prefix = "$"
        currency_suffix = ""
        price_fmt = ",.2f"

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
        day_ret = daily_returns.get(sym, 0.0)
        c_ret = cum_returns.get(sym, 0.0)

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
        table_lines.append(f"  일간: {day_ret * 100:+.2f}%")

        # 누적 수익률 뒤에 보유 정보 추가
        cum_text = f"  누적: {c_ret * 100:+.2f}%"
        if sym == last_target:
            holding_days = result.get("holding_days", 0)
            if holding_days > 0:
                cum_text += f"({holding_days}거래일째 보유중)"
        else:
            cum_text += "(미보유)"
        table_lines.append(cum_text)

        table_lines.append(f"  현재가: {currency_prefix}{format(price, price_fmt)}{currency_suffix}")
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

    if is_changed:
        print(f"⚠️ 포지션 변경 감지: {prev_target} -> {target_display}")
    else:
        print(f"ℹ️ 포지션 유지: {target_display}")

    # Slack 전송 내용 요약 출력
    market_name = "🇺🇸 미국" if country.lower() == "us" else "🇰🇷 한국"
    header_text = f"{market_name} 스위칭 {'포지션 변경 알림' if is_changed else '정기 보고'}"

    print("\n=== Slack 전송 요약 ===")
    print(f"{header_text} (기준일: {end_date})")
    for line in table_lines:
        if line.strip():
            print(line.strip())
    print(f"🎯 최종 타깃: {target_display}")
    print("========================\n")

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
            is_changed=is_changed,
            holding_days=result.get("holding_days", 0),
        )


if __name__ == "__main__":
    main()
