"""Slack 알림 전송 유틸리티."""

import os
from typing import Any

from dotenv import load_dotenv

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    WebClient = None
    SlackApiError = None

load_dotenv()


def send_slack_recommendation(
    country: str,
    as_of: str,
    target_display: str,
    table_lines: list[str],
    tuning_meta: dict[str, Any] | None = None,
    is_changed: bool = False,
    holding_days: int = 0,
) -> bool:
    """나스닥 스위칭 추천 결과를 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("TARGET_CHANNEL_ID")

    if not token or not channel_id:
        print(" [SLACK] SLACK_BOT_TOKEN 또는 TARGET_CHANNEL_ID가 설정되지 않았습니다.")
        return False

    if WebClient is None:
        print(" [SLACK] slack-sdk가 설치되지 않았습니다.")
        return False

    client = WebClient(token=token)
    market_name = "🇺🇸 미국" if country.lower() == "us" else "🇰🇷 한국"

    # 이모지 및 타이틀 분기
    if is_changed:
        header_emoji = "🚨"  # 변경 시 더 주목도 높게
        header_text = f"{market_name} 스위칭 포지션 변경 알림"
    else:
        header_emoji = "📊"
        header_text = f"{market_name} 스위칭 정기 보고"

    # 메시지 블록 구성
    blocks = []

    # 1. 헤더
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{header_emoji} {header_text}",
                "emoji": True,
            },
        }
    )

    # 2. 최적 파라미터 정보 (최근 튜닝 결과)
    if tuning_meta:
        # 변경 시에는 파라미터 정보보다는 변경 사실이 중요하므로 간소화하거나 그대로 둠
        # 여기서는 유지하되 위치 조절 가능. 일단 유지.
        tuning_text = (
            f"*🏆 최적 파라미터 (CAGR 기준)*\n"
            f"• 방어 자산: {tuning_meta.get('defense_ticker', 'N/A')}\n"
            f"• 매수 컷: {tuning_meta.get('buy_cutoff', 0):.1f}%\n"
            f"• 매도 컷: {tuning_meta.get('sell_cutoff', 0):.1f}%\n"
            f"• CAGR: {tuning_meta.get('cagr', 0):.2f}%"
        )
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": tuning_text}})
        blocks.append({"type": "divider"})

    # 3. 추천 목록 (상세 테이블)
    if table_lines:
        # 가독성을 위해 간결하게 변환
        clean_lines = []
        for line in table_lines:
            if line.strip().startswith("📌"):
                clean_lines.append(f"*{line.strip()}*")
            elif line.strip():
                clean_lines.append(f"  {line.strip()}")

        table_text = "\n".join(clean_lines)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*=== 추천 목록 ===*\n{table_text}",
                },
            }
        )
        blocks.append({"type": "divider"})

    # 4. 요약 정보
    # holding_days가 0이면 "신규 진입" 또는 "0일째" 등으로 표시하거나, 1일째부터 시작할 수도 있음.
    # runner.py 로직상 당일 포함 카운트되므로 1 이상임.
    holding_text = f"({holding_days}거래일째 보유중)" if holding_days > 0 else "(신규 진입)"
    summary_text = f"ℹ️ *기준일*: {as_of}\n🎯 *최종 타깃*: *{target_display}* {holding_text}"
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_text}})

    # 5. 채널 맨션 (변경이 있을 때만)
    if is_changed:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "<!channel> 포지션이 변경되었습니다! 확인해주세요.",
                },
            }
        )

    try:
        client.chat_postMessage(
            channel=channel_id,
            text=f"[{market_name}] {header_text} ({as_of})",
            blocks=blocks,
        )
        print(f" [SLACK] Slack 알림 전송 완료 (channel={channel_id})")
        return True
    except Exception as e:
        print(f" [SLACK] Slack 전송 실패: {e}")
        return False
