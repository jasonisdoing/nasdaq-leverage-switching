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
    is_warning: bool = False,
    warning_target_display: str | None = None,
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
    if is_warning:
        # 장중 경고 알림
        if is_changed:
            header_emoji = "⚠️"
            header_text = f"{market_name} 장중 포지션 변경 예상 (경고)"
        else:
            header_emoji = "📊"
            header_text = f"{market_name} 장중 스위칭 정기 보고"
    else:
        # 장 마감 직후 최종 확정 알림
        if is_changed:
            header_emoji = "🚨"
            header_text = f"{market_name} 스위칭 포지션 변경 확정! (내일 시초가 매매)"
        else:
            header_emoji = "✅"
            header_text = f"{market_name} 장마감 스위칭 정기 보고"

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
        offense_ticker = tuning_meta.get("offense_ticker", "N/A")
        offense_name = tuning_meta.get("offense_name", "")
        offense_display = f"{offense_ticker}({offense_name})" if offense_name else offense_ticker

        defense_ticker = tuning_meta.get("defense_ticker", "N/A")
        defense_name = tuning_meta.get("defense_name", "")
        defense_display = f"{defense_ticker}({defense_name})" if defense_name else defense_ticker

        tuning_text = (
            f"*🏆 최적 파라미터 (CAGR 기준)*\n"
            f"• 공격 자산: {offense_display}\n"
            f"• 방어 자산: {defense_display}\n"
            f"• 매수 컷: {tuning_meta.get('buy_cutoff', 0):.1f}%\n"
            f"• 매도 컷: {tuning_meta.get('sell_cutoff', 0):.1f}%\n"
            f"• CAGR: {tuning_meta.get('cagr', 0) * 100:.2f}%"
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
    summary_text = f"ℹ️ *기준일*: {as_of}"

    if is_warning and warning_target_display:
        # 경고 모드: 현재 보유 + 전환 가능성 안내
        summary_text += f"\n💼 *현재 보유*: *{target_display}*"
        summary_text += (
            f"\n\n*⚠️ 장중 경고*: 이대로 장 마감 시 "
            f"*{warning_target_display}*(으)로 전환될 수 있습니다. "
            "장 마감 후 최종 확정 알림을 기다려주세요."
        )
    elif is_warning:
        # 경고 모드이지만 변경 없음
        summary_text += f"\n🎯 *현재 보유*: *{target_display}*"
        summary_text += "\n\n*ℹ️ 안내*: 장 마감 시까지 변동될 수 있습니다. 장 마감 후 최종 확정 알림을 기다려주세요."
    elif is_changed:
        # 확정 모드에서 변경됨
        summary_text += f"\n🎯 *최종 타깃*: *{target_display}*"
        summary_text += (
            "\n\n*🔔 실행 안내*: 오늘 종가 기준으로 시그널이 확정되었습니다. "
            "내일(다음 거래일) 아침 시초가에 해당 종목을 매매하세요."
        )
    else:
        # 확정 모드에서 변경 없음
        summary_text += f"\n🎯 *최종 타깃*: *{target_display}*"

    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_text}})

    # 5. 채널 맨션 (확정 알림에서 변경이 있을 때만, 경고 모드에서는 멘션 안 함)
    if is_changed and not is_warning:
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
