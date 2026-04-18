#!/bin/bash
# leverage-switching crontab 을 "다른 앱 크론을 보존하면서" 병합 설치합니다.
#
# 동작:
#   1. 현재 사용자 crontab 을 읽어온다
#   2. 기존 leverage-switching 블록(마커 사이) 을 제거
#   3. 파일 끝에 최신 leverage-switching 블록을 append
#   4. crontab - 로 교체 설치
#
# 여러 번 실행해도 안전 (idempotent). momentum-etf 등 다른 앱 항목은 건드리지 않습니다.
#
# 사용:
#   bash /home/ubuntu/apps/leverage-switching/infra/cron/install.sh
#
# 제거:
#   bash /home/ubuntu/apps/leverage-switching/infra/cron/install.sh --uninstall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_CRON="$SCRIPT_DIR/crontab"
MARKER_BEGIN="# >>> leverage-switching >>>"
MARKER_END="# <<< leverage-switching <<<"

current="$(crontab -l 2>/dev/null || true)"

# 기존 블록 제거 (마커 쌍 사이 모두 삭제)
filtered="$(printf '%s\n' "$current" | awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
    $0 == b { skip=1; next }
    $0 == e { skip=0; next }
    skip != 1 { print }
')"

if [ "${1:-}" = "--uninstall" ]; then
    printf '%s\n' "$filtered" | crontab -
    echo "[install.sh] leverage-switching 블록 제거 완료"
    exit 0
fi

if [ ! -f "$APP_CRON" ]; then
    echo "[install.sh] crontab 파일을 찾을 수 없음: $APP_CRON" >&2
    exit 1
fi

{
    # 기존 내용 (leverage-switching 블록 제외) 출력
    printf '%s\n' "$filtered" | sed '/./,$!d'  # 앞쪽 공백줄 정리
    echo
    echo "$MARKER_BEGIN"
    cat "$APP_CRON"
    echo "$MARKER_END"
} | crontab -

echo "[install.sh] leverage-switching crontab 설치 완료"
echo "--- 현재 crontab ---"
crontab -l
