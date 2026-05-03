#!/usr/bin/env python3
"""VM cron 배치 래퍼 (leverage-switching 전용).

사용법:
    python infra/cron/run_batch.py <job_name> <command...>

예시:
    python infra/cron/run_batch.py leverage_kor /path/to/.venv/bin/python recommend.py kor --slack

동작:
    1) subprocess 로 <command> 를 실행
    2) 종료 코드/소요시간/마지막 로그 꼬리(15줄)를 슬랙 웹훅으로 전송
       - 실패 시: 항상 알림
       - 성공 시: SUCCESS_NOTIFICATION_DISABLED_JOBS 에 포함된 job 은 알림 생략
    3) 프로세스 종료 코드를 그대로 반환

환경변수:
    LOGS_SLACK_WEBHOOK  - 배치 운영 로그용 Slack incoming webhook URL
    APP_TYPE            - (선택) 알림 라벨. 미지정 시 "leverage-switching"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# .env 로드 (python-dotenv 없이도 동작하도록)
# python-dotenv 와 동일하게 값 주위의 동일 따옴표(" 또는 ')는 벗겨낸다.
# 따옴표가 남아 있으면 하위 프로세스(slack-sdk, urllib 등)가
# invalid_auth / "unknown url type" 로 실패한다.
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip()
        _v = _v.strip()
        if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in ('"', "'"):
            _v = _v[1:-1]
        os.environ.setdefault(_k, _v)

MAX_TAIL_LINES = 15
MAX_TAIL_CHARS = 1500

LOCK_DIR = PROJECT_ROOT / "logs" / "cron"

# 성공 시 알림을 생략할 job 목록 (정기 성공 알림이 과하면 추가)
SUCCESS_NOTIFICATION_DISABLED_JOBS: set[str] = set()


def _acquire_lock(job_name: str) -> Path:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{job_name}.lock"
    lock_path.write_text(
        f"pid={os.getpid()}\nstarted={datetime.now(KST).isoformat()}\n",
        encoding="utf-8",
    )
    return lock_path


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover
        print(f"[run_batch] 락 해제 실패: {exc}", file=sys.stderr)


def _notify(text: str) -> None:
    """Slack incoming webhook 으로 전송. 실패해도 배치 결과에 영향 없도록 방어."""
    webhook = os.environ.get("LOGS_SLACK_WEBHOOK", "").strip()
    if not webhook:
        print("[run_batch] LOGS_SLACK_WEBHOOK 미설정 → 알림 생략", file=sys.stderr)
        return
    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib_request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
    except urllib_error.URLError as exc:
        print(f"[run_batch] 슬랙 전송 실패: {exc}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover
        print(f"[run_batch] 슬랙 전송 예외: {exc}", file=sys.stderr)


def _format_tail(stdout: str, stderr: str) -> str:
    combined = (stdout or "") + (stderr or "")
    lines = combined.strip().splitlines()
    if not lines:
        return "(출력 없음)"
    tail = "\n".join(lines[-MAX_TAIL_LINES:])
    if len(tail) > MAX_TAIL_CHARS:
        tail = "…(생략)…\n" + tail[-MAX_TAIL_CHARS:]
    return tail


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: run_batch.py <job_name> <command...>", file=sys.stderr)
        return 2

    job_name = argv[1]
    command = argv[2:]

    started_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    started_monotonic = time.monotonic()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[run_batch] START job={job_name} cmd={' '.join(command)} at={started_at}")

    lock_path = _acquire_lock(job_name)
    app_label = os.environ.get("APP_TYPE", "leverage-switching").strip() or "leverage-switching"

    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        _release_lock(lock_path)
        elapsed = time.monotonic() - started_monotonic
        _notify(
            f"❌ *[{app_label}] 배치 실행 불가*: `{job_name}`\n"
            f"• 시작: {started_at}\n"
            f"• 소요: {elapsed:.1f}s\n"
            f"• 에러: `{exc}`"
        )
        print(f"[run_batch] FAIL {exc}", file=sys.stderr)
        return 127
    except Exception as exc:
        _release_lock(lock_path)
        elapsed = time.monotonic() - started_monotonic
        _notify(
            f"❌ *[{app_label}] 배치 예외*: `{job_name}`\n• 시작: {started_at}\n• 소요: {elapsed:.1f}s\n• 에러: `{exc}`"
        )
        print(f"[run_batch] EXCEPTION {exc}", file=sys.stderr)
        return 1
    finally:
        _release_lock(lock_path)

    elapsed = time.monotonic() - started_monotonic
    exit_code = result.returncode
    success = exit_code == 0

    # 원본 출력은 그대로 흘려서 cron 로그파일에도 남김
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    emoji = "✅" if success else "❌"
    status = "성공" if success else "실패"
    tail = _format_tail(result.stdout, result.stderr)

    should_notify = (not success) or (success and job_name not in SUCCESS_NOTIFICATION_DISABLED_JOBS)
    if should_notify:
        _notify(
            f"{emoji} *[{app_label}] 배치 {status}*: `{job_name}`\n"
            f"• 시작: {started_at}\n"
            f"• 소요: {elapsed:.1f}s\n"
            f"• exit: {exit_code}\n"
            f"```\n{tail}\n```"
        )

    print(f"[run_batch] END job={job_name} status={status} exit={exit_code} elapsed={elapsed:.1f}s")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
