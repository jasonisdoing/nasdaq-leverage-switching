# QQQM/QLD/TQQQ 레버리지 전환 백테스트

`settings.json`에 모든 파라미터를 명시한 뒤 `backtest.py`를 실행해 로그를 생성하는 구조입니다. 기본값/자동 보정은 없으며, 필수 키가 없으면 즉시 오류를 냅니다.

## 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 설정 (필수 키)
- `settings.json`
  - `signal_symbol`: 시그널 계산 티커(예: QQQM)
  - `trade_symbols`: 실제 매매 대상 티커 배열(예: QQQM, QLD, TQQQ)
  - `ma_short`, `ma_long`: 이동평균 단기/장기 길이
  - `vol_lookback`: 변동성 계산 기간
  - `vol_cutoff`: 변동성 임계값
  - `drawdown_cutoff`: 방어 진입 낙폭 기준
  - `months_range`: 오늘 기준 과거 n개월 데이터를 백테스트 구간으로 사용
  - `benchmarks`: 벤치마크 티커 배열
- `config.py`
  - `INITIAL_CAPITAL_KRW`: 초기 자본(원화). 첫 거래일 환율로 USD 환전 후 전체 기간 USD로 계산, 마지막에 KRW 환산.

## 실행
```bash
python backtest.py
```

## 출력
- 파일: `zresults/backtest_YYYY-MM-DD.log`
  - 섹션 2: 일자별 성과(총자산, 일간/누적 수익률, 테이블 상태 BUY/HOLD/SELL/WAIT, 보유일, 가격, 수량, 금액, 손익, 비중 등)
  - 섹션 7: 종목별 성과 요약(기여도 USD, 손익 USD/KRW, 노출일수, 거래횟수, 승률)
  - 섹션 8: 백테스트 결과 요약(기간, 초기/최종 자산, 기간수익률, 벤치마크 수익률/CAGR, 전략 CAGR, MDD)
- 터미널: 주요 지표 요약 출력

## 전략 개요
- 50/200일 이동평균으로 추세, 20일 변동성으로 공격/수비 결정.
- 조건: 큰 낙폭 시 QQQM, 상승+저변동성 시 TQQQ, 상승+고변동성 시 QLD, 그 외 QQQM.
- 일일 전액 전환, 거래비용/세금 미반영. 상태는 BUY/HOLD/SELL/WAIT로 기록.

## 유의사항
- QQQM 상장 이전 데이터가 없으므로 실제 시작일은 데이터/워밍업 길이에 따라 뒤로 밀릴 수 있습니다.
- yfinance 다운로드가 필요하므로 실행 시 인터넷 연결이 필요합니다(테스트는 사용자가 직접 수행).
- 자동 기본값이 없으므로 `settings.json`의 필수 키를 모두 채워야 합니다.
