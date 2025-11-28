# QQQM/QLD/TQQQ 레버리지 전환 백테스트

`settings.json`에 모든 파라미터를 명시한 뒤 `backtest.py`(백테스트)와 `tune.py`(튜닝)를 실행하는 구조입니다. 기본값/자동 보정은 없으며, 필수 키가 없으면 즉시 오류를 냅니다. 튜닝 결과를 바탕으로 `settings.json`을 수동으로 업데이트해 사용할 수 있습니다.

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
  - `BACKTEST_SLIPPAGE`: 슬리피지 설정(%) 딕셔너리. 매수는 시초가에 `buy_pct`만큼 가산, 매도는 `sell_pct`만큼 할인해 체결가를 추정.

## 실행
```bash
python backtest.py
python tune.py      # 병렬로 튜닝 수행, 결과 로그 확인 후 settings.json을 직접 갱신
```

## 출력
- 파일: `zresults/backtest_YYYY-MM-DD.log`
  - 섹션 2: 일자별 성과(총자산, 일간/누적 수익률, 테이블 상태 BUY/HOLD/SELL/WAIT, 보유일, 가격, 수량, 금액, 손익, 비중 등)
  - 섹션 7: 종목별 성과 요약(기여도 USD, 손익 USD/KRW, 노출일수, 거래횟수, 승률)
  - 섹션 8: 백테스트 결과 요약(기간, 초기/최종 자산, 기간수익률, 벤치마크 수익률/CAGR, 전략 CAGR, MDD)
- 터미널: 주요 지표 요약 출력
- 파일: `zresults/tune_YYYY-MM-DD.log`
  - 진행률에 따라 상위 조합을 중간 기록, 완료 시 상위 결과를 CAGR 기준으로 정렬해 저장

## 전략 개요
- 나스닥 계열 ETF(1배/2배/3배) 사이를 전액 전환합니다.
- 추세(이동평균)와 변동성 지표로 공격/수비 전환:
  - 큰 낙폭 구간: 방어적으로 QQQM
  - 상승 + 낮은 변동성: 공격적으로 TQQQ
  - 상승 + 높은 변동성: 중간 레버리지 QLD
  - 그 외: 방어적으로 QQQM
- 체결 가정: 매매 신호 다음날 시초가로 전액 전환하며, 매수는 슬리피지만큼 높은 가격(+buy_pct), 매도는 슬리피지만큼 낮은 가격(-sell_pct)으로 보수적으로 체결가를 추정합니다. 거래비용/세금은 별도 반영하지 않습니다. 상태는 BUY/HOLD/SELL/WAIT로 기록.

## 유의사항
- QQQM 상장 이전 데이터가 없으므로 실제 시작일은 데이터/워밍업 길이에 따라 뒤로 밀릴 수 있습니다.
- yfinance 다운로드가 필요하므로 실행 시 인터넷 연결이 필요합니다(테스트는 사용자가 직접 수행).
- 자동 기본값이 없으므로 `settings.json`의 필수 키를 모두 채워야 합니다.
- 튜닝은 yfinance 호출을 많이 하므로, rate limit이 걸리면 안내 메시지와 함께 중단되며 시간이 지난 뒤 재시도해야 합니다.
