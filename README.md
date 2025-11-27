# QQQM/QLD/TQQQ 레버리지 전환 기본 전략

기초적인 추세·변동성 기반 전환 전략 코드입니다. 시그널은 QQQ(또는 QQQM) 종가로 계산하고, 조건에 따라 QQQM/QLD/TQQQ 중 하나로 전환합니다. 자유롭게 파라미터를 바꾸며 실험하세요.

## 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 실행
```bash
python strategy.py
```

출력: 시작/종료일, 최종 자산, CAGR, 변동성, 샤프, 최대 낙폭, 일간 승률, 마지막 매수 대상.

## 주요 설정
- `config.py`의 `SIMULATION_START_DATE`에서 백테스트 시작일(YYYY-MM-DD) 변경.
- `strategy.py`의 `StrategyParams`에서 MA 길이, 변동성 기준, 낙폭 방어선, 초기자본 등을 수정.

## 전략 개요
- 50/200일 이동평균으로 추세 측정, 20일 연율화 변동성으로 공격/수비 결정.
- 조건
  - 큰 낙폭(`drawdown >= 20%`) 시 방어: `QQQM`
  - 상승 추세 + 낮은 변동성: `TQQQ`
  - 상승 추세(변동성 높음): `QLD`
  - 그 외: `QQQM`
- 일일 리밸런싱(전액 전환) 단순화 버전. 거래비용/세금 미반영.

## 유의사항
- QQQM 데이터는 2020년 이후만 존재합니다. 시작일을 더 이르게 잡으면 사용 가능한 데이터 구간만 자동으로 사용합니다.
- yfinance 다운로드가 필요하므로 인터넷 연결이 필요합니다.
- 교육용 예제이며 실거래 전 반드시 추가 검증과 리스크 관리 규칙을 추가하세요.
