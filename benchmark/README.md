# Benchmark: 절삭음 데이터셋에서 비지도 이상탐지로 절삭 구간 찾기

`ts_anomaly_app` 의 모형(Darts ForecastingAnomalyModel + 스코어러)을 **공개 산업 데이터**로 검증한 기록이다.
데이터는 Purdue LAMM(Martin Jun 교수 연구실)이 공개한 CNC 밀링 절삭음 데이터셋
[purduelamm/mt_cutting_dataset](https://github.com/purduelamm/mt_cutting_dataset) 을 쓴다.

## 질문

1. 비절삭(idle) 소리만으로 학습한 이상탐지 모형이 절삭 구간을 찾아내는가?
2. 실험실(IMI, Hurco 3축)에서 학습한 모형이 현장(KRPM, Mazak 4축, 주철)으로 옮겨 가는가?
3. 평가를 잘못 설계하면(프레임 무작위 분할) 숫자가 얼마나 부풀려지는가?

## 데이터

| 구분 | 기계 | 센서 | 라벨 |
|---|---|---|---|
| IMI (실험실, 18개 실험) | Hurco VM20i 3축 밀링, AL6061·AISI4140·AISI1018 | 환경 마이크 + 스핀들 청진기 + 베이스 청진기, 48 kHz mono | `cutting.csv` 절삭 구간(초), 전문가가 영상·소리로 표기 |
| KRPM (현장, 1개) | Mazak HCN-6000 4축, 주철 | 동일 | 동일 (주철이라 경계가 불명확하다고 README 가 명시) |

데이터는 저장소에 포함하지 않는다. 받는 방법:

```bash
git clone https://github.com/purduelamm/mt_cutting_dataset  C:/Users/<you>/data/mt_cutting_dataset
```

## 방법

- **특징**: 50 ms 프레임마다 센서당 14개(에너지·영점교차·무게중심·대역폭·평탄도·롤오프·8개 대역 에너지) → 42차원, 20 Hz 시계열. `features.py`, numpy FFT 로 직접 계산.
- **라벨**: 프레임 중앙이 절삭 구간 안이면 1. `dataset.py`.
- **탐지기**: `energy_spindle`(학습 없음, 베이스라인), `flatness_spindle`(채터용 물리 가설 베이스라인), `iforest`, `darts_linearregression_norm`, `darts_linearregression_kmeans`. 뒤의 둘은 앱의 `utils/anomaly_detection.py` 팩토리를 그대로 재사용.
- **임계값**: 학습 '정상' 점수의 99% 분위수. 테스트 라벨은 쓰지 않는다. 대조용 `f1_oracle` 은 테스트 라벨로 최적화한 값.
- **지표**: `metrics.py` 에 직접 구현(AUC-ROC, AUC-PR, P/R/F1, 이벤트 재현율, 분당 오경보). `tests/test_metrics.py` 가 scikit-learn 과 대조.
- **과제 1 (절삭 탐지)**: 정상 = 비절삭. 프로토콜 `grouped_cv`(실험 단위 6겹), `lab_to_industry`, `random_leaky`(대조군).
- **과제 2 (채터 탐지)**: 정상 = 두 작업자가 모두 "정상"으로 표기한 절삭 프레임, 이상 = 모두 "확실한 채터"로 표기한 프레임. 불확실·의견 불일치·비절삭은 제외. 프로토콜 `chatter_grouped_cv`, `chatter_random_leaky`. 라벨은 `IMI/labeling_all_details.xlsx`.
- **보조 분석**: `analyze.py` → `results/analysis.md` (도메인별 에너지 분포, 센서별 전이, 채터 라벨과 절삭 조건의 교차표, 특징별 부호).

## 실행

```bash
python -m pytest -q                          # 지표·특징·라벨 테스트 (43개)
python -m benchmark.run_benchmark --quick    # 빠른 점검 (실험 6개, 학습 없는 탐지기 위주)
python -m benchmark.run_benchmark            # 전체 (18개 실험, 5개 탐지기, 5개 프로토콜, 약 4분)
python -m benchmark.run_benchmark --per-recording-norm --tag _pernorm --protocols grouped_cv lab_to_industry   # 2.5절 추가 실험
python -m benchmark.analyze                  # 보조 분석 → results/analysis.md
```

결과는 `benchmark/results/` 에 `results_long.csv`, `results_summary.md`, `dataset_summary.csv`, `run_meta.json`(패키지 버전·시드·커밋) 으로 남는다.

## 결과와 해석

→ [REPORT.md](REPORT.md) ("이 분석이 틀릴 수 있는 지점" 절 포함)
