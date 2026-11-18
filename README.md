# Multivariate Time-Series Anomaly Detection: Streamlit app + industrial benchmark

**English summary.** Upload a multivariate time-series CSV and the app runs EDA → preprocessing →
forecasting-based anomaly detection (Darts `ForecastingAnomalyModel` with Norm/KMeans/Wasserstein scorers)
→ an evaluation dashboard (AUC-ROC, AUC-PR, F1, confusion matrix, or unsupervised score statistics when no labels exist).

The app was built for a smart-manufacturing / time-series course. To find out whether its detectors hold up
on real industrial data, `benchmark/` evaluates the same model families on the public CNC milling sound dataset
released by Purdue's LAMM lab ([purduelamm/mt_cutting_dataset](https://github.com/purduelamm/mt_cutting_dataset)):
lab-vs-industry transfer, experiment-grouped cross-validation, a deliberately leaky split as a control, and a
chatter-detection task. Every metric is implemented from its definition in `benchmark/metrics.py` and checked
against scikit-learn in `tests/`. Results and a "where this analysis could be wrong" section are in
[benchmark/REPORT.md](benchmark/REPORT.md).

```bash
pip install -r requirements-dev.txt
python -m pytest -q                       # 43 tests: metrics vs scikit-learn, feature extraction, label conversion
python -m benchmark.run_benchmark --quick # needs the dataset, see benchmark/README.md
```

---

# 🚨 시계열 이상탐지 자동화 웹앱

다변량 시계열 CSV를 업로드하면 **EDA → 전처리 → Darts 기반 이상탐지 → 평가 대시보드**까지
자동으로 수행하는 Streamlit 웹 애플리케이션.

> 과목: 스마트제조 / 시계열 분석
> 핵심 라이브러리: Streamlit, Darts, pandas, plotly, scikit-learn
> 검증: `benchmark/` 에서 Purdue LAMM 공개 절삭음 데이터로 벤치마크, `tests/` 에서 지표 구현을 scikit-learn 과 대조

---

## 📂 디렉토리 구조

```
ts_anomaly_app/
├── app.py                          # 메인 엔트리포인트
├── pages/
│   ├── 1_📁_데이터_업로드.py
│   ├── 2_🔍_탐색적_분석.py
│   ├── 3_⚙️_이상탐지_설정.py
│   └── 4_📊_결과_대시보드.py
├── utils/
│   ├── data_loader.py              # CSV 로드 + TimeSeries 변환
│   ├── eda.py                      # EDA 시각화/검정
│   ├── preprocessing.py            # 스케일링/차분/분할
│   ├── anomaly_detection.py        # ForecastingAnomalyModel 파이프라인
│   └── evaluation.py               # 지도/비지도 평가지표
├── benchmark/                      # 공개 산업 데이터 벤치마크 (README.md, REPORT.md 참고)
│   ├── metrics.py                  # 지표를 정의에서 직접 구현 (AUC-ROC, AUC-PR, P/R/F1, 이벤트 지표)
│   ├── features.py                 # WAV → 50 ms 프레임 특징 시계열 (numpy FFT)
│   ├── dataset.py                  # mt_cutting_dataset 로더, 절삭·채터 라벨
│   └── run_benchmark.py            # 프로토콜 실행 → results/
├── tests/                          # pytest: 지표 vs scikit-learn, 특징, 라벨 변환
├── sample_data/
│   ├── _generate_sample.py
│   ├── multivariate_sensor.csv
│   ├── multivariate_sensor_labels.csv
│   └── multivariate_sensor_with_label.csv
├── .streamlit/config.toml
├── requirements.txt
├── runtime.txt                     # Streamlit Cloud용 Python 버전
└── README.md
```

## 🏃 실행 방법

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 → 사이드바에서 페이지 1~4를 순서대로 진행.

## 🧪 샘플 데이터

`sample_data/` 폴더에 시간 단위 30일치(720시점) 3변수 센서 시뮬레이션이 포함됩니다.
- `multivariate_sensor.csv`: 시계열 (timestamp + temperature + vibration + pressure)
- `multivariate_sensor_labels.csv`: 라벨 (timestamp + label, 양성 88/720, 12.2%)
- `multivariate_sensor_with_label.csv`: 위 두 파일을 한 CSV로 결합 (컬럼 지정 시나리오용)

데이터에는 5가지 종류의 인위적 이상이 삽입되어 있습니다.
1. 단발성 스파이크 (temperature, +8℃)
2. 단발성 음의 스파이크 (pressure, -3.5 bar)
3. 점진적 드리프트 (vibration)
4. 분산 증가 (전 채널, 짧은 구간)
5. 레벨 시프트 (temperature, +6℃)
6. 다변량 동시 이상

샘플을 재생성하려면:
```bash
python sample_data/_generate_sample.py
```

## 🔍 페이지 흐름

| 페이지 | 기능 |
|---|---|
| 📁 데이터 업로드 | CSV 업로드(파일 변경 시 세션 초기화) / 시간 인덱스·변수 선택 / (선택) 라벨 등록 |
| 🔍 탐색적 분석 | 기초 통계 / 시계열 시각화 / ADF·Ljung-Box / 상관행렬 / 결측치 위치 |
| ⚙️ 이상탐지 설정 | 학습-테스트 분할 / 스케일러·차분 / 다변량 처리 방식 / 모형 + Scorer / 실행 |
| 📊 결과 대시보드 | 이상 구간 시각화 / 임계값 슬라이더 / 점수 분포 / 지도(AUC-ROC/PR/F1/CM) ↔ 비지도 지표 자동 전환 / Scorer 비교 / CSV 다운로드 |

## 🤖 지원 모형 & Scorer

**예측 모형**: LinearRegression / Ridge / XGBoost / LightGBM / ExponentialSmoothing / AutoARIMA
- 회귀·트리 모형: `ForecastingAnomalyModel`에 직접 투입 (다변량 지원)
- 통계 모형(ES/AutoARIMA): LocalForecastingModel → 단변량 잔차 기반 점수로 자동 처리

**Scorer**: NormScorer / KMeansScorer / WassersteinScorer (복수 선택 가능)

**다변량 처리 방식**:
- `Joint (기본)` — 다변량 TimeSeries 단일 학습 + scorer 다변량 적용
- `Per-Variable` — 변수별 독립 학습 후 평균 통합
- `Reproduce` — `final.md`/`mid.ipynb` 방식 재현 (해당 데이터셋 한정)

## 📈 평가지표

라벨이 있으면 자동으로 **지도 평가**로 전환:
- AUC-ROC, AUC-PR, Precision, Recall, F1, Confusion Matrix
- ROC/PR 곡선 시각화

라벨이 없으면 **비지도 평가**:
- 이상 점수 분포(히스토그램+박스플롯)
- 탐지 이상 수/비율, 평균/표준편차/분위수
- 예측 잔차 통계

## ☁️ 배포 (Streamlit Community Cloud)

1. 이 디렉토리를 GitHub 리포지토리로 푸시
2. https://streamlit.io/cloud 에서 새 앱 생성 → `app.py` 지정
3. `runtime.txt` (Python 3.10) + `requirements.txt` 자동 감지
4. `.streamlit/config.toml`의 `server.maxUploadSize=200` 적용

> Cloud는 CPU 전용이므로 통계/회귀/트리 모형으로 운영 권장.

## ⚠️ 주의

- Windows + Streamlit의 한글 페이지 파일명은 정상 동작하지만, OS·Python 인코딩에 따라
  Streamlit Cloud에서는 파일 이름의 이모지가 메뉴에 표시될 수도 있습니다.
- darts 0.44+ 환경에서 `TimeSeries.to_dataframe()` API를 사용합니다 (코드에 양 버전 fallback 처리됨).
