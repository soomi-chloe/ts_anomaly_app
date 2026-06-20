"""시계열 이상탐지 자동화 웹앱 — 메인 엔트리포인트.

구조:
- 사이드바: 전역 정보 + 현재 단계
- 메인: 프로젝트 소개 + 사용 흐름 안내
- 페이지: pages/ 하위 4개
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="시계열 이상탐지 자동화",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 세션 상태 기본값
# ---------------------------------------------------------------------------

DEFAULT_KEYS = {
    "file_hash": None,
    "df": None,
    "ts": None,
    "ts_train": None,
    "ts_test": None,
    "anomaly_scores": None,
    "anomaly_model": None,
    "config": {},
    "labels": None,
    "multivariate_mode": "joint",
    "scaler": None,
    "ts_test_orig": None,
}

for k, v in DEFAULT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# 사이드바 — 전역 상태
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚨 이상탐지 자동화")
    st.caption("Darts 기반 다변량 시계열 이상탐지 워크플로우")
    st.divider()

    st.subheader("📌 현재 상태")
    df_loaded = st.session_state.get("df") is not None
    ts_built = st.session_state.get("ts") is not None
    scored = st.session_state.get("anomaly_scores") is not None
    labels_loaded = st.session_state.get("labels") is not None

    st.markdown(
        f"""
        - 데이터 업로드: {'✅' if df_loaded else '⬜'}
        - TimeSeries 구성: {'✅' if ts_built else '⬜'}
        - 라벨 데이터: {'✅' if labels_loaded else '⬜ (선택)'}
        - 이상탐지 실행: {'✅' if scored else '⬜'}
        """
    )

    if df_loaded:
        df = st.session_state["df"]
        st.metric("행 수", f"{len(df):,}")
        st.metric("열 수", f"{df.shape[1]}")

    st.divider()
    st.caption("좌측 페이지 메뉴에서 단계별로 진행하세요.")


# ---------------------------------------------------------------------------
# 메인 콘텐츠
# ---------------------------------------------------------------------------

st.title("🚨 시계열 이상탐지 자동화 웹앱")
st.markdown(
    """
    이 앱은 임의의 **다변량 시계열 CSV**를 업로드받아,
    자동으로 EDA → 전처리 → Darts 기반 이상탐지 → 평가 대시보드까지 수행합니다.
    """
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(
        """
        ### 📁 1. 데이터 업로드
        - CSV 파일 업로드
        - 시간 인덱스 자동 감지
        - (선택) 라벨 CSV 업로드
        """
    )
with col2:
    st.info(
        """
        ### 🔍 2. 탐색적 분석
        - 기초 통계, 결측치
        - 시계열 시각화
        - 정상성 검정, 상관관계
        """
    )
with col3:
    st.info(
        """
        ### ⚙️ 3. 이상탐지 설정
        - 모형/Scorer 선택
        - 다변량 처리 방식
        - 학습 + 점수 계산
        """
    )
with col4:
    st.info(
        """
        ### 📊 4. 결과 대시보드
        - 이상 구간 시각화
        - 임계값 조정
        - 지도/비지도 평가지표
        """
    )

st.divider()

st.subheader("🚀 빠른 시작")
st.markdown(
    """
    1. 좌측 사이드바의 **`📁 데이터 업로드`** 페이지로 이동합니다.
    2. CSV를 업로드하거나 "샘플 데이터 사용" 버튼을 누릅니다.
    3. 시간 인덱스/분석 변수를 확인하고, (선택) 라벨 데이터를 등록합니다.
    4. **`🔍 탐색적 분석`** → **`⚙️ 이상탐지 설정`** 으로 순차 진행 후 실행하세요.
    5. **`📊 결과 대시보드`** 에서 시각화와 평가지표를 확인합니다.
    """
)

with st.expander("📦 기술 스택 / 아키텍처", expanded=False):
    st.markdown(
        """
        - **Framework**: Streamlit (멀티페이지)
        - **Anomaly Detection**: Darts `ForecastingAnomalyModel`, `NormScorer` / `KMeansScorer` / `WassersteinScorer`
        - **예측 모형**: LinearRegression / Ridge / XGBoost / LightGBM / ExponentialSmoothing / AutoARIMA
        - **시각화**: Plotly (Dark 모드)
        - **평가**: scikit-learn (AUC-ROC / AUC-PR / F1 / Confusion Matrix)
        """
    )

st.divider()
st.caption("© 시계열 이상탐지 자동화 — 시계열 분석 과제")
