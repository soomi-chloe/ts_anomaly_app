"""Page 3 — 이상탐지 설정 + 실행."""

from __future__ import annotations

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import preprocess_pipeline, split_train_test
from utils.anomaly_detection import run_anomaly_detection


st.set_page_config(page_title="이상탐지 설정", page_icon="⚙️", layout="wide")
st.title("⚙️ 이상탐지 설정 & 실행")


ts = st.session_state.get("ts")
if ts is None:
    st.warning("먼저 **`📁 데이터 업로드`** 페이지에서 TimeSeries를 생성해주세요.")
    st.stop()


st.markdown(f"**현재 데이터**: 길이 `{len(ts)}`, 변수 `{ts.n_components}`개, freq `{ts.freq_str}`")


# ---------------------------------------------------------------------------
# 1. 학습/테스트 분할
# ---------------------------------------------------------------------------

st.subheader("✂️ 학습/테스트 분할")
train_ratio = st.slider("학습 데이터 비율", 0.3, 0.95, 0.7, 0.05)
train_size = int(len(ts) * train_ratio)
st.caption(f"학습: {train_size:,} 시점 / 테스트: {len(ts) - train_size:,} 시점")


# ---------------------------------------------------------------------------
# 2. 전처리
# ---------------------------------------------------------------------------

st.subheader("🧪 전처리 옵션")
c1, c2 = st.columns(2)
with c1:
    scaler_name = st.selectbox("스케일링", ["MinMax", "Standard", "None"], index=0)
with c2:
    diff_order = st.selectbox("차분 차수", [0, 1, 2], index=0)


# ---------------------------------------------------------------------------
# 3. 다변량 처리 방식
# ---------------------------------------------------------------------------

st.subheader("🔀 다변량 처리 방식")
mode_label = st.radio(
    "전략 선택",
    options=["Joint (동시 고려) — 기본", "Per-Variable (변수별 독립)", "Reproduce (수업 예제 재현)"],
    horizontal=False,
    help="다변량을 한 번에 학습할지(Joint), 변수별로 따로(Per-Variable), 또는 수업 예제 재현 방식인지 선택합니다.",
)
mode_map = {
    "Joint (동시 고려) — 기본": "joint",
    "Per-Variable (변수별 독립)": "per_variable",
    "Reproduce (수업 예제 재현)": "reproduce",
}
multivariate_mode = mode_map[mode_label]


# ---------------------------------------------------------------------------
# 4. 모형 / Scorer
# ---------------------------------------------------------------------------

st.subheader("🤖 예측 모형 + 이상 점수 산출기")


# statsforecast 미설치 / darts 측 호환 불가 환경에서는 AutoARIMA를 안전하게 제외
def _autoarima_available() -> bool:
    """Darts AutoARIMA를 실제로 인스턴스화할 수 있는지 검증.

    1) `statsforecast` 패키지 import 가능 여부
    2) `darts.models.AutoARIMA`가 NotImportedModule stub이 아닌지 (`usable != False`)
    둘 다 통과해야 사용 가능으로 판정한다.
    """
    try:
        import statsforecast  # noqa: F401
    except Exception:
        return False
    try:
        from darts.models import AutoARIMA as _DartsAutoARIMA
    except Exception:
        return False
    # darts는 import 실패 시 클래스 속성 `usable=False`인 stub을 등록한다.
    if getattr(_DartsAutoARIMA, "usable", True) is False:
        return False
    return True


_HAS_SF = _autoarima_available()

_MODEL_OPTIONS = [
    "LinearRegression",
    "Ridge",
    "XGBoost",
    "LightGBM",
    "ExponentialSmoothing",
]
if _HAS_SF:
    _MODEL_OPTIONS.append("AutoARIMA")
else:
    st.info(
        "ℹ️ 현재 환경에서 **AutoARIMA** 를 사용할 수 없어 옵션에서 제외되었습니다. "
        "`statsforecast` 패키지가 미설치이거나, Darts와 호환되지 않을 수 있습니다. "
        "사용하려면 `pip install -U statsforecast darts` 또는 `requirements.txt` 에 명시 후 재배포하세요."
    )

c3, c4 = st.columns(2)
with c3:
    model_name = st.selectbox(
        "예측 모형",
        options=_MODEL_OPTIONS,
        index=0,
        help="회귀/트리 계열은 다변량을 함께 학습하고 lag 기반 lookback을 사용합니다.",
    )
with c4:
    scorer_names = st.multiselect(
        "이상 점수 산출기 (복수 선택)",
        options=["NormScorer", "KMeansScorer", "WassersteinScorer"],
        default=["NormScorer"],
    )

with st.expander("🔧 고급 파라미터", expanded=False):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        lags = st.number_input("lags / input_chunk_length", min_value=1, max_value=200, value=24)
    with cc2:
        output_chunk_length = st.number_input("output_chunk_length", min_value=1, max_value=24, value=1)
    with cc3:
        window = st.number_input("Scorer window 크기", min_value=1, max_value=200, value=10)
    k = st.number_input("KMeansScorer k", min_value=2, max_value=10, value=2)


# ---------------------------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🚀 실행")

if not scorer_names:
    st.warning("Scorer를 1개 이상 선택해주세요.")
    st.stop()

run = st.button("🚀 이상탐지 실행", type="primary", use_container_width=True)

if run:
    progress = st.progress(0, text="전처리 중...")
    try:
        # 전처리
        ts_train, ts_test, scaler = preprocess_pipeline(
            ts, train_ratio=train_ratio, scaler_name=scaler_name, diff_order=diff_order,
        )
        progress.progress(30, text="모형 학습 중...")

        params = {
            "lags": int(lags),
            "output_chunk_length": int(output_chunk_length),
            "window": int(window),
            "k": int(k),
        }

        t0 = time.time()
        with st.spinner("이상탐지 파이프라인 실행 중..."):
            result = run_anomaly_detection(
                ts_train=ts_train,
                ts_test=ts_test,
                model_name=model_name,
                scorer_names=scorer_names,
                params=params,
                multivariate_mode=multivariate_mode,
            )
        elapsed = time.time() - t0
        progress.progress(90, text="결과 저장 중...")

        # 결과 저장
        st.session_state["anomaly_scores"] = result.get("scores")
        st.session_state["anomaly_model"] = result.get("anomaly_model")
        st.session_state["ts_train"] = ts_train
        st.session_state["ts_test"] = ts_test
        st.session_state["scaler"] = scaler
        st.session_state["multivariate_mode"] = multivariate_mode
        st.session_state["per_variable_scores"] = result.get("per_variable_scores")

        cfg = st.session_state.get("config", {}) or {}
        cfg.update({
            "train_ratio": train_ratio,
            "scaler_name": scaler_name,
            "diff_order": diff_order,
            "model_name": model_name,
            "scorer_names": scorer_names,
            "params": params,
            "multivariate_mode": multivariate_mode,
        })
        st.session_state["config"] = cfg

        progress.progress(100, text="완료")
        st.success(f"✅ 이상탐지 실행 완료 — {elapsed:.2f}초 소요")
        st.balloons()
    except Exception as e:
        progress.empty()
        st.error(f"실행 실패: {e}")
        st.exception(e)
        st.stop()

    st.divider()
    st.info("➡️ **`📊 결과 대시보드`** 페이지에서 시각화/평가지표를 확인하세요.")
