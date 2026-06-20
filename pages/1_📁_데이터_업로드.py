"""Page 1 — 데이터 업로드."""

from __future__ import annotations

import sys
import os
import pandas as pd
import streamlit as st

# utils 임포트 경로
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    compute_file_hash, read_csv_safely,
    detect_time_column, detect_numeric_columns,
    infer_frequency, to_darts_timeseries, load_labels,
)


st.set_page_config(page_title="데이터 업로드", page_icon="📁", layout="wide")
st.title("📁 데이터 업로드")


# ---------------------------------------------------------------------------
# 샘플 데이터 사용
# ---------------------------------------------------------------------------

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
SAMPLE_CSV = os.path.join(SAMPLE_DIR, "multivariate_sensor.csv")
SAMPLE_LABELS = os.path.join(SAMPLE_DIR, "multivariate_sensor_labels.csv")


def reset_session(new_hash: str, df: pd.DataFrame, source_name: str):
    """파일이 바뀌면 분석 관련 세션 키 초기화."""
    keys_to_reset = [
        "ts", "ts_train", "ts_test", "anomaly_scores", "anomaly_model",
        "config", "labels", "scaler", "ts_test_orig",
    ]
    for k in keys_to_reset:
        st.session_state[k] = None if k != "config" else {}
    st.session_state["file_hash"] = new_hash
    st.session_state["df"] = df
    st.session_state["source_name"] = source_name


# ---------------------------------------------------------------------------
# 업로드 UI
# ---------------------------------------------------------------------------

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        "다변량 시계열 CSV 파일을 업로드하세요",
        type=["csv"],
        help="시간 인덱스 컬럼과 1개 이상의 숫자 컬럼이 포함되어야 합니다.",
    )

with col_sample:
    st.write("")
    st.write("")
    use_sample = st.button("🎯 샘플 데이터 사용", use_container_width=True)


# 새 파일 업로드
if uploaded is not None:
    raw = uploaded.getvalue()
    new_hash = compute_file_hash(raw)
    if st.session_state.get("file_hash") != new_hash:
        try:
            df = read_csv_safely(uploaded)
        except Exception as e:
            st.error(f"CSV 파싱 실패: {e}")
            st.stop()
        reset_session(new_hash, df, uploaded.name)
        st.success(f"✅ 새 파일 업로드 — `{uploaded.name}` ({len(df):,} 행) — 세션이 초기화되었습니다.")

# 샘플 데이터 로드
if use_sample:
    if not os.path.exists(SAMPLE_CSV):
        st.error(f"샘플 파일을 찾을 수 없습니다: {SAMPLE_CSV}")
    else:
        with open(SAMPLE_CSV, "rb") as f:
            raw = f.read()
        new_hash = compute_file_hash(raw)
        if st.session_state.get("file_hash") != new_hash:
            df = pd.read_csv(SAMPLE_CSV)
            reset_session(new_hash, df, "multivariate_sensor.csv (sample)")
            st.success(f"✅ 샘플 데이터 로드 완료 ({len(df):,} 행)")


# ---------------------------------------------------------------------------
# 데이터 미리보기 + 인덱스/변수 설정
# ---------------------------------------------------------------------------

df = st.session_state.get("df")
if df is None:
    st.info("위에서 CSV를 업로드하거나 샘플 데이터를 사용해주세요.")
    st.stop()

st.divider()
st.subheader("📋 데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"전체 형상: {df.shape[0]:,} 행 × {df.shape[1]} 열 | 출처: {st.session_state.get('source_name', '-')}")


st.divider()
st.subheader("⚙️ 시간 인덱스 & 분석 변수 설정")

auto_time = detect_time_column(df)
all_cols = list(df.columns)

c1, c2 = st.columns(2)
with c1:
    default_idx = all_cols.index(auto_time) if auto_time in all_cols else 0
    time_col = st.selectbox(
        "⏰ 시간 인덱스 컬럼",
        options=all_cols,
        index=default_idx,
        help="자동 감지된 컬럼이 기본값으로 선택됩니다.",
    )

with c2:
    numeric_cols = detect_numeric_columns(df, exclude=[time_col])
    default_vars = [c for c in numeric_cols if c.lower() not in ("label", "anomaly", "target")]
    if not default_vars:
        default_vars = numeric_cols
    value_cols = st.multiselect(
        "📈 분석 대상 변수 (다중 선택)",
        options=numeric_cols,
        default=default_vars,
        help="이상탐지의 대상이 될 숫자형 변수를 선택하세요.",
    )

# 주파수
try:
    time_series_parsed = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values()
    inferred = infer_frequency(time_series_parsed)
except Exception:
    inferred = None

c3, c4 = st.columns(2)
with c3:
    freq_options = ["auto"] + ["min", "5min", "10min", "15min", "30min", "h", "D", "W", "MS", "M", "Q", "Y"]
    freq_choice = st.selectbox(
        "📐 주파수 (freq)",
        options=freq_options,
        index=0,
        help=f"자동 추론된 freq: {inferred or '추론 실패'}",
    )
with c4:
    fill_missing = st.checkbox("결측치 자동 보간 (선형)", value=True)

freq_val = inferred if freq_choice == "auto" else freq_choice
st.caption(f"적용될 주파수: `{freq_val}`")


# ---------------------------------------------------------------------------
# TimeSeries 변환
# ---------------------------------------------------------------------------

if not value_cols:
    st.warning("분석 대상 변수를 1개 이상 선택해주세요.")
    st.stop()

if st.button("🛠️ TimeSeries 생성", type="primary", use_container_width=True):
    try:
        with st.spinner("TimeSeries 변환 중..."):
            ts = to_darts_timeseries(df, time_col, value_cols, freq=freq_val, fill_missing=fill_missing)
        st.session_state["ts"] = ts
        st.session_state["config"] = {
            "time_col": time_col,
            "value_cols": value_cols,
            "freq": freq_val,
            "fill_missing": fill_missing,
        }
        st.success(f"✅ TimeSeries 생성 완료 — 길이 {len(ts)}, 변수 {ts.n_components}개")
    except Exception as e:
        st.error(f"TimeSeries 변환 실패: {e}")


if st.session_state.get("ts") is not None:
    ts = st.session_state["ts"]
    c_a, c_b, c_c, c_d = st.columns(4)
    c_a.metric("시점 수", f"{len(ts):,}")
    c_b.metric("변수 수", f"{ts.n_components}")
    c_c.metric("시작", str(ts.start_time())[:19])
    c_d.metric("끝", str(ts.end_time())[:19])


# ---------------------------------------------------------------------------
# 라벨 데이터 (선택)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🏷️ 라벨 데이터 (선택) — 지도 평가지표용")

mode = st.radio(
    "라벨 입력 방법",
    options=["사용 안 함", "원본 CSV의 컬럼 사용", "별도 CSV 업로드"],
    horizontal=True,
)

if mode == "원본 CSV의 컬럼 사용":
    label_col = st.selectbox("라벨 컬럼 (0=정상, 1=이상)", options=all_cols)
    if st.button("🏷️ 라벨 등록", use_container_width=True):
        try:
            labels_df = load_labels(df, time_col, label_col)
            st.session_state["labels"] = labels_df
            st.success(f"✅ 라벨 등록 완료 ({len(labels_df)} 시점, 양성 {int(labels_df['label'].sum())}개)")
        except Exception as e:
            st.error(f"라벨 처리 실패: {e}")

elif mode == "별도 CSV 업로드":
    use_sample_label = st.button("샘플 라벨 사용")
    label_file = st.file_uploader("라벨 CSV 업로드", type=["csv"], key="label_uploader")

    label_df_raw = None
    label_source = None
    if use_sample_label and os.path.exists(SAMPLE_LABELS):
        label_df_raw = pd.read_csv(SAMPLE_LABELS)
        label_source = "샘플 라벨"
    elif label_file is not None:
        try:
            label_df_raw = read_csv_safely(label_file)
            label_source = label_file.name
        except Exception as e:
            st.error(f"라벨 CSV 파싱 실패: {e}")

    if label_df_raw is not None:
        st.dataframe(label_df_raw.head(5), use_container_width=True)
        cols = list(label_df_raw.columns)
        c5, c6 = st.columns(2)
        with c5:
            l_time = st.selectbox("라벨 시간 컬럼", options=cols, key="ltime")
        with c6:
            l_label = st.selectbox("라벨 값 컬럼", options=cols,
                                   index=1 if len(cols) > 1 else 0, key="lval")
        if st.button("🏷️ 라벨 등록", key="register_label", use_container_width=True):
            try:
                labels_df = load_labels(label_df_raw, l_time, l_label)
                st.session_state["labels"] = labels_df
                st.success(f"✅ 라벨 등록 완료 ({len(labels_df)} 시점, 양성 {int(labels_df['label'].sum())}개) — 출처: {label_source}")
            except Exception as e:
                st.error(f"라벨 처리 실패: {e}")
else:
    if st.session_state.get("labels") is not None:
        if st.button("🗑️ 라벨 제거"):
            st.session_state["labels"] = None
            st.success("라벨이 제거되었습니다.")


if st.session_state.get("labels") is not None:
    lab = st.session_state["labels"]
    c1, c2 = st.columns(2)
    c1.metric("라벨 시점 수", f"{len(lab):,}")
    c2.metric("양성(이상) 비율", f"{100*lab['label'].mean():.2f}%")


st.divider()
st.success("➡️ 다음 단계: 좌측 사이드바에서 **`🔍 탐색적 분석`** 페이지로 이동하세요.")
