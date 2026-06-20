"""Page 2 — 탐색적 분석 (EDA)."""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import summarize_dataframe
from utils.eda import (
    plot_timeseries_overlay, plot_timeseries_subplots,
    plot_correlation_heatmap, plot_missing_heatmap,
    adf_test_table, ljungbox_test_table,
)


st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")
st.title("🔍 탐색적 분석 (EDA)")


df = st.session_state.get("df")
config = st.session_state.get("config") or {}
time_col = config.get("time_col")
value_cols = config.get("value_cols") or []

if df is None or not time_col or not value_cols:
    st.warning("먼저 **`📁 데이터 업로드`** 페이지에서 데이터를 등록하고 TimeSeries를 생성해주세요.")
    st.stop()

# 정렬된 작업용 DF
work = df.copy()
work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
work = work.dropna(subset=[time_col]).sort_values(time_col)


# ---------------------------------------------------------------------------
# 1. 기초 통계
# ---------------------------------------------------------------------------

st.subheader("📊 기초 통계")
summary = summarize_dataframe(work[value_cols])
st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("시점 수", f"{len(work):,}")
c2.metric("변수 수", len(value_cols))
miss_total = work[value_cols].isna().mean().mean() * 100
c3.metric("평균 결측률", f"{miss_total:.2f}%")
date_range = (work[time_col].max() - work[time_col].min())
c4.metric("기간", str(date_range))


# ---------------------------------------------------------------------------
# 2. 시계열 시각화
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📈 시계열 시각화")

mode = st.radio("표시 방식", ["오버레이", "변수별 subplot"], horizontal=True, key="eda_mode")

# 시간 범위 슬라이더
t_min, t_max = work[time_col].min(), work[time_col].max()
try:
    rng = st.slider(
        "시간 범위",
        min_value=t_min.to_pydatetime(),
        max_value=t_max.to_pydatetime(),
        value=(t_min.to_pydatetime(), t_max.to_pydatetime()),
        format="YYYY-MM-DD HH:mm",
    )
    mask = (work[time_col] >= rng[0]) & (work[time_col] <= rng[1])
    work_slice = work.loc[mask]
except Exception:
    work_slice = work

if mode == "오버레이":
    fig = plot_timeseries_overlay(work_slice, time_col, value_cols)
else:
    fig = plot_timeseries_subplots(work_slice, time_col, value_cols)
st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. 정상성 / 자기상관 검정
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🧪 정상성 / 자기상관 검정")

c_adf, c_lb = st.columns(2)
with c_adf:
    st.markdown("**ADF (Augmented Dickey-Fuller)**")
    adf_df = adf_test_table(work, value_cols)
    if adf_df.empty:
        st.info("statsmodels이 설치되지 않았거나 검정을 수행할 수 없습니다.")
    else:
        st.dataframe(adf_df, use_container_width=True, hide_index=True)

with c_lb:
    st.markdown("**Ljung-Box (lag=10)**")
    lb_df = ljungbox_test_table(work, value_cols, lags=10)
    if lb_df.empty:
        st.info("Ljung-Box 검정을 수행할 수 없습니다.")
    else:
        st.dataframe(lb_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 4. 상관관계
# ---------------------------------------------------------------------------

if len(value_cols) >= 2:
    st.divider()
    st.subheader("🔗 변수 간 상관 관계")
    fig_corr = plot_correlation_heatmap(work, value_cols)
    st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. 결측치 시각화
# ---------------------------------------------------------------------------

if work[value_cols].isna().any().any():
    st.divider()
    st.subheader("🩹 결측치 위치")
    fig_miss = plot_missing_heatmap(work, time_col, value_cols)
    st.plotly_chart(fig_miss, use_container_width=True)
else:
    st.info("✨ 결측치가 없습니다.")


st.divider()
st.success("➡️ 다음 단계: 좌측 사이드바에서 **`⚙️ 이상탐지 설정`** 페이지로 이동하세요.")
