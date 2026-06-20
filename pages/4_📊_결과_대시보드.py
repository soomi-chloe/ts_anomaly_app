"""Page 4 — 결과 대시보드 (시각화 + 평가지표)."""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.anomaly_detection import (
    scores_to_dataframe, apply_threshold, find_anomaly_segments, suggest_threshold,
)
from utils.evaluation import (
    unsupervised_summary, supervised_metrics, align_scores_labels,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    compare_scorers_table,
)
from utils.eda import plot_distribution


st.set_page_config(page_title="결과 대시보드", page_icon="📊", layout="wide")
st.title("📊 결과 대시보드")


scores = st.session_state.get("anomaly_scores")
if not scores:
    st.warning("먼저 **`⚙️ 이상탐지 설정`** 페이지에서 이상탐지를 실행해주세요.")
    st.stop()

ts = st.session_state.get("ts")
ts_test = st.session_state.get("ts_test")
labels_df = st.session_state.get("labels")
config = st.session_state.get("config", {})


# ---------------------------------------------------------------------------
# Scorer 선택
# ---------------------------------------------------------------------------

scorer_names = list(scores.keys())
c_top1, c_top2 = st.columns([2, 1])
with c_top1:
    selected_scorer = st.selectbox("🎯 표시할 Scorer", options=scorer_names, index=0)
with c_top2:
    mode_label_map = {"joint": "Joint", "per_variable": "Per-Variable",
                      "per_variable_auto": "Per-Variable (자동)", "reproduce": "Reproduce"}
    st.metric("다변량 모드", mode_label_map.get(st.session_state.get("multivariate_mode", "joint"), "-"))

current_score_ts = scores[selected_scorer]
scores_df = scores_to_dataframe(current_score_ts)


# ---------------------------------------------------------------------------
# 임계값 설정
# ---------------------------------------------------------------------------

score_arr = scores_df["score"].values
score_arr = score_arr[~np.isnan(score_arr)]
if len(score_arr) == 0:
    st.error("이상 점수에 유효 값이 없습니다.")
    st.stop()

suggest = suggest_threshold(score_arr, percentile=95.0)

st.subheader("🎚️ 임계값")
c_t1, c_t2 = st.columns([3, 1])
with c_t1:
    threshold = st.slider(
        "이상 임계값",
        min_value=float(np.nanmin(score_arr)),
        max_value=float(np.nanmax(score_arr)),
        value=float(suggest),
        step=float((np.nanmax(score_arr) - np.nanmin(score_arr)) / 200 or 0.001),
    )
with c_t2:
    if st.button("📍 자동 추천(95%)"):
        threshold = float(suggest)
    st.metric("현재 임계값", f"{threshold:.4f}")

anomaly_df = apply_threshold(scores_df, threshold)


# ---------------------------------------------------------------------------
# 4.1 메인 차트 — 원본 시계열 + 이상 구간 하이라이트
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📈 원본 시계열 + 이상 구간 하이라이트")

variables = list(ts.components) if ts is not None else []
if variables:
    var_tabs = st.tabs([str(v) for v in variables])
    for var, tab in zip(variables, var_tabs):
        with tab:
            # 원본 시계열 (테스트 구간 포함 전체)
            try:
                series = ts[var]
                if hasattr(series, "to_dataframe"):
                    ts_df = series.to_dataframe().reset_index()
                else:
                    ts_df = series.pd_dataframe().reset_index()
                ts_df.columns = ["time", "value"]
            except Exception:
                continue

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=ts_df["time"], y=ts_df["value"], mode="lines",
                           name=str(var), line=dict(width=1.0, color="#1f77b4"))
            )
            # 이상 구간 점 표시
            anom_pts = anomaly_df[anomaly_df["anomaly"] == 1]
            if not anom_pts.empty:
                # 시계열에서 매칭되는 점만
                merged = anom_pts.merge(ts_df, on="time", how="inner")
                if not merged.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=merged["time"], y=merged["value"],
                            mode="markers", name="이상 시점",
                            marker=dict(color="#ff4b4b", size=6, symbol="circle"),
                        )
                    )
            # 이상 구간 음영
            segments = find_anomaly_segments(anomaly_df)
            for _, seg in segments.iterrows():
                fig.add_vrect(
                    x0=seg["start"], x1=seg["end"],
                    fillcolor="#ff4b4b", opacity=0.18, layer="below", line_width=0,
                )

            fig.update_layout(
                template="plotly_dark", title=f"{var}",
                xaxis_title="시간", yaxis_title="값",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 4.2 이상 점수 + 임계값
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📉 이상 점수 시계열")

fig_score = go.Figure()
fig_score.add_trace(
    go.Scatter(x=scores_df["time"], y=scores_df["score"], mode="lines",
               name=selected_scorer, line=dict(color="#ffa500", width=1.2))
)
fig_score.add_hline(y=threshold, line_dash="dash", line_color="#ff4b4b",
                    annotation_text=f"임계값={threshold:.3f}", annotation_position="top right")
# 이상 점수 마커
anom_pts = anomaly_df[anomaly_df["anomaly"] == 1]
if not anom_pts.empty:
    fig_score.add_trace(
        go.Scatter(x=anom_pts["time"], y=anom_pts["score"],
                   mode="markers", name="이상 시점",
                   marker=dict(color="#ff4b4b", size=5))
    )
fig_score.update_layout(template="plotly_dark", xaxis_title="시간", yaxis_title="이상 점수",
                        height=350)
st.plotly_chart(fig_score, use_container_width=True)


# ---------------------------------------------------------------------------
# 4.3 평가지표 대시보드
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📐 평가지표")

unsup = unsupervised_summary(scores_df["score"].values, threshold)

# 비지도 KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("탐지된 이상 수", f"{unsup.get('n_anomalies', 0):,}")
k2.metric("이상 비율(%)", f"{unsup.get('anomaly_ratio_pct', 0):.2f}")
k3.metric("점수 평균", f"{unsup.get('score_mean', 0):.3f}")
k4.metric("점수 최대", f"{unsup.get('score_max', 0):.3f}")

c_dist, c_stats = st.columns(2)
with c_dist:
    st.plotly_chart(
        plot_distribution(scores_df["score"].values, title="이상 점수 분포", threshold=threshold),
        use_container_width=True,
    )
with c_stats:
    st.markdown("**점수 통계 요약**")
    stat_rows = {k: v for k, v in unsup.items()
                 if k not in ("n_total", "n_anomalies", "anomaly_ratio_pct")}
    st.dataframe(pd.DataFrame(stat_rows.items(), columns=["지표", "값"]),
                 use_container_width=True, hide_index=True)


# 지도 평가지표
has_labels = labels_df is not None and not labels_df.empty
if has_labels:
    st.divider()
    st.subheader("🏷️ 지도 평가지표 (라벨 사용)")

    aligned_scores, aligned_labels = align_scores_labels(
        scores_df["time"].values, scores_df["score"].values, labels_df,
    )

    if len(aligned_scores) == 0:
        st.warning("점수 시점과 라벨 시점이 겹치지 않습니다.")
    elif len(np.unique(aligned_labels)) < 2:
        st.warning("정상/이상 라벨이 모두 존재해야 평가가 가능합니다.")
    else:
        metrics = supervised_metrics(aligned_scores, aligned_labels, threshold)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC-ROC", metrics.get("AUC-ROC", "—"))
        m2.metric("AUC-PR", metrics.get("AUC-PR", "—"))
        m3.metric("Precision", metrics.get("Precision", "—"))
        m4.metric("Recall", metrics.get("Recall", "—"))
        m5.metric("F1", metrics.get("F1", "—"))

        c_roc, c_pr, c_cm = st.columns(3)
        with c_roc:
            st.plotly_chart(plot_roc_curve(aligned_scores, aligned_labels), use_container_width=True)
        with c_pr:
            st.plotly_chart(plot_pr_curve(aligned_scores, aligned_labels), use_container_width=True)
        with c_cm:
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
else:
    st.info("라벨 데이터가 없으면 비지도 지표만 표시됩니다. 페이지 1에서 라벨 CSV를 등록하면 지도 평가지표가 자동 활성화됩니다.")


# ---------------------------------------------------------------------------
# 4.4 Scorer 비교
# ---------------------------------------------------------------------------

if len(scorer_names) > 1:
    st.divider()
    st.subheader("🔀 Scorer 비교")
    # 오버레이 차트
    fig_cmp = go.Figure()
    for name in scorer_names:
        sdf = scores_to_dataframe(scores[name])
        fig_cmp.add_trace(go.Scatter(x=sdf["time"], y=sdf["score"], mode="lines",
                                     name=name, line=dict(width=1.0)))
    fig_cmp.update_layout(template="plotly_dark", title="Scorer별 이상 점수 오버레이",
                          xaxis_title="시간", yaxis_title="점수", height=350)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # 비교 테이블
    score_arrays = {n: scores_to_dataframe(scores[n])["score"].values for n in scorer_names}
    labels_arr = None
    if has_labels:
        # 첫 scorer 기준으로 라벨 정렬
        base = scores_to_dataframe(scores[scorer_names[0]])
        _, labels_arr = align_scores_labels(base["time"].values, base["score"].values, labels_df)
    cmp_table = compare_scorers_table(score_arrays, labels_arr, threshold)
    st.dataframe(cmp_table, use_container_width=True, hide_index=True)
    if has_labels and "AUC-ROC" in cmp_table.columns:
        try:
            best = cmp_table.sort_values("AUC-ROC", ascending=False).iloc[0]
            st.success(f"🏆 추천 Scorer (AUC-ROC 기준): **{best['Scorer']}** ({best['AUC-ROC']:.4f})")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 4.5 이상 구간 상세
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📋 이상 구간 상세")

segments = find_anomaly_segments(anomaly_df)
if segments.empty:
    st.info("현재 임계값에서는 이상 구간이 없습니다.")
else:
    st.dataframe(segments, use_container_width=True, hide_index=True)
    st.caption(f"총 {len(segments)}개 구간 — 평균 점수 {segments['mean_score'].mean():.3f}")

# CSV 다운로드
download_df = anomaly_df.copy()
download_df["threshold"] = threshold
download_df["scorer"] = selected_scorer
csv_bytes = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ 이상 점수 + 판정 결과 CSV 다운로드",
    data=csv_bytes,
    file_name=f"anomaly_result_{selected_scorer}.csv",
    mime="text/csv",
    use_container_width=True,
)
