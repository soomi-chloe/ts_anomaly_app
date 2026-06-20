"""탐색적 분석(EDA) 함수 모음."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# 시계열 시각화
# ---------------------------------------------------------------------------

def plot_timeseries_overlay(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    title: str = "시계열 (오버레이)",
) -> go.Figure:
    fig = go.Figure()
    for c in value_cols:
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[c],
                mode="lines",
                name=c,
                line=dict(width=1.2),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="시간",
        yaxis_title="값",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_timeseries_subplots(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    title: str = "변수별 시계열",
) -> go.Figure:
    n = len(value_cols)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=value_cols)
    for i, c in enumerate(value_cols, start=1):
        fig.add_trace(
            go.Scatter(x=df[time_col], y=df[c], mode="lines", name=c, line=dict(width=1.0)),
            row=i, col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=220 * n,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 상관관계
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, value_cols: List[str]) -> go.Figure:
    corr = df[value_cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="변수 간 상관 행렬",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 결측치 시각화
# ---------------------------------------------------------------------------

def plot_missing_heatmap(df: pd.DataFrame, time_col: str, value_cols: List[str]) -> go.Figure:
    mask = df[value_cols].isna().astype(int).T
    fig = go.Figure(
        data=go.Heatmap(
            z=mask.values,
            x=df[time_col],
            y=value_cols,
            colorscale=[[0, "#1f77b4"], [1, "#ff4b4b"]],
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="결측치 위치 (붉은색 = 결측)",
        height=120 + 30 * len(value_cols),
    )
    return fig


# ---------------------------------------------------------------------------
# 정상성 검정
# ---------------------------------------------------------------------------

def adf_test_table(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """ADF (Augmented Dickey-Fuller) 검정 결과 테이블."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return pd.DataFrame()

    rows = []
    for c in value_cols:
        series = df[c].dropna()
        if len(series) < 20:
            rows.append({"변수": c, "ADF 통계량": np.nan, "p-value": np.nan, "정상성": "데이터 부족"})
            continue
        try:
            stat, pval, *_ = adfuller(series, autolag="AIC")
            verdict = "정상 (Stationary)" if pval < 0.05 else "비정상 (Non-stationary)"
            rows.append({"변수": c, "ADF 통계량": round(float(stat), 4),
                         "p-value": round(float(pval), 4), "정상성": verdict})
        except Exception as e:
            rows.append({"변수": c, "ADF 통계량": np.nan, "p-value": np.nan,
                         "정상성": f"실패: {e}"})
    return pd.DataFrame(rows)


def ljungbox_test_table(df: pd.DataFrame, value_cols: List[str], lags: int = 10) -> pd.DataFrame:
    """Ljung-Box 검정 (자기상관) 결과."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except Exception:
        return pd.DataFrame()

    rows = []
    for c in value_cols:
        series = df[c].dropna()
        if len(series) < lags + 5:
            rows.append({"변수": c, "p-value(lag=10)": np.nan, "자기상관": "데이터 부족"})
            continue
        try:
            res = acorr_ljungbox(series, lags=[lags], return_df=True)
            pval = float(res["lb_pvalue"].iloc[0])
            verdict = "자기상관 존재" if pval < 0.05 else "자기상관 없음"
            rows.append({"변수": c, "p-value(lag=10)": round(pval, 4), "자기상관": verdict})
        except Exception as e:
            rows.append({"변수": c, "p-value(lag=10)": np.nan, "자기상관": f"실패: {e}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 점수 분포 시각화
# ---------------------------------------------------------------------------

def plot_distribution(
    values: np.ndarray,
    title: str = "이상 점수 분포",
    threshold: Optional[float] = None,
) -> go.Figure:
    fig = px.histogram(
        x=values,
        nbins=50,
        title=title,
        marginal="box",
        template="plotly_dark",
    )
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="#ff4b4b",
                      annotation_text=f"임계값={threshold:.3f}", annotation_position="top right")
    fig.update_layout(height=400, showlegend=False, xaxis_title="이상 점수", yaxis_title="빈도")
    return fig
