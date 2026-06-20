"""평가지표 계산 및 시각화 (지도/비지도)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 비지도 (라벨 없음)
# ---------------------------------------------------------------------------

def unsupervised_summary(scores: np.ndarray, threshold: float) -> Dict:
    arr = np.asarray(scores)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {}
    n_anom = int(np.sum(arr >= threshold))
    return {
        "n_total": int(len(arr)),
        "n_anomalies": n_anom,
        "anomaly_ratio_pct": round(100.0 * n_anom / len(arr), 3),
        "score_mean": float(np.mean(arr)),
        "score_std": float(np.std(arr)),
        "score_min": float(np.min(arr)),
        "score_max": float(np.max(arr)),
        "score_p50": float(np.percentile(arr, 50)),
        "score_p90": float(np.percentile(arr, 90)),
        "score_p99": float(np.percentile(arr, 99)),
    }


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """예측 성능 (MAE/MSE/MAPE)."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    n = min(len(yt), len(yp))
    if n == 0:
        return {"MAE": np.nan, "MSE": np.nan, "MAPE": np.nan}
    yt, yp = yt[:n], yp[:n]
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    denom = np.where(np.abs(yt) > 1e-8, np.abs(yt), 1e-8)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100)
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 3)}


# ---------------------------------------------------------------------------
# 지도 (라벨 있음)
# ---------------------------------------------------------------------------

def align_scores_labels(
    score_times: pd.DatetimeIndex,
    score_values: np.ndarray,
    label_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """점수와 라벨을 시간 기준으로 정렬해 1D 배열로 반환."""
    s_df = pd.DataFrame({"time": pd.to_datetime(score_times), "score": score_values.reshape(-1)})
    l_df = label_df.copy()
    l_df["time"] = pd.to_datetime(l_df["time"])
    merged = s_df.merge(l_df, on="time", how="inner")
    merged = merged.dropna(subset=["score", "label"])
    return merged["score"].values, merged["label"].values.astype(int)


def supervised_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    """지도 평가지표 (AUC-ROC, AUC-PR, F1, Precision, Recall, 혼동행렬)."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, confusion_matrix,
    )
    s = np.asarray(scores).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    if len(s) == 0 or len(np.unique(y)) < 2:
        return {"note": "라벨 데이터에 양/음 클래스가 모두 있어야 평가 가능"}
    pred = (s >= threshold).astype(int)
    try:
        roc = float(roc_auc_score(y, s))
    except Exception:
        roc = np.nan
    try:
        pr = float(average_precision_score(y, s))
    except Exception:
        pr = np.nan
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "AUC-ROC": round(roc, 4),
        "AUC-PR": round(pr, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "confusion_matrix": cm,
        "n_positive_true": int(y.sum()),
        "n_positive_pred": int(pred.sum()),
    }


def plot_roc_curve(scores: np.ndarray, labels: np.ndarray) -> go.Figure:
    from sklearn.metrics import roc_curve, roc_auc_score
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
    except Exception:
        return go.Figure().update_layout(template="plotly_dark", title="ROC curve (계산 실패)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc:.3f})", line=dict(width=2, color="#ff4b4b")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        template="plotly_dark", title="ROC Curve",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=400,
    )
    return fig


def plot_pr_curve(scores: np.ndarray, labels: np.ndarray) -> go.Figure:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    try:
        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
    except Exception:
        return go.Figure().update_layout(template="plotly_dark", title="PR curve (계산 실패)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"PR (AP={ap:.3f})", line=dict(width=2, color="#1f77b4")))
    fig.update_layout(
        template="plotly_dark", title="Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        height=400,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm, x=["예측: 정상", "예측: 이상"], y=["실제: 정상", "실제: 이상"],
            colorscale="Blues",
            text=cm, texttemplate="%{text}",
        )
    )
    fig.update_layout(template="plotly_dark", title="Confusion Matrix", height=400)
    return fig


# ---------------------------------------------------------------------------
# Scorer 비교
# ---------------------------------------------------------------------------

def compare_scorers_table(
    scores_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for name, arr in scores_dict.items():
        arr = np.asarray(arr).reshape(-1)
        row = {"Scorer": name, "Mean": round(float(np.nanmean(arr)), 4),
               "Std": round(float(np.nanstd(arr)), 4),
               "Max": round(float(np.nanmax(arr)), 4)}
        if labels is not None:
            min_len = min(len(arr), len(labels))
            arr_, lab_ = arr[:min_len], labels[:min_len]
            try:
                metrics = supervised_metrics(arr_, lab_, threshold)
                row["AUC-ROC"] = metrics.get("AUC-ROC")
                row["AUC-PR"] = metrics.get("AUC-PR")
                row["F1"] = metrics.get("F1")
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)
