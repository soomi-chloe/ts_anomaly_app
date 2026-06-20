"""전처리 파이프라인."""

from __future__ import annotations

from typing import Optional, Tuple


def split_train_test(ts, train_ratio: float = 0.7):
    """학습/테스트 분할."""
    n = len(ts)
    cut = max(1, int(n * train_ratio))
    cut = min(cut, n - 1)
    return ts[:cut], ts[cut:]


def apply_scaler(ts_train, ts_test, scaler_name: str = "MinMax"):
    """학습 데이터에 fit, 양쪽에 transform.

    Returns (ts_train_scaled, ts_test_scaled, scaler)
    """
    from darts.dataprocessing.transformers import Scaler

    if scaler_name in (None, "None", "none"):
        return ts_train, ts_test, None

    if scaler_name == "MinMax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = Scaler(MinMaxScaler())
    elif scaler_name == "Standard":
        from sklearn.preprocessing import StandardScaler
        scaler = Scaler(StandardScaler())
    else:
        return ts_train, ts_test, None

    ts_train_s = scaler.fit_transform(ts_train)
    ts_test_s = scaler.transform(ts_test)
    return ts_train_s, ts_test_s, scaler


def apply_differencing(ts_train, ts_test, order: int = 0):
    """차분 적용 (0/1/2)."""
    if order <= 0:
        return ts_train, ts_test
    try:
        from darts.dataprocessing.transformers import Diff
        diff = Diff(lags=[1] * order)
        ts_train_d = diff.fit_transform(ts_train)
        ts_test_d = diff.transform(ts_test)
        return ts_train_d, ts_test_d
    except Exception:
        # Fallback: 단순 차분
        return ts_train, ts_test


def preprocess_pipeline(
    ts,
    train_ratio: float = 0.7,
    scaler_name: str = "MinMax",
    diff_order: int = 0,
):
    """전체 전처리 파이프라인."""
    ts_train, ts_test = split_train_test(ts, train_ratio)
    ts_train, ts_test = apply_differencing(ts_train, ts_test, diff_order)
    ts_train, ts_test, scaler = apply_scaler(ts_train, ts_test, scaler_name)
    return ts_train, ts_test, scaler
