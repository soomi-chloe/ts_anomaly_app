"""CSV 로드 + Darts TimeSeries 변환 유틸리티."""

from __future__ import annotations

import hashlib
import io
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def compute_file_hash(file_bytes: bytes) -> str:
    """업로드 파일의 MD5 해시 계산 (파일 변경 감지용)."""
    return hashlib.md5(file_bytes).hexdigest()


def read_csv_safely(file_obj) -> pd.DataFrame:
    """업로드된 파일 객체를 안전하게 DataFrame으로 읽기.

    파일 객체 포인터를 다시 처음으로 되돌리고, 한글/UTF-8/CP949 등을 시도한다.
    """
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass

    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            # 다른 에러는 마지막 인코딩 시도
            continue
    # 마지막 시도
    return pd.read_csv(io.BytesIO(raw))


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """DataFrame에서 시간 인덱스로 사용할 수 있는 컬럼을 자동 감지."""
    candidates = []
    for col in df.columns:
        lower = str(col).lower()
        if any(key in lower for key in ("time", "date", "timestamp", "datetime", "기간", "시간", "일자")):
            candidates.append(col)

    # 우선순위 후보 먼저 시도
    for col in candidates:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            if parsed.notna().any():
                return col
        except Exception:
            continue

    # 후보 외 컬럼도 시도
    for col in df.columns:
        if col in candidates:
            continue
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                if parsed.notna().sum() / max(len(df), 1) > 0.9:
                    return col
            except Exception:
                continue
    return None


def detect_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """숫자형 컬럼만 추출 (시간/라벨 등 제외)."""
    exclude = set(exclude or [])
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def infer_frequency(time_series: pd.Series) -> Optional[str]:
    """시계열의 주파수를 자동 추론."""
    if len(time_series) < 3:
        return None
    try:
        freq = pd.infer_freq(time_series)
        return freq
    except Exception:
        return None


def to_darts_timeseries(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    freq: Optional[str] = None,
    fill_missing: bool = True,
):
    """pandas DataFrame을 Darts TimeSeries로 변환.

    - 시간 컬럼을 datetime으로 변환
    - 시간 순 정렬, 중복 제거
    - 결측치는 보간 처리(선택)
    """
    from darts import TimeSeries

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    work = work.sort_values(time_col)
    work = work.drop_duplicates(subset=[time_col])

    # 숫자 변환
    for c in value_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # 결측치 보간
    if fill_missing:
        work[value_cols] = work[value_cols].interpolate(method="linear", limit_direction="both")
        work[value_cols] = work[value_cols].ffill().bfill()

    # 주파수 추론
    if freq is None:
        freq = infer_frequency(work[time_col])

    try:
        ts = TimeSeries.from_dataframe(
            work,
            time_col=time_col,
            value_cols=value_cols,
            freq=freq,
            fill_missing_dates=True,
        )
    except Exception:
        # freq 추론 실패 시 freq 없이 시도
        ts = TimeSeries.from_dataframe(
            work,
            time_col=time_col,
            value_cols=value_cols,
            fill_missing_dates=False,
        )
    return ts


def load_labels(
    df_labels: pd.DataFrame,
    time_col: str,
    label_col: str,
    reference_time_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """라벨 CSV를 표준화. 시간 인덱스와 정렬해 0/1 라벨로 정리.

    Returns
    -------
    DataFrame[time, label]
    """
    out = df_labels[[time_col, label_col]].copy()
    out.columns = ["time", "label"]
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"])
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    out["label"] = (out["label"] != 0).astype(int)
    out = out.drop_duplicates(subset=["time"]).sort_values("time")

    if reference_time_index is not None:
        ref = pd.DataFrame({"time": reference_time_index})
        out = ref.merge(out, on="time", how="left")
        out["label"] = out["label"].fillna(0).astype(int)
    return out


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """기초 통계 요약 (count, mean, std, min, max, 결측치 비율)."""
    desc = df.describe(include=[np.number]).T
    miss = df.isna().mean().rename("missing_ratio")
    summary = desc.join(miss, how="left")
    return summary
