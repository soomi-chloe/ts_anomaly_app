"""Darts 기반 이상탐지 핵심 로직."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 모델 팩토리
# ---------------------------------------------------------------------------

def create_forecasting_model(model_name: str, params: Dict):
    """이름으로 Darts 예측 모형 생성."""
    lags = int(params.get("lags", 24))
    output_chunk_length = int(params.get("output_chunk_length", 1))

    name = (model_name or "").lower()

    if name in ("linearregression", "linearregressionmodel", "linear"):
        from darts.models import LinearRegressionModel
        return LinearRegressionModel(lags=lags, output_chunk_length=output_chunk_length)

    if name in ("ridge", "sklearnmodel(ridge)", "sklearn_ridge"):
        try:
            from darts.models import SKLearnModel
            from sklearn.linear_model import Ridge
            return SKLearnModel(lags=lags, output_chunk_length=output_chunk_length,
                                model=Ridge(alpha=1.0))
        except Exception:
            # 구버전 호환
            from darts.models import RegressionModel
            from sklearn.linear_model import Ridge
            return RegressionModel(lags=lags, output_chunk_length=output_chunk_length,
                                   model=Ridge(alpha=1.0))

    if name in ("xgb", "xgboost", "xgbmodel"):
        try:
            from darts.models import XGBModel
            return XGBModel(lags=lags, output_chunk_length=output_chunk_length,
                            random_state=42, verbosity=0)
        except Exception:
            from darts.models import LinearRegressionModel
            return LinearRegressionModel(lags=lags, output_chunk_length=output_chunk_length)

    if name in ("lgbm", "lightgbm", "lightgbmmodel"):
        try:
            from darts.models import LightGBMModel
            return LightGBMModel(lags=lags, output_chunk_length=output_chunk_length,
                                 random_state=42, verbose=-1)
        except Exception:
            from darts.models import LinearRegressionModel
            return LinearRegressionModel(lags=lags, output_chunk_length=output_chunk_length)

    if name in ("exponentialsmoothing", "es", "exp_smoothing"):
        from darts.models import ExponentialSmoothing
        return ExponentialSmoothing()

    if name in ("autoarima", "arima"):
        from darts.models import AutoARIMA
        return AutoARIMA()

    # default
    from darts.models import LinearRegressionModel
    return LinearRegressionModel(lags=lags, output_chunk_length=output_chunk_length)


def model_supports_multivariate(model_name: str) -> bool:
    """모델이 다변량을 자체 지원하는지(통계 모형은 단변량)."""
    name = (model_name or "").lower()
    if name in ("exponentialsmoothing", "es", "autoarima", "arima"):
        return False
    return True


def is_local_forecasting_model(model_name: str) -> bool:
    """Darts ForecastingAnomalyModel은 GlobalForecastingModel만 받음.
    LocalForecastingModel(ES/ARIMA 등)은 직접 잔차 기반 점수를 계산해야 함.
    """
    name = (model_name or "").lower()
    return name in ("exponentialsmoothing", "es", "autoarima", "arima")


# ---------------------------------------------------------------------------
# Scorer 팩토리
# ---------------------------------------------------------------------------

def create_scorers(scorer_names: List[str], params: Dict, component_wise: bool = False) -> List:
    """이름 목록으로 Darts AD scorer 리스트 생성."""
    from darts.ad import NormScorer, KMeansScorer, WassersteinScorer

    window = int(params.get("window", 10))
    k = int(params.get("k", 2))

    scorers = []
    for name in scorer_names:
        nm = name.lower()
        if nm in ("normscorer", "norm"):
            try:
                scorers.append(NormScorer(component_wise=component_wise))
            except TypeError:
                scorers.append(NormScorer())
        elif nm in ("kmeansscorer", "kmeans"):
            try:
                scorers.append(KMeansScorer(k=k, window=window,
                                            component_wise=component_wise))
            except TypeError:
                scorers.append(KMeansScorer(k=k, window=window))
        elif nm in ("wassersteinscorer", "wasserstein"):
            try:
                scorers.append(WassersteinScorer(window=max(window, 10),
                                                 component_wise=component_wise))
            except TypeError:
                scorers.append(WassersteinScorer(window=max(window, 10)))
    return scorers


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def run_anomaly_detection(
    ts_train,
    ts_test,
    model_name: str,
    scorer_names: List[str],
    params: Optional[Dict] = None,
    multivariate_mode: str = "joint",
):
    """Darts ForecastingAnomalyModel 기반 이상탐지 파이프라인.

    Parameters
    ----------
    multivariate_mode : "joint" | "per_variable" | "reproduce"

    Returns
    -------
    dict with keys:
        scores : dict[scorer_name -> TimeSeries]
        anomaly_model
        forecast : TimeSeries (테스트 구간 예측)
        per_variable_scores : dict (per_variable 모드에서만)
    """
    from darts.ad import ForecastingAnomalyModel

    params = params or {}
    is_multivariate = ts_train.n_components > 1

    # Per-Variable 모드: 변수별 독립 학습
    if multivariate_mode == "per_variable" and is_multivariate:
        per_var = {}
        for comp in ts_train.components:
            ts_tr_c = ts_train[comp]
            ts_te_c = ts_test[comp]
            sub = _run_single(ts_tr_c, ts_te_c, model_name, scorer_names, params,
                              component_wise=False)
            per_var[comp] = sub
        # 통합 점수: 변수별 첫 번째 scorer의 평균 (간단)
        agg_scores = _aggregate_per_var_scores(per_var, scorer_names)
        return {
            "mode": "per_variable",
            "scores": agg_scores,
            "per_variable_scores": per_var,
        }

    # Joint / Reproduce 모드
    component_wise = False
    return _run_single(ts_train, ts_test, model_name, scorer_names, params,
                       component_wise=component_wise)


def _run_single(ts_train, ts_test, model_name, scorer_names, params, component_wise=False):
    """단일 ForecastingAnomalyModel 실행."""
    from darts.ad import ForecastingAnomalyModel

    is_multivariate = ts_train.n_components > 1

    # 단변량 통계 모형 + 다변량 시계열 → Per-Variable 강제 전환
    if is_multivariate and not model_supports_multivariate(model_name):
        per_var = {}
        for comp in ts_train.components:
            ts_tr_c = ts_train[comp]
            ts_te_c = ts_test[comp]
            sub = _run_single(ts_tr_c, ts_te_c, model_name, scorer_names, params,
                              component_wise=False)
            per_var[comp] = sub
        agg_scores = _aggregate_per_var_scores(per_var, scorer_names)
        return {
            "mode": "per_variable_auto",
            "scores": agg_scores,
            "per_variable_scores": per_var,
        }

    # LocalForecastingModel(ES/ARIMA)은 ForecastingAnomalyModel 사용 불가 → 잔차 직접 계산
    if is_local_forecasting_model(model_name):
        return _run_local_residual(ts_train, ts_test, model_name, scorer_names, params)

    forecasting_model = create_forecasting_model(model_name, params)

    scorers = create_scorers(scorer_names, params, component_wise=component_wise)
    if not scorers:
        from darts.ad import NormScorer
        scorers = [NormScorer()]

    anomaly_model = ForecastingAnomalyModel(model=forecasting_model, scorer=scorers)

    # 학습
    try:
        anomaly_model.fit(ts_train, allow_model_training=True)
    except TypeError:
        anomaly_model.fit(ts_train)

    # 스코어 계산 (start로 테스트 구간 지정)
    try:
        scores = anomaly_model.score(ts_test, start=ts_test.start_time())
    except Exception:
        # 일부 모형에서는 학습+테스트를 함께 넘겨야 함
        from darts import concatenate
        full = concatenate([ts_train, ts_test], axis=0)
        scores = anomaly_model.score(full, start=ts_test.start_time())

    # 정규화: scorers 1개여도 리스트로 통일
    if not isinstance(scores, (list, tuple)):
        scores = [scores]

    scores_dict = {}
    for name, sc in zip(scorer_names, scores):
        scores_dict[name] = sc

    return {
        "mode": "joint",
        "scores": scores_dict,
        "anomaly_model": anomaly_model,
    }


def _run_local_residual(ts_train, ts_test, model_name, scorer_names, params):
    """LocalForecastingModel (ES/ARIMA) — 단변량 잔차 기반 점수 계산.

    forecasting_model.fit(ts_train) → predict(len(ts_test)) → |잔차| → NormScorer로 처리.
    KMeans/Wasserstein 등 다른 scorer가 선택되어도 잔차를 입력으로 fit/score.
    """
    from darts import TimeSeries

    fm = create_forecasting_model(model_name, params)
    fm.fit(ts_train)

    # 학습 구간 잔차 — Scorer fit용
    try:
        train_pred = fm.predict(len(ts_train), series=ts_train)
    except Exception:
        # 일부 모형은 in-sample 예측을 지원하지 않음 → 잔차 fit 단계 skip
        train_pred = None

    test_pred = fm.predict(len(ts_test))

    # 잔차 계산 (절대값)
    test_resid = np.abs(
        ts_test.values().reshape(-1)[:len(test_pred)] - test_pred.values().reshape(-1)
    )
    test_resid_ts = TimeSeries.from_times_and_values(
        ts_test.time_index[:len(test_resid)], test_resid
    )

    scores_dict = {}
    for s_name in scorer_names:
        nm = s_name.lower()
        if nm in ("normscorer", "norm"):
            # 그대로 절대 잔차 사용
            scores_dict[s_name] = test_resid_ts
        else:
            # KMeans/Wasserstein 등은 잔차 시계열을 입력으로 fit/score
            try:
                scorer = create_scorers([s_name], params)[0]
                # train 잔차로 fit
                if train_pred is not None:
                    train_resid = np.abs(
                        ts_train.values().reshape(-1)[:len(train_pred)]
                        - train_pred.values().reshape(-1)
                    )
                    train_resid_ts = TimeSeries.from_times_and_values(
                        ts_train.time_index[:len(train_resid)], train_resid
                    )
                    scorer.fit(train_resid_ts)
                    out = scorer.score(test_resid_ts)
                else:
                    out = test_resid_ts
                scores_dict[s_name] = out
            except Exception:
                # 실패 시 절대 잔차로 대체
                scores_dict[s_name] = test_resid_ts

    return {
        "mode": "joint_local_residual",
        "scores": scores_dict,
        "anomaly_model": fm,
    }


def _aggregate_per_var_scores(per_var: Dict, scorer_names: List[str]) -> Dict:
    """변수별 점수를 scorer 별로 평균 통합."""
    agg = {}
    for s_name in scorer_names:
        series_list = []
        for var, res in per_var.items():
            sc = res["scores"].get(s_name)
            if sc is not None:
                series_list.append(sc)
        if not series_list:
            continue
        # 시간축이 같다고 가정하고 평균
        try:
            arrs = [s.values().reshape(-1) for s in series_list]
            min_len = min(len(a) for a in arrs)
            arrs = [a[:min_len] for a in arrs]
            mean_arr = np.mean(np.stack(arrs, axis=0), axis=0)
            base = series_list[0]
            from darts import TimeSeries
            agg[s_name] = TimeSeries.from_times_and_values(
                base.time_index[:min_len], mean_arr
            )
        except Exception:
            agg[s_name] = series_list[0]
    return agg


# ---------------------------------------------------------------------------
# 임계값 / 이상 구간 추출
# ---------------------------------------------------------------------------

def scores_to_dataframe(scores_ts) -> pd.DataFrame:
    """Darts TimeSeries 점수를 DataFrame으로 변환."""
    # darts 신/구 버전 호환
    if hasattr(scores_ts, "to_dataframe"):
        df = scores_ts.to_dataframe()
    else:
        df = scores_ts.pd_dataframe()
    df = df.reset_index()
    df.columns = ["time"] + [str(c) for c in df.columns[1:]]
    # 다변량 점수면 평균을 score로
    val_cols = [c for c in df.columns if c != "time"]
    df["score"] = df[val_cols].mean(axis=1)
    return df


def apply_threshold(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = scores_df.copy()
    out["anomaly"] = (out["score"] >= threshold).astype(int)
    return out


def find_anomaly_segments(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """연속된 이상 구간을 (start, end, duration, mean_score)로 그룹화."""
    if anomaly_df.empty or "anomaly" not in anomaly_df.columns:
        return pd.DataFrame(columns=["start", "end", "duration", "mean_score", "max_score"])

    a = anomaly_df["anomaly"].values
    times = anomaly_df["time"].values
    scores = anomaly_df["score"].values

    segments = []
    in_seg = False
    seg_start = None
    seg_scores = []
    for i, v in enumerate(a):
        if v == 1 and not in_seg:
            in_seg = True
            seg_start = times[i]
            seg_scores = [scores[i]]
        elif v == 1 and in_seg:
            seg_scores.append(scores[i])
        elif v == 0 and in_seg:
            seg_end = times[i - 1]
            duration = pd.to_datetime(seg_end) - pd.to_datetime(seg_start)
            segments.append({
                "start": seg_start, "end": seg_end,
                "duration": str(duration),
                "mean_score": float(np.mean(seg_scores)),
                "max_score": float(np.max(seg_scores)),
            })
            in_seg = False
            seg_scores = []
    if in_seg:
        seg_end = times[-1]
        duration = pd.to_datetime(seg_end) - pd.to_datetime(seg_start)
        segments.append({
            "start": seg_start, "end": seg_end,
            "duration": str(duration),
            "mean_score": float(np.mean(seg_scores)),
            "max_score": float(np.max(seg_scores)),
        })
    return pd.DataFrame(segments)


def suggest_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """기본 임계값 추천: 학습 점수 분포의 95 percentile."""
    arr = np.asarray(scores)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, percentile))
