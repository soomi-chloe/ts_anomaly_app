"""절삭음 데이터셋 벤치마크 실행 스크립트.

질문
----
"정상(비절삭) 소리만 보고 학습한 이상탐지 모형이 절삭 구간을 찾아낼 수 있는가?
 실험실에서 학습한 모형이 현장(다른 기계·다른 재료)으로 옮겨 가는가?"

프로토콜
--------
grouped_cv       : 실험실(IMI) 18개 실험을 실험 단위로 K겹 분할. 학습 실험의 비절삭 프레임으로 학습,
                   다른 실험 전체를 채점. 같은 실험의 프레임이 학습과 평가에 동시에 들어가지 않는다.
lab_to_industry  : IMI 전체로 학습, KRPM(현장) 채점. 연구실 논문(INDIN 2024)의 설정을 흉내 낸다.
random_leaky     : (대조군) 프레임을 무작위로 섞어 70/30 분할. 인접 프레임이 양쪽에 들어가므로 누수.
                   이 수치가 얼마나 부풀려지는지를 리포트에서 보여 준다.

탐지기
------
energy_spindle   : 스핀들 센서 RMS(dB) 그 자체를 점수로. 학습 없음. "이걸 못 이기면 모형이 필요 없다".
iforest          : IsolationForest, 표준화된 42차원 특징, 비절삭 프레임으로 학습.
darts_linreg_norm: ts_anomaly_app 과 같은 Darts ForecastingAnomalyModel(LinearRegression, lags) + NormScorer.
darts_linreg_kmeans: 같은 예측기 + KMeansScorer(window).

임계값
------
학습에 쓴 비절삭 프레임 점수의 상위 분위수(기본 0.99). 테스트 라벨은 임계값 선택에 쓰지 않는다.
대조용으로 '테스트 라벨로 F1 최대화한 임계값'의 F1 도 함께 저장한다 (f1_oracle).

실행
----
python -m benchmark.run_benchmark --data-root C:/Users/109so/data/mt_cutting_dataset
python -m benchmark.run_benchmark --quick          # 실험 6개, 탐지기 2개로 빠르게
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import metrics as M
from .dataset import (Experiment, attach_chatter_labels, chatter_summary, discover_experiments,
                      featurize, normal_segments, runs_where, summarize)
from .features import FEATURE_NAMES, FrameConfig

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CACHE_DIR = os.path.join(HERE, "cache")


# ---------------------------------------------------------------------------
# 탐지기: 공통 인터페이스  fit(normal_segments) → score(X_full) → 1차원 점수
# ---------------------------------------------------------------------------

class EnergyBaseline:
    """스핀들 센서(sensor1)의 rms_db 열을 그대로 점수로 쓴다."""
    name = "energy_spindle"
    min_segment_len = 1          # 학습·채점에 필요한 최소 연속 프레임 수

    def __init__(self, columns: Sequence[str]):
        self.col = list(columns).index("spindle__rms_db")

    def fit(self, segments: Sequence[np.ndarray]) -> "EnergyBaseline":
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.col].astype(float)


class FlatnessBaseline:
    """스핀들 센서 스펙트럼 평탄도의 음수. 채터는 특정 주파수가 튀는 tonal 소리라 평탄도가 낮아진다.

    학습 없음. "채터 = 덜 평탄한 스펙트럼" 이라는 물리적 직관 하나로 만든 베이스라인.
    """
    name = "flatness_spindle"
    min_segment_len = 1

    def __init__(self, columns: Sequence[str]):
        self.col = list(columns).index("spindle__flatness")

    def fit(self, segments: Sequence[np.ndarray]) -> "FlatnessBaseline":
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -X[:, self.col].astype(float)


class IsolationForestDetector:
    name = "iforest"
    min_segment_len = 1

    def __init__(self, seed: int):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1)

    def fit(self, segments: Sequence[np.ndarray]) -> "IsolationForestDetector":
        Xn = np.vstack(segments)
        self.model.fit(self.scaler.fit_transform(Xn))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # sklearn 의 score_samples 는 "정상일수록 큼" → 부호를 뒤집어 "이상일수록 큼"
        return -self.model.score_samples(self.scaler.transform(X))


class DartsForecastingDetector:
    """ts_anomaly_app/utils/anomaly_detection.py 의 모형·스코어러 팩토리를 그대로 재사용한다.

    차이점: 앱은 단일 학습 시계열을 받지만, 여기서는 비절삭 구간이 여러 개라
    Darts 의 다중 시계열 학습(global model)을 쓴다. 점수 계산은 실험 하나를 통째로 넣는다.
    """

    def __init__(self, model_name: str, scorer_name: str, lags: int, window: int, seed: int, columns: Sequence[str]):
        sys.path.insert(0, os.path.dirname(HERE))            # 저장소 루트 → utils 임포트
        from utils.anomaly_detection import create_forecasting_model, create_scorers
        from darts.ad import ForecastingAnomalyModel
        self.name = f"darts_{model_name}_{scorer_name}"
        self.lags, self.window, self.seed = lags, window, seed
        self.columns = list(columns)
        fm = create_forecasting_model(model_name, {"lags": lags, "output_chunk_length": 1})
        scorers = create_scorers([scorer_name], {"window": window, "k": 2})
        self.model = ForecastingAnomalyModel(model=fm, scorer=scorers)
        self._scaler = None
        # 예측은 lags+1 번째 프레임부터 시작하고, 창 기반 스코어러는 잔차가 window 개 이상 필요하다.
        self.min_segment_len = lags + 1 + window + 2

    @staticmethod
    def _to_ts(X: np.ndarray):
        from darts import TimeSeries
        return TimeSeries.from_values(np.asarray(X, dtype=np.float32))

    def fit(self, segments: Sequence[np.ndarray]) -> "DartsForecastingDetector":
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(np.vstack(segments))
        series = [self._to_ts(self._scaler.transform(s)) for s in segments if len(s) >= self.min_segment_len]
        if not series:
            raise ValueError(f"{self.name}: 길이 {self.min_segment_len} 이상인 정상 구간이 없다.")
        self.model.fit(series, allow_model_training=True, start=self.lags + 1, verbose=False)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        ts = self._to_ts(self._scaler.transform(X))
        sc = self.model.score(ts, start=self.lags + 1, verbose=False)
        if isinstance(sc, (list, tuple)):
            sc = sc[0]
        vals = sc.values(copy=False)
        vals = vals.mean(axis=1) if vals.ndim == 2 and vals.shape[1] > 1 else vals.reshape(-1)
        out = np.full(len(X), np.nan)
        idx = np.asarray(sc.time_index)              # RangeIndex → 정수 위치
        out[idx] = vals
        return out


def make_detectors(names: Sequence[str], columns: Sequence[str], seed: int, lags: int, window: int):
    out = []
    for n in names:
        if n == "energy_spindle":
            out.append(EnergyBaseline(columns))
        elif n == "flatness_spindle":
            out.append(FlatnessBaseline(columns))
        elif n == "iforest":
            out.append(IsolationForestDetector(seed))
        elif n.startswith("darts_"):
            _, model_name, scorer_name = n.split("_", 2)
            out.append(DartsForecastingDetector(model_name, scorer_name, lags, window, seed, columns))
        else:
            raise ValueError(f"모르는 탐지기: {n}")
    return out


# ---------------------------------------------------------------------------
# 평가 루틴
# ---------------------------------------------------------------------------

def fit_and_threshold(detector, train_exps: Sequence[Experiment], min_seg_len: int, quantile: float,
                      task: str = "cutting") -> float:
    """학습 실험들의 '정상' 구간으로 학습하고, 그 구간의 점수 분위수를 임계값으로."""
    min_len = max(min_seg_len, getattr(detector, "min_segment_len", 1))
    segs = [s for e in train_exps for s in normal_segments(e, min_len, task)]
    detector.fit(segs)
    train_scores = np.concatenate([detector.score(s) for s in segs])
    return M.threshold_from_normal_quantile(train_scores, quantile)


def evaluate_on(detector, exps: Sequence[Experiment], threshold: float, frame_rate_hz: float,
                task: str = "cutting") -> Tuple[Dict, Dict[str, np.ndarray]]:
    """여러 실험을 채점해 풀링한 지표와 실험별 점수를 돌려준다.

    task="chatter" 이면 y_chatter 가 -1 인 프레임(비절삭·불확실)은 평가에서 뺀다.
    """
    scores, labels, per_exp = [], [], {}
    for e in exps:
        s = detector.score(e.X)
        per_exp[e.name] = s
        y = e.y if task == "cutting" else e.y_chatter
        scores.append(s); labels.append(y)
    s_all, y_all = np.concatenate(scores), np.concatenate(labels)
    ok = ~np.isnan(s_all) & (y_all >= 0)
    res = M.evaluate(s_all[ok], y_all[ok], threshold, frame_rate_hz).as_dict()
    thr_oracle, f1_oracle = M.threshold_max_f1(s_all[ok], y_all[ok])
    res.update({"f1_oracle": f1_oracle, "threshold_oracle": thr_oracle, "n_scored": int(ok.sum()), "n_total": int(len(ok))})
    return res, per_exp


def kfold_groups(exps: Sequence[Experiment], k: int, seed: int) -> List[List[Experiment]]:
    """실험 단위 K겹. 이름순 정렬 후 시드로 섞어 균등 분할 (재현 가능)."""
    rng = np.random.default_rng(seed)
    order = sorted(exps, key=lambda e: e.name)
    idx = rng.permutation(len(order))
    folds = [[] for _ in range(k)]
    for j, i in enumerate(idx):
        folds[j % k].append(order[i])
    return folds


def run_grouped_cv(exps_lab, detector_names, cfg, args) -> List[Dict]:
    rows = []
    folds = kfold_groups(exps_lab, args.folds, args.seed)
    for fi, test_exps in enumerate(folds):
        train_exps = [e for e in exps_lab if e not in test_exps]
        for det in make_detectors(detector_names, exps_lab[0].columns, args.seed, args.lags, args.window):
            t0 = time.time()
            thr = fit_and_threshold(det, train_exps, 2, args.quantile)
            res, _ = evaluate_on(det, test_exps, thr, cfg.frame_rate_hz)
            rows.append({"protocol": "grouped_cv", "fold": fi, "detector": det.name,
                         "test_experiments": ";".join(e.name for e in test_exps),
                         "seconds": round(time.time() - t0, 1), **res})
            print(f"[grouped_cv fold {fi}] {det.name:24s} AUC-PR={res['auc_pr']:.3f} AUC-ROC={res['auc_roc']:.3f} F1={res['f1']:.3f} (oracle {res['f1_oracle']:.3f})", flush=True)
    return rows


def run_lab_to_industry(exps_lab, exps_ind, detector_names, cfg, args) -> List[Dict]:
    rows = []
    for det in make_detectors(detector_names, exps_lab[0].columns, args.seed, args.lags, args.window):
        t0 = time.time()
        thr = fit_and_threshold(det, exps_lab, 2, args.quantile)
        res, per_exp = evaluate_on(det, exps_ind, thr, cfg.frame_rate_hz)
        rows.append({"protocol": "lab_to_industry", "fold": 0, "detector": det.name,
                     "test_experiments": ";".join(e.name for e in exps_ind),
                     "seconds": round(time.time() - t0, 1), **res})
        for name, s in per_exp.items():
            np.savez_compressed(os.path.join(RESULTS_DIR, f"scores_lab2ind_{det.name}_{name.replace('/', '__')}.npz"), scores=s)
        print(f"[lab_to_industry] {det.name:24s} AUC-PR={res['auc_pr']:.3f} AUC-ROC={res['auc_roc']:.3f} F1={res['f1']:.3f} (oracle {res['f1_oracle']:.3f})", flush=True)
    return rows


def run_random_leaky(exps_lab, detector_names, cfg, args, task: str = "cutting") -> List[Dict]:
    """프레임 단위 무작위 분할. 예측 기반 탐지기는 연속성이 필요해 제외한다."""
    rows = []
    X = np.vstack([e.X for e in exps_lab])
    y = np.concatenate([e.y if task == "cutting" else e.y_chatter for e in exps_lab])
    keep = y >= 0
    X, y = X[keep], y[keep]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y)); n_tr = int(0.7 * len(y))
    tr, te = perm[:n_tr], perm[n_tr:]
    proto = "random_leaky" if task == "cutting" else "chatter_random_leaky"
    for name in [n for n in detector_names if not n.startswith("darts_")]:
        det = make_detectors([name], exps_lab[0].columns, args.seed, args.lags, args.window)[0]
        t0 = time.time()
        det.fit([X[tr][y[tr] == 0]])
        thr = M.threshold_from_normal_quantile(det.score(X[tr][y[tr] == 0]), args.quantile)
        s = det.score(X[te])
        res = M.evaluate(s, y[te], thr, cfg.frame_rate_hz).as_dict()
        thr_o, f1_o = M.threshold_max_f1(s, y[te])
        res.update({"f1_oracle": f1_o, "threshold_oracle": thr_o, "n_scored": int(len(te)), "n_total": int(len(te))})
        rows.append({"protocol": proto, "fold": 0, "detector": det.name,
                     "test_experiments": "(frames shuffled across all IMI experiments)",
                     "seconds": round(time.time() - t0, 1), **res})
        print(f"[{proto}] {det.name:24s} AUC-PR={res['auc_pr']:.3f} AUC-ROC={res['auc_roc']:.3f} F1={res['f1']:.3f}", flush=True)
    return rows


def run_chatter_grouped_cv(exps_lab, detector_names, cfg, args) -> List[Dict]:
    """채터 과제: 정상 절삭 프레임으로 학습 → 다른 실험의 절삭 프레임에서 채터(1) vs 정상(0) 구분.

    채터 절삭이 하나도 없는 실험만 테스트 폴드에 들어가면 AUC 가 정의되지 않으므로,
    폴드 구성 뒤 테스트 폴드에 양성·음성이 모두 있는지 확인한다.
    """
    rows = []
    usable = [e for e in exps_lab if e.y_chatter is not None and (e.y_chatter == 0).sum() > 0]
    folds = kfold_groups(usable, args.folds, args.seed)
    for fi, test_exps in enumerate(folds):
        train_exps = [e for e in usable if e not in test_exps]
        y_te = np.concatenate([e.y_chatter for e in test_exps])
        if (y_te == 1).sum() == 0 or (y_te == 0).sum() == 0:
            print(f"[chatter_grouped_cv fold {fi}] 테스트 폴드에 채터 또는 정상 절삭이 없어 건너뜀", flush=True)
            continue
        for det in make_detectors(detector_names, usable[0].columns, args.seed, args.lags, args.window):
            t0 = time.time()
            try:
                thr = fit_and_threshold(det, train_exps, 2, args.quantile, task="chatter")
            except ValueError as err:
                print(f"[chatter_grouped_cv fold {fi}] {det.name}: {err}", flush=True)
                continue
            res, _ = evaluate_on(det, test_exps, thr, cfg.frame_rate_hz, task="chatter")
            rows.append({"protocol": "chatter_grouped_cv", "fold": fi, "detector": det.name,
                         "test_experiments": ";".join(e.name for e in test_exps),
                         "seconds": round(time.time() - t0, 1), **res})
            print(f"[chatter_grouped_cv fold {fi}] {det.name:24s} AUC-PR={res['auc_pr']:.3f} AUC-ROC={res['auc_roc']:.3f} F1={res['f1']:.3f} (oracle {res['f1_oracle']:.3f})", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 요약 표
# ---------------------------------------------------------------------------

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["auc_pr", "auc_roc", "f1", "f1_oracle", "event_recall", "false_alarms_per_min"]
    g = df.groupby(["protocol", "detector"])[keys]
    mean, std, n = g.mean(), g.std(ddof=0), g.size()
    out = pd.DataFrame(index=mean.index)
    for k in keys:
        out[k] = [f"{m:.3f} ± {s:.3f}" if cnt > 1 else f"{m:.3f}" for m, s, cnt in zip(mean[k], std[k], n)]
    out["n_folds"] = n
    return out.reset_index()


def environment_info(args) -> Dict:
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(HERE), text=True).strip()
    except Exception:
        commit = None
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
            "platform": platform.platform(), "git_commit": commit,
            "packages": {m: ver(m) for m in ("numpy", "pandas", "sklearn", "darts", "soundfile")},
            "args": vars(args)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("MT_CUTTING_ROOT", "C:/Users/109so/data/mt_cutting_dataset"))
    p.add_argument("--frame-ms", type=float, default=50.0)
    p.add_argument("--hop-ms", type=float, default=50.0)
    p.add_argument("--detectors", nargs="+", default=["energy_spindle", "iforest", "darts_linearregression_norm", "darts_linearregression_kmeans"])
    p.add_argument("--chatter-detectors", nargs="+", default=["energy_spindle", "flatness_spindle", "iforest", "darts_linearregression_norm", "darts_linearregression_kmeans"])
    p.add_argument("--protocols", nargs="+", default=["grouped_cv", "lab_to_industry", "random_leaky", "chatter_grouped_cv", "chatter_random_leaky"])
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--lags", type=int, default=10)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="실험 6개·탐지기 2개로 빠른 점검")
    p.add_argument("--max-experiments", type=int, default=None)
    p.add_argument("--per-recording-norm", action="store_true",
                   help="각 녹음(실험)의 특징을 자기 자신의 중앙값·IQR 로 정규화한 뒤 탐지 (라벨 불필요, 녹음 조건 차이 보정 가설 검증용)")
    p.add_argument("--tag", default="", help="결과 파일 이름 접미사 (예: _pernorm)")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):          # Windows 콘솔(cp949)에서 한글·± 깨짐 방지
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.quick:
        args.max_experiments = args.max_experiments or 6
        args.detectors = ["energy_spindle", "iforest"]
        args.chatter_detectors = ["energy_spindle", "flatness_spindle", "iforest"]
        args.folds = 3

    cfg = FrameConfig(sample_rate=48_000, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    exps = discover_experiments(args.data_root)
    exps_lab = [e for e in exps if e.domain == "IMI"]
    exps_ind = [e for e in exps if e.domain == "KRPM"]
    if args.max_experiments:
        exps_lab = exps_lab[: args.max_experiments]
    print(f"실험실 {len(exps_lab)}개, 현장 {len(exps_ind)}개. 특징 추출 중 (캐시: {CACHE_DIR})", flush=True)
    for e in exps_lab + exps_ind:
        featurize(e, cfg, CACHE_DIR)
        if args.per_recording_norm:
            # 라벨을 쓰지 않는 정규화: 녹음 전체의 중앙값·IQR. 절삭 비율이 다르면 기준도 조금 달라진다(리포트에 명시).
            med = np.median(e.X, axis=0)
            iqr = np.subtract(*np.percentile(e.X, [75, 25], axis=0))
            e.X = (e.X - med) / np.where(iqr > 1e-9, iqr, 1.0)
    attach_chatter_labels(exps_lab, args.data_root, cfg)
    summ = summarize(exps_lab + exps_ind)
    summ.to_csv(os.path.join(RESULTS_DIR, "dataset_summary.csv"), index=False)
    print(summ.to_string(index=False), flush=True)
    chat = chatter_summary(exps_lab)
    if len(chat):
        chat.to_csv(os.path.join(RESULTS_DIR, "chatter_summary.csv"), index=False)
        print("\n채터 라벨 요약:\n" + chat.to_string(index=False), flush=True)

    rows: List[Dict] = []
    if "grouped_cv" in args.protocols:
        rows += run_grouped_cv(exps_lab, args.detectors, cfg, args)
    if "lab_to_industry" in args.protocols and exps_ind:
        rows += run_lab_to_industry(exps_lab, exps_ind, args.detectors, cfg, args)
    if "random_leaky" in args.protocols:
        rows += run_random_leaky(exps_lab, args.detectors, cfg, args)
    if "chatter_grouped_cv" in args.protocols and len(chat):
        rows += run_chatter_grouped_cv(exps_lab, args.chatter_detectors, cfg, args)
    if "chatter_random_leaky" in args.protocols and len(chat):
        rows += run_random_leaky(exps_lab, args.chatter_detectors, cfg, args, task="chatter")

    tag = args.tag
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"results_long{tag}.csv"), index=False)
    summ_tbl = summary_table(df)
    summ_tbl.to_csv(os.path.join(RESULTS_DIR, f"results_summary{tag}.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"results_summary{tag}.md"), "w", encoding="utf-8") as f:
        f.write(summ_tbl.to_markdown(index=False))
    with open(os.path.join(RESULTS_DIR, f"run_meta{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(environment_info(args), f, ensure_ascii=False, indent=2)
    print("\n" + summ_tbl.to_string(index=False))
    print(f"\n저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
