"""purduelamm/mt_cutting_dataset 로더.

데이터셋 구조 (README 기준)
---------------------------
<root>/IMI/Exp*/      : 실험실 (Purdue IN-MaC, Hurco VM20i 3축 밀링). 18개 실험 폴더.
<root>/KRPM/...       : 산업 현장 (Kirby Risk Precision Machining, Mazak HCN-6000 4축). 1개 폴더.
각 폴더:
    *_sensor0.wav     : 작업장 환경 마이크
    *_sensor1.wav     : 스핀들 근처 내부음 센서(청진기 + 마이크)
    *_sensor2.wav     : 기계 베이스 내부음 센서
    cutting.csv       : 절삭 구간 라벨. 열 = start, end (초). 전문가가 영상+소리로 표기.
    *.mp4             : (선택) 영상. 벤치마크에서는 쓰지 않는다.
<root>/IMI/labeling_all_details.xlsx : 절삭 조건·채터 라벨 (실험실만)

라벨 변환 규칙
--------------
프레임의 "중앙 시각"이 어떤 절삭 구간 [start, end] 안에 들어가면 1(절삭), 아니면 0(비절삭).
프레임 경계가 구간 경계와 겹치는 경우는 중앙 기준으로 한쪽에 귀속시킨다.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import FrameConfig, multi_sensor_features

SENSOR_NAMES = ("mic", "spindle", "base")          # sensor0, sensor1, sensor2


@dataclass
class Experiment:
    name: str                 # 예: "IMI/Exp01"
    domain: str               # "IMI"(실험실) 또는 "KRPM"(현장)
    folder: str
    wav_paths: List[str]      # sensor0, sensor1, sensor2 순
    intervals: np.ndarray     # (n_cuts, 2) [start, end] 초
    times: Optional[np.ndarray] = None       # 프레임 시작 시각
    X: Optional[np.ndarray] = None           # (n_frames, n_features)
    y: Optional[np.ndarray] = None           # (n_frames,) 0/1  절삭 여부
    columns: List[str] = field(default_factory=list)
    y_chatter: Optional[np.ndarray] = None   # (n_frames,) 1 채터, 0 정상 절삭, -1 제외(비절삭·불확실·의견 불일치)
    cut_meta: Optional[pd.DataFrame] = None  # 절삭 1회당 조건 (rpm, feed, direction, doc, woc, op1, op2)

    @property
    def duration_sec(self) -> float:
        return float(self.times[-1]) if self.times is not None and len(self.times) else 0.0


def _find_sensor_wavs(folder: str) -> List[str]:
    """IMI 는 ``*_sensor{k}.wav``, KRPM 은 ``*_sound{k}.wav`` 로 이름이 다르다. 둘 다 받는다."""
    paths = []
    for k in range(3):
        cands = sorted(glob.glob(os.path.join(folder, f"*_sensor{k}.wav"))
                       + glob.glob(os.path.join(folder, f"*_sound{k}.wav")))
        if not cands:
            raise FileNotFoundError(f"{folder}: *_sensor{k}.wav / *_sound{k}.wav 없음")
        if len(cands) > 1:
            raise ValueError(f"{folder}: 센서 {k} WAV 가 {len(cands)}개 — 폴더당 1개를 가정한다")
        paths.append(cands[0])
    return paths


def _read_intervals(csv_path: str) -> np.ndarray:
    # cutting.csv 는 UTF-8 BOM 으로 시작한다 → utf-8-sig 로 읽어야 첫 열 이름이 "start" 가 된다
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in df.columns}
    if "start" not in cols or "end" not in cols:
        raise ValueError(f"{csv_path}: start/end 열이 없다. 열 = {list(df.columns)}")
    iv = df[[cols["start"], cols["end"]]].to_numpy(dtype=float)
    iv = iv[np.argsort(iv[:, 0])]
    bad = iv[:, 1] <= iv[:, 0]
    if bad.any():
        raise ValueError(f"{csv_path}: end <= start 인 행 {int(bad.sum())}개")
    return iv


def discover_experiments(root: str, domains: Sequence[str] = ("IMI", "KRPM")) -> List[Experiment]:
    """루트 아래에서 cutting.csv 가 있는 폴더를 모두 찾는다."""
    exps: List[Experiment] = []
    for dom in domains:
        dom_dir = os.path.join(root, dom)
        if not os.path.isdir(dom_dir):
            continue
        for csv_path in sorted(glob.glob(os.path.join(dom_dir, "**", "cutting.csv"), recursive=True)):
            folder = os.path.dirname(csv_path)
            rel = os.path.relpath(folder, root).replace("\\", "/")
            exps.append(Experiment(
                name=rel, domain=dom, folder=folder,
                wav_paths=_find_sensor_wavs(folder),
                intervals=_read_intervals(csv_path),
            ))
    if not exps:
        raise FileNotFoundError(f"{root} 아래에서 cutting.csv 를 찾지 못했다.")
    return exps


def intervals_to_frame_labels(times: np.ndarray, hop_sec: float, intervals: np.ndarray) -> np.ndarray:
    """프레임 중앙 시각이 절삭 구간 안에 있으면 1."""
    mid = np.asarray(times, dtype=float) + hop_sec / 2.0
    y = np.zeros(len(mid), dtype=int)
    for s, e in intervals:
        y |= ((mid >= s) & (mid <= e)).astype(int)
    return y


def featurize(exp: Experiment, cfg: FrameConfig, cache_dir: Optional[str] = None) -> Experiment:
    """WAV → 특징 시계열 + 프레임 라벨. cache_dir 가 있으면 npz 로 캐시."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"{exp.name.replace('/', '__')}_f{cfg.frame_ms:g}_h{cfg.hop_ms:g}.npz"
        cache_path = os.path.join(cache_dir, tag)
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            exp.times, exp.X, exp.y = z["times"], z["X"], z["y"]
            exp.columns = list(z["columns"])
            return exp

    times, X, cols = multi_sensor_features(exp.wav_paths, cfg, SENSOR_NAMES)
    y = intervals_to_frame_labels(times, cfg.hop_len / cfg.sample_rate, exp.intervals)
    exp.times, exp.X, exp.y, exp.columns = times, X, y, cols
    if cache_path:
        np.savez_compressed(cache_path, times=times, X=X, y=y, columns=np.array(cols))
    return exp


def runs_where(mask: np.ndarray) -> List[Tuple[int, int]]:
    """mask 가 True 인 연속 구간 [start, end) 목록."""
    m = np.asarray(mask).astype(int)
    padded = np.r_[0, m, 0]
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def normal_segments(exp: Experiment, min_len: int, task: str = "cutting") -> List[np.ndarray]:
    """'정상' 프레임의 연속 구간들을 (segment_len, n_features) 배열 목록으로.

    task="cutting" : 정상 = 비절삭(y==0)
    task="chatter" : 정상 = 채터 없는 절삭(y_chatter==0)
    예측 기반 이상탐지 모형은 '연속된' 정상 시계열로 학습해야 하므로 구간 단위로 돌려준다.
    min_len 보다 짧은 구간은 버린다 (lags·window 보다 짧으면 학습에 못 쓴다).
    """
    mask = (exp.y == 0) if task == "cutting" else (exp.y_chatter == 0)
    return [exp.X[a:b] for a, b in runs_where(mask) if (b - a) >= min_len]


# ---------------------------------------------------------------------------
# 채터 라벨 (IMI/labeling_all_details.xlsx)
# ---------------------------------------------------------------------------

CHATTER_CODES = {0: "normal", 1: "uncertain", 2: "chatter"}


def load_chatter_table(root: str) -> Dict[str, pd.DataFrame]:
    """엑셀의 실험별 시트 → {"IMI/Exp0-1": DataFrame[start, end, rpm, feed, direction, doc, woc, op1, op2]}.

    시트마다 첫 행이 헤더다. 열 이름은 'start', 'end', 'Chatter (operator 1)', 'Chatter (operator 2)',
    'Spindle speed [RPM]', 'Feedrate [IPM]', 'Cutting direction', 'DOC [in.]', 'WOC [in.]'.
    """
    path = os.path.join(root, "IMI", "labeling_all_details.xlsx")
    if not os.path.exists(path):
        return {}
    x = pd.ExcelFile(path)
    out: Dict[str, pd.DataFrame] = {}
    for sheet in x.sheet_names:
        if not sheet.lower().startswith("exp"):
            continue
        df = x.parse(sheet)
        low = {str(c).strip().lower(): c for c in df.columns}

        def pick(*keys):
            for k in keys:
                for lc, orig in low.items():
                    if lc.startswith(k):
                        return orig
            return None

        c_start, c_end = pick("start"), pick("end")
        c_op1, c_op2 = pick("chatter (operator 1)"), pick("chatter (operator 2)")
        if None in (c_start, c_end, c_op1, c_op2):
            print(f"[chatter] {sheet}: 필요한 열을 찾지 못해 건너뜀 (열={list(df.columns)})")
            continue
        tbl = pd.DataFrame({
            "start": pd.to_numeric(df[c_start], errors="coerce"),
            "end": pd.to_numeric(df[c_end], errors="coerce"),
            "op1": pd.to_numeric(df[c_op1], errors="coerce"),
            "op2": pd.to_numeric(df[c_op2], errors="coerce"),
            "rpm": pd.to_numeric(df[pick("spindle")], errors="coerce") if pick("spindle") else np.nan,
            "feed": pd.to_numeric(df[pick("feedrate")], errors="coerce") if pick("feedrate") else np.nan,
            "direction": df[pick("cutting direction")].astype(str).str.strip().str.lower().str[:2] if pick("cutting direction") else "",
            "doc": pd.to_numeric(df[pick("doc")], errors="coerce") if pick("doc") else np.nan,
            "woc": pd.to_numeric(df[pick("woc")], errors="coerce") if pick("woc") else np.nan,
        }).dropna(subset=["start", "end"])
        out[f"IMI/{sheet}"] = tbl.reset_index(drop=True)
    return out


def chatter_frame_labels(times: np.ndarray, hop_sec: float, tbl: pd.DataFrame) -> np.ndarray:
    """두 작업자가 모두 2 → 1(채터), 모두 0 → 0(정상 절삭), 그 외·비절삭 → -1(제외)."""
    mid = np.asarray(times, dtype=float) + hop_sec / 2.0
    y = np.full(len(mid), -1, dtype=int)
    for r in tbl.itertuples(index=False):
        inside = (mid >= r.start) & (mid <= r.end)
        if r.op1 == 2 and r.op2 == 2:
            y[inside] = 1
        elif r.op1 == 0 and r.op2 == 0:
            y[inside] = 0
        # 1(불확실) 이나 의견 불일치는 -1 로 남긴다
    return y


def attach_chatter_labels(exps: Sequence[Experiment], root: str, cfg: FrameConfig) -> None:
    tables = load_chatter_table(root)
    hop = cfg.hop_len / cfg.sample_rate
    for e in exps:
        tbl = tables.get(e.name)
        if tbl is None or e.times is None:
            continue
        e.cut_meta = tbl
        e.y_chatter = chatter_frame_labels(e.times, hop, tbl)


def chatter_summary(exps: Sequence[Experiment]) -> pd.DataFrame:
    rows = []
    for e in exps:
        if e.cut_meta is None:
            continue
        t = e.cut_meta
        both2 = int(((t.op1 == 2) & (t.op2 == 2)).sum())
        both0 = int(((t.op1 == 0) & (t.op2 == 0)).sum())
        rows.append({
            "experiment": e.name, "n_cuts": len(t), "chatter_cuts": both2, "normal_cuts": both0,
            "excluded_cuts": len(t) - both2 - both0,
            "chatter_frames": int((e.y_chatter == 1).sum()), "normal_frames": int((e.y_chatter == 0).sum()),
            "rpm_set": sorted(set(t.rpm.dropna().astype(int).tolist())),
            "chatter_up_cuts": int(((t.op1 == 2) & (t.op2 == 2) & (t.direction == "up")).sum()),
            "chatter_down_cuts": int(((t.op1 == 2) & (t.op2 == 2) & (t.direction == "do")).sum()),
        })
    return pd.DataFrame(rows)


def summarize(exps: Sequence[Experiment]) -> pd.DataFrame:
    rows = []
    for e in exps:
        n = len(e.y) if e.y is not None else 0
        rows.append({
            "experiment": e.name, "domain": e.domain,
            "duration_s": round(e.duration_sec, 1),
            "n_frames": n,
            "n_cut_intervals": int(len(e.intervals)),
            "cut_frac": round(float(e.y.mean()), 3) if n else np.nan,
        })
    return pd.DataFrame(rows)
