"""benchmark/dataset.py 의 라벨 변환 로직 검증 (실제 WAV 없이)."""

import numpy as np
import pandas as pd

from benchmark import dataset as D


def test_intervals_to_frame_labels_uses_frame_midpoint():
    times = np.arange(10) * 0.05            # 0.00, 0.05, ..., 0.45  (20 Hz)
    iv = np.array([[0.12, 0.27]])           # 구간 [0.12, 0.27]
    y = D.intervals_to_frame_labels(times, 0.05, iv)
    # 중앙 시각: 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, ...
    # 0.125, 0.175, 0.225 만 구간 안 (0.275 > 0.27)
    assert y.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_intervals_overlapping_are_unioned():
    times = np.arange(6) * 1.0
    iv = np.array([[0.0, 2.6], [2.0, 4.2]])
    y = D.intervals_to_frame_labels(times, 1.0, iv)
    assert y.tolist() == [1, 1, 1, 1, 0, 0]   # 중앙 0.5,1.5,2.5,3.5 안 / 4.5,5.5 밖


def test_runs_where():
    assert D.runs_where(np.array([0, 1, 1, 0, 0, 1])) == [(1, 3), (5, 6)]
    assert D.runs_where(np.array([1, 1])) == [(0, 2)]
    assert D.runs_where(np.array([0, 0])) == []


def test_normal_segments_min_len_and_task(tmp_path):
    e = D.Experiment(name="t", domain="IMI", folder=str(tmp_path), wav_paths=[], intervals=np.empty((0, 2)))
    e.X = np.arange(10 * 2).reshape(10, 2).astype(float)
    e.y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    segs = D.normal_segments(e, min_len=2, task="cutting")
    assert [len(s) for s in segs] == [3, 2, 2]
    assert D.normal_segments(e, min_len=3, task="cutting")[0].shape == (3, 2)
    e.y_chatter = np.array([-1, -1, -1, 0, 0, -1, -1, 1, -1, -1])
    segs_c = D.normal_segments(e, min_len=1, task="chatter")
    assert len(segs_c) == 1 and segs_c[0].shape == (2, 2)


def test_chatter_frame_labels_agreement_rule():
    times = np.arange(8) * 1.0
    tbl = pd.DataFrame({
        "start": [0.0, 2.0, 4.0, 6.0], "end": [1.9, 3.9, 5.9, 7.9],
        "op1": [2, 0, 1, 2], "op2": [2, 0, 1, 0],
    })
    y = D.chatter_frame_labels(times, 1.0, tbl)
    # 구간1: 둘 다 2 → 1 / 구간2: 둘 다 0 → 0 / 구간3: 불확실 → -1 / 구간4: 불일치 → -1
    assert y.tolist() == [1, 1, 0, 0, -1, -1, -1, -1]
