"""benchmark/metrics.py 가 scikit-learn 과 같은 값을 내는지 확인한다.

실행: (저장소 루트에서)  python -m pytest -q
"""

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from benchmark import metrics as M


def _random_case(seed: int, n: int = 300, pos_ratio: float = 0.15, n_ties: int = 0):
    """양성이 드문 임의의 (score, label) 쌍. n_ties > 0 이면 동점을 일부러 만든다."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_ratio).astype(int)
    s = rng.normal(0, 1, n) + 1.5 * y          # 양성이 평균적으로 더 높은 점수
    if n_ties:
        s = np.round(s, 1)                      # 소수 첫째 자리로 반올림 → 동점 다수
    return s, y


# ---------------------------------------------------------------------------
# 임계값 지표
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_precision_recall_f1_matches_sklearn(seed):
    s, y = _random_case(seed)
    thr = float(np.quantile(s, 0.8))
    pred = (s >= thr).astype(int)
    ours = M.precision_recall_f1(s, y, thr)
    assert ours["precision"] == pytest.approx(precision_score(y, pred, zero_division=0))
    assert ours["recall"] == pytest.approx(recall_score(y, pred, zero_division=0))
    assert ours["f1"] == pytest.approx(f1_score(y, pred, zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    assert (ours["tn"], ours["fp"], ours["fn"], ours["tp"]) == tuple(cm.ravel())


def test_threshold_is_inclusive():
    s = np.array([0.1, 0.5, 0.9])
    y = np.array([0, 1, 1])
    c = M.confusion_counts(s, y, threshold=0.5)
    assert c["tp"] == 2 and c["fp"] == 0


def test_zero_division_gives_zero():
    s = np.array([0.0, 0.0, 0.0])
    y = np.array([1, 0, 1])
    r = M.precision_recall_f1(s, y, threshold=10.0)   # 아무것도 양성으로 판정 안 함
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,n_ties", [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)])
def test_auc_roc_matches_sklearn(seed, n_ties):
    s, y = _random_case(seed, n_ties=n_ties)
    assert M.auc_roc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_auc_roc_two_definitions_agree(seed):
    s, y = _random_case(seed, n=120, n_ties=1)
    assert M.auc_roc(s, y) == pytest.approx(M.auc_roc_by_ranking(s, y), abs=1e-12)


def test_roc_curve_matches_sklearn_points():
    s, y = _random_case(7, n=80, n_ties=1)
    fpr_sk, tpr_sk, _ = roc_curve(y, s, drop_intermediate=False)
    fpr, tpr, _ = M.roc_curve_points(s, y)
    assert np.allclose(fpr, fpr_sk) and np.allclose(tpr, tpr_sk)


def test_auc_roc_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert M.auc_roc(np.array([1, 2, 3, 4, 5, 6]), y) == pytest.approx(1.0)
    assert M.auc_roc(np.array([6, 5, 4, 3, 2, 1]), y) == pytest.approx(0.0)
    assert M.auc_roc(np.ones(6), y) == pytest.approx(0.5)      # 전부 동점 → 0.5


def test_auc_roc_requires_both_classes():
    with pytest.raises(ValueError):
        M.auc_roc(np.array([0.1, 0.2]), np.array([1, 1]))


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,n_ties", [(0, 0), (1, 1), (2, 1), (3, 0), (4, 1)])
def test_average_precision_matches_sklearn(seed, n_ties):
    s, y = _random_case(seed, n_ties=n_ties)
    assert M.average_precision(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_average_precision_rare_positive_is_harsher_than_auc_roc():
    """양성이 드물면 같은 순위 품질에서도 AP 가 AUC-ROC 보다 낮다 (리포트의 핵심 논점)."""
    s, y = _random_case(11, n=2000, pos_ratio=0.02)
    assert M.average_precision(s, y) < M.auc_roc(s, y)


# ---------------------------------------------------------------------------
# 이벤트 지표
# ---------------------------------------------------------------------------

def test_label_runs():
    assert M.label_runs(np.array([0, 1, 1, 0, 1, 0, 0, 1])) == [(1, 3), (4, 5), (7, 8)]
    assert M.label_runs(np.array([1, 1, 1])) == [(0, 3)]
    assert M.label_runs(np.array([0, 0])) == []


def test_event_recall_counts_runs_not_frames():
    y = np.array([0, 1, 1, 1, 1, 0, 1, 0])          # 이벤트 2개: 긴 것, 짧은 것
    s = np.array([0, 9, 9, 9, 9, 0, 0, 0])          # 긴 것만 탐지
    r = M.event_recall(s, y, threshold=1.0)
    assert r["n_events"] == 2 and r["n_detected"] == 1 and r["event_recall"] == 0.5
    # 프레임 recall 은 4/5 = 0.8 로 더 낙관적
    assert M.precision_recall_f1(s, y, 1.0)["recall"] == pytest.approx(0.8)


def test_false_alarms_per_minute():
    y = np.zeros(120, dtype=int)                    # 정상 120프레임, 1 Hz → 2분
    s = np.zeros(120); s[10:13] = 5; s[50] = 5      # 오경보 덩어리 2개
    r = M.false_alarms_per_minute(s, y, threshold=1.0, frame_rate_hz=1.0)
    assert r["n_false_alarm_runs"] == 2 and r["normal_minutes"] == pytest.approx(2.0)
    assert r["false_alarms_per_min"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 임계값 선택
# ---------------------------------------------------------------------------

def test_threshold_from_normal_quantile():
    normal = np.arange(100, dtype=float)
    assert M.threshold_from_normal_quantile(normal, 0.99) == pytest.approx(np.quantile(normal, 0.99))


def test_threshold_max_f1_is_optimistic_on_same_data():
    """테스트 라벨로 고른 임계값의 F1 은 정상 분위수 임계값의 F1 보다 작을 수 없다."""
    s, y = _random_case(5, n=500)
    thr_q = M.threshold_from_normal_quantile(s[y == 0], 0.95)
    f1_q = M.precision_recall_f1(s, y, thr_q)["f1"]
    _, f1_best = M.threshold_max_f1(s, y)
    assert f1_best >= f1_q - 1e-12


def test_evaluate_bundle_runs():
    s, y = _random_case(3)
    r = M.evaluate(s, y, threshold=float(np.quantile(s, 0.9)), frame_rate_hz=20.0)
    d = r.as_dict()
    assert 0 <= d["auc_roc"] <= 1 and 0 <= d["auc_pr"] <= 1
    assert d["tp"] + d["fp"] + d["fn"] + d["tn"] == d["n"]
