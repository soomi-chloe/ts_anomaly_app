"""이상탐지 평가 지표를 라이브러리 없이 직접 구현한 모듈.

왜 직접 쓰는가
--------------
scikit-learn 의 ``roc_auc_score`` 같은 함수는 한 줄이면 끝나지만, 그 숫자가 무엇을
세고 있는지 설명하지 못하면 결과를 검증할 수 없다. 이 모듈의 모든 함수는
정의에서 출발해 numpy 만으로 계산하고, ``tests/test_metrics.py`` 에서 scikit-learn 과
같은 값이 나오는지 확인한다.

용어
----
score  : 모델이 각 시점에 매긴 이상 점수 (클수록 이상). 1차원 배열.
label  : 각 시점의 정답. 1 = 이상(양성), 0 = 정상(음성).
threshold : 이 값 이상이면 "이상"으로 판정.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. 임계값 기반 지표: 혼동행렬 → precision / recall / F1
# ---------------------------------------------------------------------------

def confusion_counts(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, int]:
    """임계값으로 판정한 뒤 TP / FP / FN / TN 을 센다.

    판정 규칙은 ``score >= threshold`` (등호 포함). scikit-learn 의 ``confusion_matrix`` 는
    판정을 입력으로 받으므로, 비교할 때도 같은 규칙으로 판정한 배열을 넘겨야 한다.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    if s.shape != y.shape:
        raise ValueError(f"scores({s.shape})와 labels({y.shape}) 길이가 다릅니다.")
    pred = (s >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    """precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 조화평균.

    분모가 0이면 0으로 둔다 (scikit-learn 의 ``zero_division=0`` 과 같은 선택).
    """
    c = confusion_counts(scores, labels, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **c}


# ---------------------------------------------------------------------------
# 2. 순위 기반 지표: ROC 곡선과 AUC-ROC
# ---------------------------------------------------------------------------

def roc_curve_points(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가능한 모든 임계값에서 (FPR, TPR) 을 계산한다.

    방법: 점수를 내림차순으로 정렬하고, 위에서부터 한 점씩 "이상"으로 판정해 나가면
    TP 와 FP 가 누적된다. 같은 점수(동점)는 한 번에 처리해야 scikit-learn 과 같은 곡선이 된다.

    반환: fpr, tpr, thresholds (모두 길이가 같은 배열, 원점 (0,0) 포함)
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC 곡선은 양성과 음성이 모두 있어야 정의된다.")

    order = np.argsort(-s, kind="mergesort")      # 내림차순, 안정 정렬
    s_sorted = s[order]
    y_sorted = y[order]

    # 동점 구간의 "마지막" 위치만 남긴다 → 같은 점수는 한 덩어리로 판정
    distinct = np.where(np.diff(s_sorted) != 0)[0]
    cut_idx = np.r_[distinct, len(s_sorted) - 1]

    tps = np.cumsum(y_sorted)[cut_idx]            # 누적 TP
    fps = (1 + cut_idx) - tps                     # 누적 판정 수 - TP = FP
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, s_sorted[cut_idx]]
    return fpr, tpr, thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """사다리꼴 공식으로 곡선 아래 면적. x 는 단조증가여야 한다."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC = ROC 곡선 아래 면적.

    해석: 무작위로 고른 (이상, 정상) 한 쌍에서 이상 쪽 점수가 더 높을 확률.
    동점은 0.5 로 센다. 아래 ``auc_roc_by_ranking`` 과 같은 값이 나와야 한다.
    """
    fpr, tpr, _ = roc_curve_points(scores, labels)
    return auc_trapezoid(fpr, tpr)


def auc_roc_by_ranking(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC 를 정의 그대로 (모든 양성-음성 쌍 비교) 계산한다. 검산용.

    O(n_pos * n_neg) 라서 큰 데이터에는 느리다. 테스트에서 작은 배열로만 쓴다.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("양성과 음성이 모두 있어야 한다.")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# 3. 순위 기반 지표: PR 곡선과 Average Precision (AUC-PR)
# ---------------------------------------------------------------------------

def precision_recall_points(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가능한 모든 임계값에서 (precision, recall).

    ROC 와 같은 누적 방식. scikit-learn 의 ``precision_recall_curve`` 는 recall 이 1 에
    도달한 뒤의 점을 잘라내고, 마지막에 (precision=1, recall=0) 을 덧붙인다. 여기서는
    곡선 자체보다 면적(AP)이 목적이므로 그 관례를 따르지 않고 raw 점만 돌려준다.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("양성이 하나도 없으면 PR 곡선이 정의되지 않는다.")

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    distinct = np.where(np.diff(s_sorted) != 0)[0]
    cut_idx = np.r_[distinct, len(s_sorted) - 1]

    tps = np.cumsum(y_sorted)[cut_idx]
    n_pred = 1 + cut_idx
    precision = tps / n_pred
    recall = tps / n_pos
    thresholds = s_sorted[cut_idx]
    return precision, recall, thresholds


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision (AP) = Σ (R_k − R_{k−1}) · P_k.

    PR 곡선 아래 면적을 사다리꼴이 아니라 "계단"으로 적분한 값. scikit-learn 의
    ``average_precision_score`` 와 같은 정의다. 사다리꼴로 계산하면 낙관적으로 커질 수 있어
    (precision 이 recall 에 대해 단조가 아니기 때문) 이 정의를 쓴다.

    이상탐지처럼 양성이 드문 문제에서는 AUC-ROC 보다 이 값이 훨씬 민감하다:
    AUC-ROC 는 음성이 많으면 FP 가 늘어도 FPR 이 거의 안 움직여서 높게 나온다.
    """
    precision, recall, _ = precision_recall_points(scores, labels)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


# ---------------------------------------------------------------------------
# 4. 이벤트(구간) 단위 지표
# ---------------------------------------------------------------------------

def label_runs(labels: np.ndarray) -> List[Tuple[int, int]]:
    """연속된 1 구간을 [start, end) 인덱스 쌍으로 돌려준다."""
    y = np.asarray(labels).reshape(-1).astype(int)
    if len(y) == 0:
        return []
    padded = np.r_[0, y, 0]
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def event_recall(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    """이벤트 단위 재현율: 정답 구간 중 "한 점이라도" 탐지된 구간의 비율.

    프레임 단위 recall 은 긴 구간이 지배한다. 짧은 절삭 한 번을 통째로 놓쳐도 긴 절삭을
    잘 맞히면 높게 나온다. 현장에서는 "몇 번의 사건 중 몇 번을 잡았나"가 더 중요하다.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    pred = s >= threshold
    runs = label_runs(y)
    if not runs:
        return {"event_recall": float("nan"), "n_events": 0, "n_detected": 0}
    detected = sum(1 for a, b in runs if pred[a:b].any())
    return {"event_recall": detected / len(runs), "n_events": len(runs), "n_detected": detected}


def false_alarms_per_minute(scores: np.ndarray, labels: np.ndarray, threshold: float,
                            frame_rate_hz: float) -> Dict[str, float]:
    """정상 구간에서 울린 "오경보 구간"의 수를 분당으로 환산.

    오경보 하나 = 정상 구간 안에서 연속으로 임계값을 넘은 덩어리 하나.
    프레임 단위 FP 수는 프레임 길이에 따라 달라져 비교가 어렵지만, 분당 오경보 횟수는
    작업자가 체감하는 양과 가깝다.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    pred = (s >= threshold).astype(int)
    fp_frames = pred * (1 - y)
    n_alarm_runs = len(label_runs(fp_frames))
    normal_minutes = float(np.sum(y == 0)) / frame_rate_hz / 60.0
    rate = n_alarm_runs / normal_minutes if normal_minutes > 0 else float("nan")
    return {"false_alarms_per_min": rate, "n_false_alarm_runs": n_alarm_runs,
            "normal_minutes": normal_minutes}


# ---------------------------------------------------------------------------
# 5. 임계값 선택 (테스트 데이터를 보지 않고)
# ---------------------------------------------------------------------------

def threshold_from_normal_quantile(normal_scores: np.ndarray, quantile: float = 0.99) -> float:
    """학습(정상) 점수 분포의 분위수를 임계값으로 쓴다.

    이상탐지의 정석: 정상 데이터만으로 임계값을 정한다. 테스트 라벨로 F1 을 최대화해
    임계값을 고르면 그 F1 은 더 이상 "예측 성능"이 아니다 (테스트에 맞춘 값).
    """
    arr = np.asarray(normal_scores, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        raise ValueError("정상 점수가 비어 있다.")
    return float(np.quantile(arr, quantile))


def threshold_max_f1(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """F1 을 최대로 만드는 임계값과 그 F1. **검증 세트에서만** 쓸 것.

    같은 데이터로 임계값을 고르고 F1 을 보고하면 낙관 편향이 생긴다. 리포트에서는
    이 함수로 얻은 F1 과 분위수 임계값 F1 을 나란히 놓고 차이를 보여 준다.
    """
    precision, recall, thresholds = precision_recall_points(scores, labels)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    k = int(np.argmax(f1))
    return float(thresholds[k]), float(f1[k])


# ---------------------------------------------------------------------------
# 6. 한 번에 계산
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    auc_roc: float
    auc_pr: float
    precision: float
    recall: float
    f1: float
    event_recall: float
    false_alarms_per_min: float
    threshold: float
    n: int
    n_pos: int
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float,
             frame_rate_hz: float) -> EvalResult:
    """순위 지표(AUC-ROC, AUC-PR) + 임계값 지표(P/R/F1) + 이벤트 지표를 한 번에."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    prf = precision_recall_f1(s, y, threshold)
    ev = event_recall(s, y, threshold)
    fa = false_alarms_per_minute(s, y, threshold, frame_rate_hz)
    both = (y.sum() > 0) and (y.sum() < len(y))
    return EvalResult(
        auc_roc=auc_roc(s, y) if both else float("nan"),
        auc_pr=average_precision(s, y) if y.sum() > 0 else float("nan"),
        precision=prf["precision"], recall=prf["recall"], f1=prf["f1"],
        event_recall=ev["event_recall"], false_alarms_per_min=fa["false_alarms_per_min"],
        threshold=float(threshold), n=int(len(y)), n_pos=int(y.sum()),
        tp=prf["tp"], fp=prf["fp"], fn=prf["fn"], tn=prf["tn"],
    )
