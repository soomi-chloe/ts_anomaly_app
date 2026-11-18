"""벤치마크 결과를 해석하기 위한 보조 분석. 리포트의 "왜 그런가" 절에 쓰는 숫자를 만든다.

1. 실험실 vs 현장: 비절삭(idle)·절삭 프레임의 센서별 에너지 분포 → 전이 실패의 원인 후보
2. 센서별 에너지 베이스라인의 lab→industry AUC → 어느 센서가 그나마 옮겨 가는가
3. 채터 라벨과 절삭 방향·RPM 의 교차표 → 라벨이 조건과 얽혀 있는지(교란)
4. 평탄도 베이스라인의 부호 → 가설이 틀린 방향인지

실행: python -m benchmark.analyze --data-root ...
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

from . import metrics as M
from .dataset import attach_chatter_labels, discover_experiments, featurize
from .features import FEATURE_NAMES, FrameConfig
from .run_benchmark import CACHE_DIR, RESULTS_DIR


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("MT_CUTTING_ROOT", "C:/Users/109so/data/mt_cutting_dataset"))
    args = p.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cfg = FrameConfig()
    exps = discover_experiments(args.data_root)
    for e in exps:
        featurize(e, cfg, CACHE_DIR)
    lab = [e for e in exps if e.domain == "IMI"]
    ind = [e for e in exps if e.domain == "KRPM"]
    attach_chatter_labels(lab, args.data_root, cfg)
    cols = lab[0].columns
    out: List[str] = ["# 보조 분석\n"]

    # 1. 도메인별 에너지 분포 ---------------------------------------------------
    out.append("## 1. 비절삭/절삭 프레임의 센서 에너지(rms_db) 분포: 실험실 vs 현장\n")
    rows = []
    for dom, group in (("IMI(실험실)", lab), ("KRPM(현장)", ind)):
        X = np.vstack([e.X for e in group]); y = np.concatenate([e.y for e in group])
        for sensor in ("mic", "spindle", "base"):
            c = cols.index(f"{sensor}__rms_db")
            idle, cut = X[y == 0, c], X[y == 1, c]
            rows.append({"domain": dom, "sensor": sensor,
                         "idle_p50": q(idle, .5), "idle_p99": q(idle, .99),
                         "cut_p50": q(cut, .5), "cut_p01": q(cut, .01),
                         "gap_cutp01_minus_idlep99": q(cut, .01) - q(idle, .99)})
    t1 = pd.DataFrame(rows).round(1)
    out.append(t1.to_markdown(index=False) + "\n")
    out.append("gap 가 양수면 '절삭의 가장 조용한 1%' 가 '비절삭의 가장 시끄러운 1%' 보다 크다 → 에너지 임계값 하나로 분리 가능.\n")

    # 2. 센서별 에너지 베이스라인 lab→industry -----------------------------------
    out.append("## 2. 센서별 에너지 베이스라인: 실험실 임계값을 현장에 적용\n")
    rows = []
    X_lab = np.vstack([e.X for e in lab]); y_lab = np.concatenate([e.y for e in lab])
    X_ind = np.vstack([e.X for e in ind]); y_ind = np.concatenate([e.y for e in ind])
    for sensor in ("mic", "spindle", "base"):
        c = cols.index(f"{sensor}__rms_db")
        thr = M.threshold_from_normal_quantile(X_lab[y_lab == 0, c], 0.99)
        r_lab = M.evaluate(X_lab[:, c], y_lab, thr, cfg.frame_rate_hz)
        r_ind = M.evaluate(X_ind[:, c], y_ind, thr, cfg.frame_rate_hz)
        thr_ind = M.threshold_from_normal_quantile(X_ind[y_ind == 0, c], 0.99)
        r_ind_own = M.evaluate(X_ind[:, c], y_ind, thr_ind, cfg.frame_rate_hz)
        rows.append({"sensor": sensor, "lab_auc_pr(in-sample)": r_lab.auc_pr, "ind_auc_pr": r_ind.auc_pr,
                     "ind_auc_roc": r_ind.auc_roc, "ind_f1_lab_thr": r_ind.f1,
                     "ind_f1_own_thr(oracle-free, 현장 idle 로 재설정)": r_ind_own.f1,
                     "lab_thr_db": thr, "ind_thr_db": thr_ind})
    t2 = pd.DataFrame(rows).round(3)
    out.append(t2.to_markdown(index=False) + "\n")
    out.append("AUC 는 임계값과 무관한 순위 품질, F1 은 임계값에 좌우된다. 'own_thr' 열은 현장의 비절삭 소리로 임계값만 다시 잡았을 때다.\n")

    # 3. 채터 라벨 vs 조건 교차표 ------------------------------------------------
    out.append("## 3. 채터 라벨(두 작업자 모두 2)과 절삭 조건의 교차표 (실험실)\n")
    metas = []
    for e in lab:
        if e.cut_meta is None:
            continue
        t = e.cut_meta.copy(); t["experiment"] = e.name
        t["label"] = np.where((t.op1 == 2) & (t.op2 == 2), "chatter", np.where((t.op1 == 0) & (t.op2 == 0), "normal", "excluded"))
        metas.append(t)
    meta = pd.concat(metas, ignore_index=True)
    out.append("전체 절삭 수: " + str(len(meta)) + " / 라벨 분포: " + meta.label.value_counts().to_dict().__repr__() + "\n")
    out.append("### 방향(up=상향, do=하향 절삭) × 라벨\n")
    out.append(pd.crosstab(meta.direction, meta.label).to_markdown() + "\n")
    out.append("### RPM × 라벨\n")
    out.append(pd.crosstab(meta.rpm.fillna(-1).astype(int), meta.label).to_markdown() + "\n")
    out.append("### 실험 × 라벨\n")
    out.append(pd.crosstab(meta.experiment, meta.label).to_markdown() + "\n")
    op_agree = float(((meta.op1 == meta.op2)).mean())
    out.append(f"두 작업자 판정 일치율: {op_agree:.3f} (불확실(1) 포함)\n")

    # 4. 채터 프레임의 특징 방향 ------------------------------------------------
    out.append("## 4. 채터 vs 정상 절삭 프레임에서 스핀들 특징의 중앙값 (가설 부호 점검)\n")
    Xc = np.vstack([e.X for e in lab if e.y_chatter is not None]); yc = np.concatenate([e.y_chatter for e in lab if e.y_chatter is not None])
    rows = []
    for f in FEATURE_NAMES:
        c = cols.index(f"spindle__{f}")
        a, b = Xc[yc == 0, c], Xc[yc == 1, c]
        auc = M.auc_roc(np.r_[a, b], np.r_[np.zeros(len(a), int), np.ones(len(b), int)])
        rows.append({"feature": f"spindle__{f}", "normal_cut_p50": q(a, .5), "chatter_p50": q(b, .5), "auc_roc_as_score": auc})
    t4 = pd.DataFrame(rows).round(3).sort_values("auc_roc_as_score", ascending=False)
    out.append(t4.to_markdown(index=False) + "\n")
    out.append("auc_roc_as_score 가 0.5 에서 멀수록 그 특징 하나로 채터를 가를 수 있다. 0.5 보다 작으면 부호가 반대라는 뜻(값이 클수록 정상).\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
