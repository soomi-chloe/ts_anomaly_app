# Does an anomaly detector trained on idle machine sound find cutting and chatter on real CNC data?

Benchmark of the detectors used in `ts_anomaly_app` on the public CNC milling sound dataset released by
Purdue LAMM ([purduelamm/mt_cutting_dataset](https://github.com/purduelamm/mt_cutting_dataset)).
Run date 2026-11-16, code at commit `5be9fbd` plus this `benchmark/` folder. Environment in `results/run_meta.json`
(Python 3.13, numpy 2.5.5, scikit-learn 1.7.2, darts 0.41.0). Everything below is reproducible with
`python -m benchmark.run_benchmark` (about 4 minutes on a laptop CPU).

## TL;DR

1. **Inside the lab, cutting detection from idle-only training is easy.** With experiment-grouped 6-fold CV,
   IsolationForest reaches AUC-PR 0.995 and detects every cutting event; a single spindle-energy threshold
   already gives AUC-PR 0.88. The forecasting-residual detectors that the app is built around
   (Darts `ForecastingAnomalyModel`) are *worse* than IsolationForest here (AUC-PR 0.91 and 0.84).
2. **Lab-to-industry transfer fails for every detector** (AUC-PR 0.35 to 0.56 on the Mazak recording).
   The forecasting detectors rank the industrial *idle* frames as more anomalous than cutting
   (AUC-ROC below 0.5). All three sensors are about 25 dB quieter at the industrial site, so a threshold
   set in the lab flags nothing (energy baseline F1 = 0.00, event recall 0).
3. **The obvious fix does not work.** Normalizing each recording by its own median and IQR (no labels) leaves the
   site AUC unchanged for the energy baseline, lifts IsolationForest only from 0.41 to 0.49 AUC-PR (floor 0.44),
   and costs up to 0.09 AUC-PR inside the lab. The site's cutting and idle distributions overlap by 20 to 28 dB;
   this is a distribution-shift problem, not a gain problem (section 2.5).
4. **Chatter detection is harder and its labels are confounded.** Treating "normal cutting" as the normal
   class, IsolationForest gets AUC-ROC 0.94 across folds, but chatter cuts are mostly up-milling at high RPM
   and three experiments contribute 86% of chatter frames. A frame-shuffled split inflates AUC-PR from 0.875
   to 0.979, which is the leakage effect this benchmark was designed to expose.
5. **My physics-motivated baseline had the wrong sign.** I expected chatter to lower spectral flatness
   (more tonal). On this data chatter frames are *flatter* and carry more high-frequency energy
   (bandwidth, zero-crossing rate, centroid separate chatter with AUC 0.83 to 0.90; flatness as I defined it
   gives AUC-ROC 0.22). The hypothesis was reasonable and wrong; the data decided.

## 1. Setup

### Data

| Part | Machine | Sensors (48 kHz, mono, 16-bit) | Labels used | Size |
|---|---|---|---|---|
| IMI (lab), 18 recordings | Hurco VM20i 3-axis mill; AISI 4140 / 1018 steel; new and worn tools | `mic` (ambient USB mic), `spindle` (stethoscope near spindle), `base` (stethoscope at machine base) | `cutting.csv` start/end of each cut, labeled by an expert from video + audio; per-cut chatter ratings by two operators (0 normal, 1 uncertain, 2 chatter) in `labeling_all_details.xlsx` | 2,790 s, 583 cuts, 35% of frames are cutting |
| KRPM (industry), 1 recording | Mazak HCN-6000 4-axis mill, cast iron, end-milling + facing + drilling | same three sensors | `cutting.csv` only; the README notes cast iron made exact cut boundaries hard to determine | 130 s, 14 cuts, 44% cutting |

### Features

Each WAV is cut into 50 ms frames (20 Hz). Per sensor and frame: RMS energy (dB), zero-crossing rate, spectral
centroid, bandwidth, flatness, 85% roll-off, and log energy in 8 bands (0–150–300–600–1200–2400–4800–9600–24000 Hz).
Three sensors give a 42-dimensional time series. All of it is numpy FFT in `features.py`, tested on synthetic
sines and noise (`tests/test_features.py`).

### Detectors

| Name | What it is | Training |
|---|---|---|
| `energy_spindle` | spindle RMS (dB) used directly as the score | none |
| `flatness_spindle` | negative spindle spectral flatness (chatter task only) | none |
| `iforest` | IsolationForest (200 trees) on standardized features | normal frames only |
| `darts_linearregression_norm` | the app's `ForecastingAnomalyModel`: linear autoregression (10 lags) on all 42 channels, score = L2 norm of the one-step residual | contiguous normal segments only |
| `darts_linearregression_kmeans` | same forecaster, KMeans scorer (k = 2, window 10) on residuals | same |

"Normal" means non-cutting frames for Task 1 and chatter-free cutting frames for Task 2. The two Darts
detectors reuse the factories in `utils/anomaly_detection.py`, so they are the app's models, not re-implementations.

### Thresholds and metrics

The operating threshold is the 99th percentile of the detector's scores on the *training* normal frames.
Test labels are never used to pick it. For contrast, `f1_oracle` is the best F1 reachable when the threshold
is tuned on the test labels; the gap between `f1` and `f1_oracle` is the price of choosing a threshold honestly.

All metrics are implemented from their definitions in `metrics.py` and checked against scikit-learn in
`tests/test_metrics.py` (43 tests pass): AUC-ROC, AUC-PR (average precision, step-wise), precision / recall / F1
at the threshold, **event recall** (fraction of labeled cutting intervals with at least one detected frame) and
**false alarms per minute** (count of contiguous false-positive runs per minute of normal time).

### Protocols

| Protocol | Train | Test | Purpose |
|---|---|---|---|
| `grouped_cv` | normal frames of 15 lab experiments | all frames of the other 3 (6 folds, seed 42) | generalization to unseen cuts on the same machine |
| `lab_to_industry` | normal frames of all 18 lab experiments | the KRPM recording | generalization to a different machine, material and site |
| `random_leaky` | 70% of lab frames, shuffled | the other 30% | control: neighboring frames land on both sides |
| `chatter_grouped_cv` | chatter-free cutting frames of training experiments | cutting frames of held-out experiments, chatter (both operators = 2) vs normal (both = 0) | Task 2 |
| `chatter_random_leaky` | same, frame-shuffled | | control for Task 2 |

Uncertain ratings and operator disagreements (160 of 583 cuts) are excluded from Task 2. Darts detectors need
contiguous input, so they are absent from the shuffled controls.

## 2. Results

Numbers are mean ± std over folds. Full per-fold rows: `results/results_long.csv`.

### 2.1 Task 1, cutting detection, lab, experiment-grouped CV (6 folds, 33 to 41% positives per fold)

| Detector | AUC-PR | AUC-ROC | F1 @ idle-99% thr | F1 oracle | Event recall | False alarms / min |
|---|---|---|---|---|---|---|
| iforest | **0.995 ± 0.006** | **0.997 ± 0.003** | **0.957 ± 0.046** | 0.979 ± 0.016 | **1.000** | 12.9 ± 8.4 |
| darts_linearregression_norm | 0.914 ± 0.012 | 0.959 ± 0.007 | 0.681 ± 0.175 | 0.860 ± 0.038 | 0.973 ± 0.038 | 26.5 ± 9.3 |
| energy_spindle | 0.880 ± 0.097 | 0.928 ± 0.053 | 0.432 ± 0.193 | 0.850 ± 0.062 | 0.484 ± 0.189 | **0.9 ± 1.8** |
| darts_linearregression_kmeans | 0.842 ± 0.029 | 0.917 ± 0.013 | 0.732 ± 0.065 | 0.790 ± 0.050 | 0.908 ± 0.069 | 35.3 ± 13.3 |

Reading: ranking quality (AUC) and operating-point quality (F1, alarms) tell different stories. The energy
threshold ranks frames almost as well as the forecasting detectors but, at the honest threshold, it misses half
of the cutting events while almost never false-alarming; the forecasting detectors catch nearly every event at
the cost of 26 to 35 isolated false-alarm frames per minute. No smoothing or minimum-duration rule was applied.

### 2.2 Task 1, lab → industry (one recording, 44% positives)

| Detector | AUC-PR | AUC-ROC | F1 @ lab thr | F1 oracle | Event recall | False alarms / min |
|---|---|---|---|---|---|---|
| energy_spindle | 0.557 | 0.618 | 0.000 | 0.638 | 0.000 | 0.0 |
| iforest | 0.405 | 0.463 | 0.613 | 0.620 | 1.000 | 14.8 |
| darts_linearregression_norm | 0.371 | 0.386 | 0.059 | 0.614 | 0.429 | 61.5 |
| darts_linearregression_kmeans | 0.353 | 0.342 | 0.159 | 0.620 | 0.357 | 14.2 |

Every detector is at or below chance (the AUC-PR floor is the prevalence, 0.44). IsolationForest's F1 of 0.61
is not a success: it flags essentially every frame, which with 44% positives yields F1 ≈ 0.61 by construction.
Why it fails is in `results/analysis.md`: at KRPM the idle floor is 15 to 18 dB lower and cutting is 20 to 25 dB
quieter than in the lab in every sensor, so lab statistics simply do not describe the site. The lab energy
threshold is −1.6 dB; the oracle threshold at KRPM is −32 dB. The base-mounted stethoscope transfers best
(energy-only AUC-PR 0.72 vs 0.56 spindle, 0.52 mic), which is consistent with the group's choice of sensor
placement in its own domain-adaptation work.

### 2.3 Task 1, leaky control

| Detector | grouped_cv AUC-PR | random_leaky AUC-PR | grouped_cv AUC-ROC | random_leaky AUC-ROC |
|---|---|---|---|---|
| iforest | 0.995 | 0.993 | 0.997 | 0.996 |
| energy_spindle | 0.880 | 0.821 | 0.928 | 0.906 |

No inflation. Cutting vs idle on one machine is separable enough that leakage cannot make it look better.
The control only bites when the task is hard, which is exactly what Task 2 shows.

### 2.4 Task 2, chatter detection, lab, experiment-grouped CV (5 usable folds)

| Detector | AUC-PR | AUC-ROC | F1 @ thr | F1 oracle | Event recall | False alarms / min |
|---|---|---|---|---|---|---|
| iforest | **0.875 ± 0.087** | **0.939 ± 0.023** | **0.700 ± 0.131** | 0.834 ± 0.089 | 0.872 ± 0.059 | 11.0 ± 4.4 |
| darts_linearregression_kmeans | 0.754 ± 0.166 | 0.792 ± 0.122 | 0.532 ± 0.231 | 0.760 ± 0.178 | 1.000 | 42.2 ± 6.3 |
| darts_linearregression_norm | 0.713 ± 0.215 | 0.803 ± 0.085 | 0.478 ± 0.166 | 0.707 ± 0.234 | 1.000 | 83.7 ± 13.2 |
| energy_spindle | 0.512 ± 0.298 | 0.542 ± 0.213 | 0.165 ± 0.321 | 0.644 ± 0.265 | 0.202 ± 0.281 | 9.9 ± 12.7 |
| flatness_spindle | 0.339 ± 0.291 | 0.218 ± 0.149 | 0.042 ± 0.023 | 0.516 ± 0.293 | 0.335 ± 0.138 | 9.8 ± 6.4 |

Per-fold prevalence of chatter frames: 15%, 75%, 21%, 79%, 13%. The AUC-PR floor therefore moves between 0.13
and 0.79 from fold to fold, which is why the energy baseline's AUC-PR has a standard deviation of 0.30 while its
AUC-ROC sits at chance. The sixth fold had no chatter-free cuts and was skipped.

Leaky control for Task 2 (IsolationForest): AUC-PR 0.979 and AUC-ROC 0.979 on shuffled frames versus
0.875 and 0.939 grouped. **Shuffling adds 0.10 AUC-PR.** A notebook that reports the shuffled number would
overstate the detector by that much.

Which spindle features separate chatter frames (single-feature AUC-ROC, from `analysis.md`): bandwidth 0.90,
zero-crossing rate 0.88, 1.2–2.4 kHz band 0.85, roll-off 0.84, centroid 0.83, flatness 0.79 (so *higher*
flatness means chatter), RMS energy 0.49 (uninformative), 300–1200 Hz bands 0.23 to 0.29 (lower in chatter).

### 2.5 Follow-up: does per-recording normalization fix the transfer?

Hypothesis from 2.2: the site is just quieter, so scaling each recording by its own statistics should recover
the detectors. Test: every recording's 42 features were centered by their own median and scaled by their own
IQR (no labels used) before training and scoring. `python -m benchmark.run_benchmark --per-recording-norm --tag _pernorm`.

| Detector | Protocol | AUC-PR before → after | AUC-ROC before → after | F1 before → after | Event recall before → after | Alarms/min before → after |
|---|---|---|---|---|---|---|
| energy_spindle | lab → industry | 0.557 → 0.557 | 0.618 → 0.618 | 0.000 → 0.365 | 0.000 → 0.786 | 0.0 → 80.8 |
| iforest | lab → industry | 0.405 → 0.489 | 0.463 → 0.624 | 0.613 → 0.033 | 1.000 → 0.214 | 14.8 → 22.3 |
| darts_linearregression_norm | lab → industry | 0.371 → 0.349 | 0.386 → 0.368 | 0.059 → 0.216 | 0.429 → 0.929 | 61.5 → 96.4 |
| iforest | grouped_cv (lab) | 0.995 → 0.978 | 0.997 → 0.988 | 0.957 → 0.780 | 1.000 → 0.920 | 12.9 → 2.0 |
| energy_spindle | grouped_cv (lab) | 0.880 → 0.862 | 0.928 → 0.911 | 0.432 → 0.375 | 0.484 → 0.278 | 0.9 → 1.5 |
| darts_linearregression_norm | grouped_cv (lab) | 0.914 → 0.821 | 0.959 → 0.903 | 0.681 → 0.500 | 0.973 → 0.971 | 26.5 → 26.0 |

**The hypothesis is rejected.** A per-recording monotone rescaling cannot change the ranking of a single feature
inside one recording, so the energy baseline's AUC is identical by construction; only its operating point moves
(from "flags nothing" to "flags most things", 81 alarms per minute). IsolationForest gains 0.08 AUC-PR and stays
near the 0.44 prevalence floor; the forecasting detector stays inverted. Inside the lab the normalization costs
0.02 to 0.09 AUC-PR because the per-recording scale now depends on how much of each recording is cutting
(23% to 58%).

What the numbers say instead: the problem is not a level offset but overlapping distributions. At KRPM the
quietest 1% of cutting frames is 20 to 28 dB *below* the loudest 1% of idle frames in every sensor
(`analysis.md`, table 1), whereas in the lab the two bands nearly touch. Cast-iron facing and drilling on a 4-axis
machine do not sound like steel end-milling on a 3-axis machine relative to their own idle, and nothing learned
from the lab recordings describes that. Closing the gap needs either labeled site data or a domain-adaptation
method, which is the direction the dataset authors took in their INDIN 2024 paper.

## 3. Where this analysis could be wrong

Each item names the evidence, the likely size of the effect, and what I did or did not do about it.

1. **Label boundaries are soft.** Cuts were labeled by a person watching video; the KRPM README says cast-iron
   cut boundaries were hard to determine. My frame label uses the frame midpoint, so every boundary contributes
   up to one 50 ms frame of unavoidable error on each side. With 583 cuts that is roughly 1,200 ambiguous frames,
   about 2% of lab frames. Not corrected; it caps the achievable F1 and slightly penalizes every detector equally.
2. **"Grouped by experiment" is still one machine, one room, one sensor placement.** The lab CV says nothing
   about new sites; section 2.2 shows the gap is total, not marginal. The honest claim from 2.1 is
   "generalizes to unseen cuts on the same setup", nothing more.
3. **AUC-PR depends on prevalence and the folds have very different prevalence.** Averaging AUC-PR across
   folds whose floors range from 0.13 to 0.79 (Task 2) produces a mean that no single fold has. I report
   AUC-ROC next to it, which is prevalence-free but hides the false-positive volume that matters at low prevalence.
   Pooled-over-folds metrics would be an alternative; I did not compute them.
4. **The threshold rule (99% of training-normal scores) is arbitrary and was chosen before seeing results.**
   F1 moves by up to 0.4 between that rule and the oracle (energy baseline: 0.43 vs 0.85). Reporting only
   `f1_oracle` would be test-set tuning; reporting only `f1` makes the comparison hinge on one percentile.
   Both are shown. No minimum-duration post-processing was applied, so the false-alarm counts include single
   isolated frames that a real system would suppress.
5. **The threshold statistics are in-sample.** The scores used to set the percentile come from the same normal
   frames the detector was fitted on. In-sample residuals are smaller than out-of-sample ones, so the threshold
   is too tight and the test false-alarm rate is higher than 1%. A held-out normal subset would fix this; it
   would also reduce training data. Not done.
6. **Chatter labels are confounded with cutting conditions.** Across the 583 cuts: up-milling has 126 chatter
   vs 57 normal cuts, down-milling 43 vs 197; at 8,000 to 12,000 RPM almost every cut is chatter, at 3,000 to
   5,625 RPM almost none. Three experiments (Exp4-3, Exp4-3-2, Exp3-2) contribute 8,070 of 9,386 chatter frames
   and Exp4-3 contains no normal cut at all. A detector that learns "high-RPM up-milling sounds different"
   would score well without detecting chatter. The data cannot separate these; a matched-condition design could.
7. **Excluding uncertain and disagreeing cuts selects the easy cases.** Operators agreed on 81% of ratings
   including the "uncertain" code. Dropping 160 of 583 cuts removes exactly the borderline chatter that a
   deployed system must handle, so Task 2 numbers are upper bounds.
8. **Features and hyperparameters were fixed a priori and never tuned** (14 features, 10 lags, window 10,
   99th percentile, 200 trees). Good for honesty, but the Darts detectors in particular may be far from their
   best configuration; the comparison with IsolationForest is "as the app ships", not "at their best".
9. **The flatness baseline shows that my physical prior was wrong for this feature definition.** Flatness over
   the full 0–24 kHz band is dominated by the low-frequency spindle harmonics; chatter adds broadband energy
   above 1 kHz and *raises* global flatness. A flatness computed relative to the tooth-passing harmonics would
   test the "tonal chatter" hypothesis properly. Flipping the sign after seeing test labels would itself be a
   form of test-set fitting, so the baseline is reported as designed.
10. **The forecasting detectors were trained on short idle segments.** Segments shorter than lags + window + 3
    frames (23 frames, 1.15 s) are discarded, which removes most inter-cut gaps in the fast experiments; the
    first 11 to 21 frames of every test recording are unscored. Both choices follow from the app's model
    design, not from the data.
11. **n = 1 for the industrial site.** One 130-second recording on one day. The lab-to-industry numbers are an
    anecdote with no confidence interval, strong enough to show the failure mode, not strong enough to rank
    detectors against each other at the site.
12. **Per-recording normalization uses whole-recording statistics**, including cutting frames, so the normalized
    scale depends on each recording's cutting fraction (23% to 58%). It uses no labels, but it is not free of
    label-correlated information, and section 2.5 shows it hurts inside the lab for exactly that reason. A
    deployable version would estimate the idle floor from the first seconds of a recording or from a robust
    lower quantile instead.
13. **One seed.** Fold assignment used seed 42 once. Per-fold numbers would change with another draw; the
    ordering of detectors in 2.1 is consistent across all six folds, the ordering in 2.4 is not (energy wins
    fold 2, loses fold 4).
14. **Multiple comparisons.** Five detectors × five protocols. Picking the best cell after the fact is a bias.
    The question fixed before running was whether the app's Darts detectors hold up; the answer is that a
    simpler IsolationForest beats them on both tasks and nothing transfers to the industrial site without adaptation.
15. **Provenance.** The code was written with AI pair-programming assistance. Every metric was re-derived from
    its definition and tested against scikit-learn; feature extraction was tested on synthetic signals with
    known answers; label conversion was tested on toy arrays. The dataset loader is exercised only by the runs
    themselves.

## 4. What I would do next

- Hold out normal data for the threshold; add a minimum-duration rule and report alarms per minute again.
- Per-recording or per-sensor normalization as a label-free first step toward the site (section 2.5), then a
  proper domain-adaptation baseline in the spirit of the dataset authors' INDIN 2024 work.
- Chatter features relative to the tooth-passing frequency instead of global spectral statistics.
- Repeat fold assignment over several seeds and bootstrap over experiments for confidence intervals.
- Pooled-over-folds metrics alongside the fold means, and prevalence next to every AUC-PR.

## 5. Reproduce

```bash
git clone https://github.com/purduelamm/mt_cutting_dataset  C:/Users/<you>/data/mt_cutting_dataset
pip install -r requirements-dev.txt
python -m pytest -q
python -m benchmark.run_benchmark --data-root C:/Users/<you>/data/mt_cutting_dataset
python -m benchmark.analyze       --data-root C:/Users/<you>/data/mt_cutting_dataset
```

Outputs: `results/results_long.csv` (every fold), `results/results_summary.md`, `results/dataset_summary.csv`,
`results/chatter_summary.csv`, `results/analysis.md`, `results/run_meta.json`.
