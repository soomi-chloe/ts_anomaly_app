"""샘플 다변량 시계열 + 라벨 CSV 생성 스크립트.

3개의 센서(temperature, vibration, pressure)를 가진 30일치 시간단위 시계열을 생성하고,
인위적으로 5종류의 이상 구간을 삽입한다.

생성물:
- multivariate_sensor.csv : timestamp + 3개 센서 컬럼
- multivariate_sensor_labels.csv : timestamp + label(0/1) (별도 라벨 파일)
- multivariate_sensor_with_label.csv : 위 두 개 결합 (한 파일로 업로드 후 컬럼 지정 시나리오용)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(seed: int = 42):
    rng = np.random.default_rng(seed)

    # 30일 × 24시간 = 720 시점
    n = 720
    start = pd.Timestamp("2025-01-01 00:00:00")
    time_index = pd.date_range(start=start, periods=n, freq="h")

    t = np.arange(n)

    # ---- 정상 패턴 ----
    # temperature: 일주기 + 주간 약한 상승 트렌드
    temp_base = (
        20.0
        + 4.0 * np.sin(2 * np.pi * t / 24.0)
        + 1.5 * np.sin(2 * np.pi * t / (24 * 7))
        + 0.002 * t
    )
    temp_noise = rng.normal(0, 0.4, n)
    temperature = temp_base + temp_noise

    # vibration: 약한 일주기 + 백색잡음
    vib_base = 0.5 + 0.05 * np.sin(2 * np.pi * t / 12.0)
    vib_noise = rng.normal(0, 0.03, n)
    vibration = np.clip(vib_base + vib_noise, 0, None)

    # pressure: 느린 트렌드 + 24h 주기
    pres_base = 101.3 + 0.6 * np.sin(2 * np.pi * t / 24.0) + 0.0005 * t
    pres_noise = rng.normal(0, 0.15, n)
    pressure = pres_base + pres_noise

    label = np.zeros(n, dtype=int)

    # ---- 이상 구간 삽입 ----
    # (1) 단발성 스파이크 (temperature) — 110시 부근
    idx = 110
    temperature[idx:idx + 2] += 8.0
    label[idx:idx + 2] = 1

    # (2) 단발성 음의 스파이크 (pressure) — 220시 부근
    idx = 220
    pressure[idx:idx + 3] -= 3.5
    label[idx:idx + 3] = 1

    # (3) 점진적 드리프트 (vibration) — 320~360
    drift_start, drift_end = 320, 360
    drift = np.linspace(0, 0.6, drift_end - drift_start)
    vibration[drift_start:drift_end] += drift
    label[drift_start:drift_end] = 1

    # (4) 분산 증가 (모든 센서, 짧은 구간) — 470~485
    a, b = 470, 485
    temperature[a:b] += rng.normal(0, 3.0, b - a)
    vibration[a:b] += rng.normal(0, 0.25, b - a)
    pressure[a:b] += rng.normal(0, 1.5, b - a)
    label[a:b] = 1

    # (5) 레벨 시프트 (temperature) — 600~620 갑작스러운 +6도 점프
    a, b = 600, 620
    temperature[a:b] += 6.0
    label[a:b] = 1

    # (6) 다변량 동시 이상 — 680~688 (temp ↑, vib ↑, pres ↓)
    a, b = 680, 688
    temperature[a:b] += 5.0
    vibration[a:b] += 0.4
    pressure[a:b] -= 2.0
    label[a:b] = 1

    df = pd.DataFrame({
        "timestamp": time_index,
        "temperature": np.round(temperature, 3),
        "vibration": np.round(vibration, 4),
        "pressure": np.round(pressure, 3),
    })
    df_labels = pd.DataFrame({
        "timestamp": time_index,
        "label": label,
    })
    df_combined = df.copy()
    df_combined["label"] = label

    # ---- 저장 ----
    df.to_csv(os.path.join(OUT_DIR, "multivariate_sensor.csv"), index=False)
    df_labels.to_csv(os.path.join(OUT_DIR, "multivariate_sensor_labels.csv"), index=False)
    df_combined.to_csv(os.path.join(OUT_DIR, "multivariate_sensor_with_label.csv"), index=False)

    print(f"Generated 3 sample CSVs in {OUT_DIR}")
    print(f"  - multivariate_sensor.csv       (rows={len(df)})")
    print(f"  - multivariate_sensor_labels.csv (rows={len(df_labels)}, positives={int(label.sum())})")
    print(f"  - multivariate_sensor_with_label.csv (rows={len(df_combined)})")
    print(f"\nTotal anomaly ratio: {100*label.mean():.2f}%")


if __name__ == "__main__":
    main()
