"""오디오 파형 → 프레임 단위 특징 시계열.

목적
----
``ts_anomaly_app`` 은 "다변량 시계열 CSV" 를 입력으로 받는다. 절삭음 WAV(48 kHz, 1초에
48,000 점)를 그대로 넣을 수는 없으므로, 짧은 구간(프레임)마다 소리의 성질을 요약한 숫자
몇 개로 바꾼다. 50 ms 프레임이면 1초에 20행, 센서 3개 × 특징 14개 = 42열짜리 시계열이 된다.

모든 특징은 numpy FFT 로 직접 계산한다. librosa 를 쓰면 한 줄이지만, 각 숫자가 무엇을
뜻하는지 설명할 수 있어야 벤치마크 결과를 해석할 수 있다.

프레임 특징 (센서 1개당)
------------------------
rms_db        : 프레임 에너지(dB). 절삭이 시작되면 가장 먼저 움직이는 값. "에너지 임계값 베이스라인"의 재료.
zcr           : 영점 교차율. 고주파 성분이 많을수록 크다.
centroid_hz   : 스펙트럼 무게중심. 소리가 "밝은지(높은지)" 를 나타낸다.
bandwidth_hz  : 무게중심 주변의 퍼짐.
flatness      : 스펙트럼 평탄도(0~1). 1에 가까우면 백색잡음, 0에 가까우면 순음(tonal). 채터는 tonal 쪽.
rolloff_hz    : 에너지의 85% 가 그 아래에 있는 주파수.
band_0..band_7: 8개 주파수 대역의 로그 에너지. 옥타브 비슷하게 나눈다 (아래 BAND_EDGES_HZ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 대역 경계 (Hz). 0~24 kHz 를 저주파는 촘촘하게, 고주파는 넓게.
BAND_EDGES_HZ: Tuple[float, ...] = (0, 150, 300, 600, 1200, 2400, 4800, 9600, 24000)
FEATURE_NAMES: Tuple[str, ...] = (
    "rms_db", "zcr", "centroid_hz", "bandwidth_hz", "flatness", "rolloff_hz",
    *[f"band_{i}" for i in range(len(BAND_EDGES_HZ) - 1)],
)
EPS = 1e-12


@dataclass(frozen=True)
class FrameConfig:
    sample_rate: int = 48_000
    frame_ms: float = 50.0      # 프레임 길이
    hop_ms: float = 50.0        # 프레임 간격 (= 길이면 겹침 없음)

    @property
    def frame_len(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def hop_len(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def frame_rate_hz(self) -> float:
        return 1000.0 / self.hop_ms


def frame_signal(x: np.ndarray, cfg: FrameConfig) -> np.ndarray:
    """1차원 파형을 (n_frames, frame_len) 행렬로 자른다. 끝에 남는 조각은 버린다."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    L, H = cfg.frame_len, cfg.hop_len
    if len(x) < L:
        return np.empty((0, L))
    n_frames = 1 + (len(x) - L) // H
    idx = np.arange(L)[None, :] + H * np.arange(n_frames)[:, None]
    return x[idx]


def frame_times(n_frames: int, cfg: FrameConfig) -> np.ndarray:
    """각 프레임의 시작 시각(초). ``cutting.csv`` 의 start/end 와 같은 기준."""
    return np.arange(n_frames) * cfg.hop_len / cfg.sample_rate


def _power_spectrum(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """해닝 창을 곱하고 rFFT → (n_frames, n_bins) 파워 스펙트럼과 bin 주파수."""
    L = frames.shape[1]
    window = np.hanning(L)
    spec = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spec) ** 2)
    return power, None  # 주파수 축은 sample_rate 가 필요해서 호출자가 만든다


def frame_features(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    """(n_frames, frame_len) → (n_frames, len(FEATURE_NAMES))."""
    if frames.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)))
    L = frames.shape[1]

    # 시간 영역
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(rms + EPS)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    # 주파수 영역
    power, _ = _power_spectrum(frames)
    freqs = np.fft.rfftfreq(L, d=1.0 / sample_rate)
    total = power.sum(axis=1) + EPS
    p_norm = power / total[:, None]                              # 확률분포처럼 정규화

    centroid = (p_norm * freqs[None, :]).sum(axis=1)
    bandwidth = np.sqrt((p_norm * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1))
    # 평탄도 = 기하평균 / 산술평균 (0 Hz bin 제외: DC 는 소리가 아니다)
    p_ac = power[:, 1:] + EPS
    flatness = np.exp(np.mean(np.log(p_ac), axis=1)) / np.mean(p_ac, axis=1)
    cum = np.cumsum(p_norm, axis=1)
    rolloff_idx = np.argmax(cum >= 0.85, axis=1)
    rolloff = freqs[rolloff_idx]

    bands = []
    for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        m = (freqs >= lo) & (freqs < hi)
        bands.append(10.0 * np.log10(power[:, m].sum(axis=1) + EPS))
    bands = np.stack(bands, axis=1)

    out = np.column_stack([rms_db, zcr, centroid, bandwidth, flatness, rolloff, bands])
    assert out.shape[1] == len(FEATURE_NAMES)
    return out


def wav_to_features(path: str, cfg: FrameConfig) -> Tuple[np.ndarray, np.ndarray]:
    """WAV 파일 → (times_sec, features). 스테레오면 채널 평균."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != cfg.sample_rate:
        raise ValueError(f"{path}: 샘플레이트 {sr} ≠ 설정 {cfg.sample_rate}")
    frames = frame_signal(x, cfg)
    return frame_times(frames.shape[0], cfg), frame_features(frames, sr)


def multi_sensor_features(paths: Sequence[str], cfg: FrameConfig,
                          sensor_names: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """센서별 WAV 를 같은 프레임 격자로 특징화하고 열 방향으로 붙인다.

    반환: times, X (n_frames, n_sensors * n_features), 열 이름 목록
    길이가 다르면 가장 짧은 쪽에 맞춘다 (README: 한 폴더의 파일은 동기화되어 같은 길이).
    """
    sensor_names = list(sensor_names or [f"s{i}" for i in range(len(paths))])
    blocks, times = [], None
    for p in paths:
        t, F = wav_to_features(p, cfg)
        blocks.append(F)
        times = t if times is None or len(t) < len(times) else times
    n = min(b.shape[0] for b in blocks)
    X = np.column_stack([b[:n] for b in blocks])
    cols = [f"{s}__{f}" for s in sensor_names for f in FEATURE_NAMES]
    return times[:n], X, cols
