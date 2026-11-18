"""benchmark/features.py 를 합성 신호로 검증한다. 정답을 아는 신호를 넣고 특징이 그 답을 내는지 본다."""

import numpy as np
import pytest

from benchmark import features as F


SR = 48_000
CFG = F.FrameConfig(sample_rate=SR, frame_ms=50, hop_ms=50)


def _sine(freq_hz: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * freq_hz * t)


def test_frame_signal_shape_and_times():
    x = np.zeros(SR)                                    # 1초
    frames = F.frame_signal(x, CFG)
    assert frames.shape == (20, 2400)                   # 50 ms × 20 = 1 s
    t = F.frame_times(frames.shape[0], CFG)
    assert t[0] == 0.0 and t[1] == pytest.approx(0.05) and t[-1] == pytest.approx(0.95)


def test_frame_signal_drops_tail():
    x = np.zeros(2400 * 3 + 100)                        # 프레임 3개 + 자투리
    assert F.frame_signal(x, CFG).shape[0] == 3


def test_centroid_tracks_sine_frequency():
    for f0 in (300.0, 2000.0, 8000.0):
        feats = F.frame_features(F.frame_signal(_sine(f0, 0.5), CFG), SR)
        centroid = feats[:, F.FEATURE_NAMES.index("centroid_hz")]
        assert np.allclose(centroid, f0, rtol=0.03)     # 해닝 창 누설 때문에 3% 허용


def test_rms_db_of_known_amplitude():
    amp = 0.5
    feats = F.frame_features(F.frame_signal(_sine(1000.0, 0.5, amp), CFG), SR)
    expected = 20 * np.log10(amp / np.sqrt(2))          # 정현파 RMS = A/√2
    assert np.allclose(feats[:, 0], expected, atol=0.05)


def test_flatness_noise_vs_tone():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, SR // 2)
    tone = _sine(1000.0, 0.5)
    fi = F.FEATURE_NAMES.index("flatness")
    flat_noise = F.frame_features(F.frame_signal(noise, CFG), SR)[:, fi].mean()
    flat_tone = F.frame_features(F.frame_signal(tone, CFG), SR)[:, fi].mean()
    assert flat_noise > 0.5 and flat_tone < 0.05        # 잡음은 평탄, 순음은 뾰족


def test_band_energy_lands_in_right_band():
    feats = F.frame_features(F.frame_signal(_sine(3000.0, 0.3), CFG), SR)
    band_cols = [i for i, n in enumerate(F.FEATURE_NAMES) if n.startswith("band_")]
    bands = feats[:, band_cols].mean(axis=0)
    # 3000 Hz 는 (2400, 4800) 구간 = band_5
    assert int(np.argmax(bands)) == 5


def test_zcr_increases_with_frequency():
    zi = F.FEATURE_NAMES.index("zcr")
    low = F.frame_features(F.frame_signal(_sine(200.0, 0.3), CFG), SR)[:, zi].mean()
    high = F.frame_features(F.frame_signal(_sine(5000.0, 0.3), CFG), SR)[:, zi].mean()
    assert high > low


def test_multi_sensor_concat(tmp_path):
    import soundfile as sf
    p1, p2 = tmp_path / "a_sensor0.wav", tmp_path / "a_sensor1.wav"
    sf.write(p1, _sine(500.0, 1.0), SR)
    sf.write(p2, _sine(4000.0, 1.0), SR)
    t, X, cols = F.multi_sensor_features([str(p1), str(p2)], CFG, ["mic", "spindle"])
    assert X.shape == (20, 2 * len(F.FEATURE_NAMES))
    assert cols[0] == "mic__rms_db" and cols[len(F.FEATURE_NAMES)] == "spindle__rms_db"
    ci = F.FEATURE_NAMES.index("centroid_hz")
    assert X[:, ci].mean() < X[:, len(F.FEATURE_NAMES) + ci].mean()
