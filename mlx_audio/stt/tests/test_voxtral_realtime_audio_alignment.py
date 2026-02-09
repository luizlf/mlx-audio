from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import numpy as np
import pytest

if "mistral_common.audio" not in sys.modules:
    mistral_pkg = types.ModuleType("mistral_common")
    mistral_pkg.__path__ = []
    mistral_pkg.__spec__ = ModuleSpec("mistral_common", loader=None, is_package=True)
    audio_mod = types.ModuleType("mistral_common.audio")
    audio_mod.__spec__ = ModuleSpec("mistral_common.audio", loader=None)

    def _mel_filter_bank(
        *,
        num_frequency_bins: int,
        num_mel_bins: int,
        min_frequency: float,
        max_frequency: float,
        sampling_rate: int,
    ) -> np.ndarray:
        del min_frequency, max_frequency, sampling_rate
        return np.zeros((num_frequency_bins, num_mel_bins), dtype=np.float32)

    audio_mod.mel_filter_bank = _mel_filter_bank
    sys.modules["mistral_common"] = mistral_pkg
    sys.modules["mistral_common.audio"] = audio_mod

from mlx_audio.stt.models.voxtral_realtime import audio as voxtral_audio
from mlx_audio.stt.models.voxtral_realtime.config import AudioConfig


def _legacy_stft(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    if center:
        pad = n_fft // 2
        audio_padded = np.pad(audio, (pad, pad), mode="reflect")
    else:
        audio_padded = audio

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    if n_frames <= 0:
        raise ValueError("Audio is too short for STFT")

    shape = (n_frames, n_fft)
    strides = (audio_padded.strides[0] * hop_length, audio_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape, strides=strides)
    windowed = frames * window[None, :]
    return np.fft.rfft(windowed, n=n_fft).T


def _legacy_compute_log_mel(audio: np.ndarray, config: AudioConfig, center: bool) -> np.ndarray:
    window = np.hanning(config.window_size)
    stft = _legacy_stft(audio, config.n_fft, config.hop_length, window, center=center)
    magnitudes = np.abs(stft[:, :-1]) ** 2

    mel_filters = _test_mel_filter_bank(
        num_frequency_bins=1 + config.n_fft // 2,
        num_mel_bins=config.num_mel_bins,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=config.sampling_rate,
    )
    mel_spec = mel_filters.T @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))

    if config.global_log_mel_max is not None:
        log_spec_max = float(config.global_log_mel_max)
    else:
        log_spec_max = float(log_spec.max())

    log_spec = np.maximum(log_spec, log_spec_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def _test_mel_filter_bank(
    *,
    num_frequency_bins: int,
    num_mel_bins: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    del min_frequency, max_frequency, sampling_rate
    freq = np.linspace(0.1, 1.0, num_frequency_bins, dtype=np.float32)[:, None]
    mel = np.linspace(0.2, 1.2, num_mel_bins, dtype=np.float32)[None, :]
    return freq * mel


def _audio_config(*, global_log_mel_max: float | None = None) -> AudioConfig:
    return AudioConfig(
        {
            "num_mel_bins": 24,
            "window_size": 400,
            "n_fft": 400,
            "sampling_rate": 16000,
            "hop_length": 160,
            "global_log_mel_max": global_log_mel_max,
        }
    )


def test_compute_log_mel_alignment_center_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voxtral_audio, "mel_filter_bank", _test_mel_filter_bank)
    config = _audio_config()
    audio = np.random.default_rng(0).standard_normal(1600).astype(np.float32)

    expected = _legacy_compute_log_mel(audio, config, center=True)
    out = voxtral_audio.compute_log_mel(audio, config, center=True)

    assert out.dtype == np.float32
    assert out.shape == expected.shape == (config.num_mel_bins, 10)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_compute_log_mel_alignment_center_false_and_global_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voxtral_audio, "mel_filter_bank", _test_mel_filter_bank)
    config = _audio_config(global_log_mel_max=3.0)
    audio = np.random.default_rng(1).standard_normal(1600).astype(np.float32)

    expected = _legacy_compute_log_mel(audio, config, center=False)
    out = voxtral_audio.compute_log_mel(audio, config, center=False)

    assert out.dtype == np.float32
    assert out.shape == expected.shape == (config.num_mel_bins, 7)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

