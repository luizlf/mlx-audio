"""Mel spectrogram computation for Voxtral Realtime.

Matches the exact computation from vLLM/mistral_common:
- Slaney-style mel filter bank (0-8000 Hz, 128 bins) via dsp.mel_filters
- Periodic Hann window (size=400)
- STFT with n_fft=400, hop=160, center=True
- Drop last frame
- Fixed global_log_mel_max=1.5 clamping
"""

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from mlx_audio.dsp import mel_filters


def compute_mel_filters(
    num_mel_bins: int = 128,
    window_size: int = 400,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Compute Slaney-normalized mel filter bank.

    Returns:
        np.ndarray: Filter bank of shape [num_frequency_bins, num_mel_bins]
    """
    fb = mel_filters(
        sample_rate=sample_rate,
        n_fft=window_size,
        n_mels=num_mel_bins,
        f_min=0,
        f_max=8000,
        norm="slaney",
        mel_scale="slaney",
    )
    return np.array(fb).T  # dsp returns [mel, freq], we need [freq, mel]


def compute_mel_spectrogram(
    audio: mx.array,
    mel_filters: mx.array,
    window_size: int = 400,
    hop_length: int = 160,
    global_log_mel_max: float = 1.5,
) -> mx.array:
    """Compute log-mel spectrogram matching vLLM voxtral computation.

    Args:
        audio: 1D audio waveform, float32
        mel_filters: Precomputed mel filter bank [freq_bins, mel_bins]
        window_size: STFT window size (n_fft)
        hop_length: STFT hop length
        global_log_mel_max: Fixed max for log clamping

    Returns:
        mx.array: Log-mel spectrogram [mel_bins, frames]
    """
    # Periodic Hann window (divide by N, not N-1)
    n = mx.arange(window_size, dtype=mx.float32)
    window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / window_size))

    # Center padding (reflect)
    pad_size = window_size // 2
    audio_np = np.array(audio)
    audio_padded = np.pad(audio_np, (pad_size, pad_size), mode="reflect")
    audio = mx.array(audio_padded, dtype=mx.float32)

    # STFT
    n_samples = audio.shape[0]
    n_frames = 1 + (n_samples - window_size) // hop_length

    # Extract frames
    indices = (
        mx.arange(window_size)[None, :] + (mx.arange(n_frames) * hop_length)[:, None]
    )
    frames = audio[indices] * window[None, :]

    # Real FFT at exact n_fft size (MLX supports arbitrary sizes)
    spectrum = mx.fft.rfft(frames, n=window_size, axis=-1)

    # Power spectrum, drop last frame, transpose to [freq, frames]
    magnitudes = mx.abs(spectrum) ** 2
    magnitudes = magnitudes[:-1, :].T  # [n_freq, n_frames-1]

    # Apply mel filter bank: [mel_bins, freq] @ [freq, frames] -> [mel_bins, frames]
    mel_spec = mel_filters.T @ magnitudes

    # Log, clamp, scale
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
    min_val = global_log_mel_max - 8.0
    log_spec = mx.maximum(log_spec, min_val)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec  # [128, frames]


@dataclass
class StreamingBuffer:
    sampling_rate: int
    frame_rate: float
    transcription_delay_ms: float
    streaming_look_ahead_ms: float
    streaming_look_back_ms: float

    _buffer_seconds: int = 30
    _buffer: np.ndarray | None = None
    _filled: int = 0
    _start: int = 0
    _end: int = 0

    def __post_init__(self) -> None:
        self._buffer = np.empty(self._buffer_seconds * self.sampling_rate, dtype=np.float32)
        streaming_size = self._ms_to_samples(1000 / self.frame_rate)
        delay = self._ms_to_samples(self.transcription_delay_ms)
        self._start = 0
        self._end = delay + streaming_size

    def _ms_to_samples(self, ms: float) -> int:
        samples = self.sampling_rate * ms / 1000
        if not samples.is_integer():
            raise ValueError(f"Streaming ms must align to samples: {ms}")
        return int(samples)

    @property
    def start_idx(self) -> int:
        look_back = self._ms_to_samples(self.streaming_look_back_ms)
        return max(self._start - look_back, 0)

    @property
    def end_idx(self) -> int:
        look_ahead = self._ms_to_samples(self.streaming_look_ahead_ms)
        return self._end + look_ahead

    @property
    def is_audio_complete(self) -> bool:
        return self._filled >= self.end_idx

    def _ensure_capacity(self, add_samples: int) -> None:
        assert self._buffer is not None
        if self._filled + add_samples <= self._buffer.shape[0]:
            return
        keep = max(self._filled - self.start_idx, 0)
        new_buffer = np.empty_like(self._buffer)
        if keep > 0:
            new_buffer[:keep] = self._buffer[self.start_idx : self._filled]
        self._buffer = new_buffer
        self._filled = keep
        look_back = self._ms_to_samples(self.streaming_look_back_ms)
        streaming_size = self._ms_to_samples(1000 / self.frame_rate)
        self._start = look_back
        self._end = self._start + streaming_size

    def write(self, audio: np.ndarray) -> None:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self._ensure_capacity(len(audio))
        assert self._buffer is not None
        self._buffer[self._filled : self._filled + len(audio)] = audio
        self._filled += len(audio)

    def read(self) -> Optional[np.ndarray]:
        if not self.is_audio_complete:
            return None
        assert self._buffer is not None
        segment = self._buffer[self.start_idx : self.end_idx]
        self._start = self._end
        streaming_size = self._ms_to_samples(1000 / self.frame_rate)
        self._end = self._start + streaming_size
        return segment.copy()


def iter_chunks(audio: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]
