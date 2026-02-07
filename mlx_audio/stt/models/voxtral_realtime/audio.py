from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mlx.core as mx
import numpy as np
from mistral_common.audio import mel_filter_bank

from mlx_audio.utils import hanning, stft
from .config import AudioConfig


def compute_log_mel(audio: np.ndarray, config: AudioConfig, center: bool = True) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError("Audio must be 1D mono waveform")
    window = hanning(config.window_size)
    freqs = stft(
        mx.array(audio),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=window,
        center=center,
    ).T
    magnitudes = np.abs(np.array(freqs[:, :-1])) ** 2  # drop last frame to match vLLM

    mel_filters = mel_filter_bank(
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
        # slide buffer window
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
