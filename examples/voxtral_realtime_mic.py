#!/usr/bin/env python
"""Live microphone transcription with Voxtral Realtime.

Usage:
  uv run python examples/voxtral_realtime_mic.py --model ../voxtral-mlx/model_mlx_audio
  uv run python examples/voxtral_realtime_mic.py --list-devices
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import time
from dataclasses import dataclass
from threading import Event
from typing import Generator, Optional

import numpy as np
import sounddevice as sd

from mlx_audio.stt.models.voxtral_realtime.api import VoxtralRealtime


@dataclass
class MicChunkSource:
    sample_rate: int
    chunk_size: int
    device: Optional[int]
    stop_event: Event
    queue_max: int = 128

    def __post_init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.queue_max)

    def callback(self, indata, frames, _time, status) -> None:
        del frames
        if self.stop_event.is_set():
            return
        if status:
            print(f"[audio] {status}", file=sys.stderr, flush=True)
        chunk = np.array(indata[:, 0], dtype=np.float32, copy=True)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Keep latency bounded by dropping the oldest chunk when overloaded.
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def iter_chunks(self) -> Generator[np.ndarray, None, None]:
        while True:
            if self.stop_event.is_set():
                try:
                    while True:
                        yield self._queue.get_nowait()
                except queue.Empty:
                    return
            try:
                yield self._queue.get(timeout=0.1)
            except queue.Empty:
                continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime microphone transcription with Voxtral Realtime"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Voxtral-Realtime-Mini-3B-2507-4bit",
        help="Model path or Hub id",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (default: auto-detect / tokenizer default)",
    )
    parser.add_argument(
        "--strategy",
        default="realtime",
        choices=["realtime"],
        help="Streaming strategy (vLLM parity supports only realtime).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max generated tokens (0 = model limit)",
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        default=4,
        help="Quantization bits for runtime load (set 0 to disable)",
    )
    parser.add_argument(
        "--quantize-group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    quantize_bits = args.quantize_bits if args.quantize_bits > 0 else None

    print(f"Loading runtime: {args.model}", flush=True)
    load_start = time.perf_counter()
    runtime = VoxtralRealtime.load(
        args.model,
        quantize_bits=quantize_bits,
        quantize_group_size=args.quantize_group_size,
    )
    load_elapsed = time.perf_counter() - load_start
    print(f"Loaded in {load_elapsed:.2f}s", flush=True)

    audio_encoder = runtime.tokenizer.instruct_tokenizer.audio_encoder
    if audio_encoder is None:
        raise RuntimeError("Tokenizer is missing audio encoder configuration")
    audio_cfg = audio_encoder.audio_config
    sample_rate = int(audio_cfg.sampling_rate)
    chunk_size = int(sample_rate / float(audio_cfg.frame_rate))

    stop_event = Event()

    def _request_stop(_sig, _frame) -> None:
        if not stop_event.is_set():
            print("\nStopping capture. Finalizing...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    mic_source = MicChunkSource(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        device=args.device,
        stop_event=stop_event,
    )

    print(
        f"Mic stream started @ {sample_rate} Hz, chunk={chunk_size} samples "
        f"({chunk_size / sample_rate:.3f}s), strategy=realtime",
        flush=True,
    )
    print("Press Ctrl+C to stop.\n", flush=True)

    with sd.InputStream(
        samplerate=sample_rate,
        blocksize=chunk_size,
        channels=1,
        dtype="float32",
        device=args.device,
        callback=mic_source.callback,
    ):
        for text_delta in runtime.stream_transcribe(
            mic_source.iter_chunks(),
            language=args.language,
            max_tokens=args.max_tokens,
            strategy="realtime",
        ):
            if text_delta:
                print(text_delta, end="", flush=True)
            if stop_event.is_set():
                continue

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
