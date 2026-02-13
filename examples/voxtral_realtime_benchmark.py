#!/usr/bin/env python3
"""Benchmark Voxtral Realtime modes with timing and peak memory metrics.

This script reports:
- wall-clock timings (load/run/total)
- MLX peak memory
- process peak memory (RSS)
- macOS process physical footprint peaks (Activity Monitor-like)

Examples:
  uv run --extra stt python examples/voxtral_realtime_benchmark.py \
      --model /path/to/model \
      --audio /path/to/audio.wav \
      --mode offline

  uv run --extra stt python examples/voxtral_realtime_benchmark.py \
      --model /path/to/model \
      --audio /path/to/audio.wav \
      --mode realtime_api --chunk-ms 100
"""

import argparse
import ctypes
import json
import os
import resource
import sys
import time
from typing import Any

import mlx.core as mx

from mlx_audio.stt import load_model


def _to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def _get_peak_rss_bytes() -> int | None:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform.startswith("linux"):
            return int(ru.ru_maxrss) * 1024
        return int(ru.ru_maxrss)
    except Exception:
        return None


def _get_darwin_rusage_v4() -> dict[str, int] | None:
    if sys.platform != "darwin":
        return None

    class RUsageInfoV4(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            ("ri_user_time", ctypes.c_uint64),
            ("ri_system_time", ctypes.c_uint64),
            ("ri_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_interrupt_wkups", ctypes.c_uint64),
            ("ri_pageins", ctypes.c_uint64),
            ("ri_wired_size", ctypes.c_uint64),
            ("ri_resident_size", ctypes.c_uint64),
            ("ri_phys_footprint", ctypes.c_uint64),
            ("ri_proc_start_abstime", ctypes.c_uint64),
            ("ri_proc_exit_abstime", ctypes.c_uint64),
            ("ri_child_user_time", ctypes.c_uint64),
            ("ri_child_system_time", ctypes.c_uint64),
            ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_child_interrupt_wkups", ctypes.c_uint64),
            ("ri_child_pageins", ctypes.c_uint64),
            ("ri_child_elapsed_abstime", ctypes.c_uint64),
            ("ri_diskio_bytesread", ctypes.c_uint64),
            ("ri_diskio_byteswritten", ctypes.c_uint64),
            ("ri_cpu_time_qos_default", ctypes.c_uint64),
            ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
            ("ri_cpu_time_qos_background", ctypes.c_uint64),
            ("ri_cpu_time_qos_utility", ctypes.c_uint64),
            ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
            ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
            ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
            ("ri_billed_system_time", ctypes.c_uint64),
            ("ri_serviced_system_time", ctypes.c_uint64),
            ("ri_logical_writes", ctypes.c_uint64),
            ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
            ("ri_instructions", ctypes.c_uint64),
            ("ri_cycles", ctypes.c_uint64),
            ("ri_billed_energy", ctypes.c_uint64),
            ("ri_serviced_energy", ctypes.c_uint64),
            ("ri_interval_max_phys_footprint", ctypes.c_uint64),
            ("ri_runnable_time", ctypes.c_uint64),
        ]

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    proc_pid_rusage = libc.proc_pid_rusage
    proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(RUsageInfoV4)]
    proc_pid_rusage.restype = ctypes.c_int

    info = RUsageInfoV4()
    ret = proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    if ret != 0:
        return None

    return {
        "resident_size": int(info.ri_resident_size),
        "phys_footprint": int(info.ri_phys_footprint),
        "lifetime_max_phys_footprint": int(info.ri_lifetime_max_phys_footprint),
        "interval_max_phys_footprint": int(info.ri_interval_max_phys_footprint),
    }


def _to_text(result: Any) -> str:
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--mode", choices=["offline", "stream", "realtime_api"], required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    t0 = time.perf_counter()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    load_t0 = time.perf_counter()
    model = load_model(args.model)
    load_s = time.perf_counter() - load_t0

    run_t0 = time.perf_counter()
    internal_total_s = None

    if args.mode == "offline":
        out = model.generate(args.audio, stream=False, max_tokens=args.max_tokens)
        text = _to_text(out).strip()
        internal_total_s = float(getattr(out, "total_time", 0.0) or 0.0)
    elif args.mode == "stream":
        parts = []
        for delta in model.generate(args.audio, stream=True, max_tokens=args.max_tokens):
            parts.append(delta)
        text = "".join(parts).strip()
    else:
        if not hasattr(model, "transcribe_realtime"):
            print(
                json.dumps(
                    {
                        "label": args.label,
                        "mode": args.mode,
                        "error": "transcribe_realtime_not_available",
                    }
                )
            )
            return 2

        from mlx_audio.stt.models.voxtral_realtime.audio import iter_chunks

        audio_np = model._load_audio(args.audio)
        sample_rate = int(getattr(model.config.audio_encoding_args, "sampling_rate", 16000))
        chunk_size = max(1, int(sample_rate * args.chunk_ms / 1000))
        out = model.transcribe_realtime(
            iter_chunks(audio_np, chunk_size),
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        text = _to_text(out).strip()
        internal_total_s = float(getattr(out, "total_time", 0.0) or 0.0)

    run_s = time.perf_counter() - run_t0
    total_s = time.perf_counter() - t0

    peak_mlx_gb = None
    if hasattr(mx, "get_peak_memory"):
        try:
            peak_mlx_gb = float(mx.get_peak_memory()) / (1024**3)
        except Exception:
            peak_mlx_gb = None

    peak_rss_bytes = _get_peak_rss_bytes()
    darwin_v4 = _get_darwin_rusage_v4() or {}

    peak_phys_bytes = darwin_v4.get("lifetime_max_phys_footprint")
    peak_phys_interval_bytes = darwin_v4.get("interval_max_phys_footprint")
    current_rss_bytes = darwin_v4.get("resident_size")
    current_phys_bytes = darwin_v4.get("phys_footprint")

    mx.clear_cache()

    result = {
        "label": args.label,
        "mode": args.mode,
        "model": args.model,
        "audio": args.audio,
        "load_s": load_s,
        "run_s": run_s,
        "total_s": total_s,
        "internal_total_s": internal_total_s,
        "peak_mlx_gb": peak_mlx_gb,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gb": _to_gb(peak_rss_bytes),
        "peak_phys_bytes": peak_phys_bytes,
        "peak_phys_gb": _to_gb(peak_phys_bytes),
        "peak_phys_interval_bytes": peak_phys_interval_bytes,
        "peak_phys_interval_gb": _to_gb(peak_phys_interval_bytes),
        "current_rss_bytes": current_rss_bytes,
        "current_rss_gb": _to_gb(current_rss_bytes),
        "current_phys_bytes": current_phys_bytes,
        "current_phys_gb": _to_gb(current_phys_bytes),
        "text": text,
        "text_len": len(text),
    }
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
