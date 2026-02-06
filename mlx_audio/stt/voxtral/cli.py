from __future__ import annotations

import argparse

import numpy as np
from mistral_common.audio import Audio

from .api import VoxtralRealtime
from .audio import iter_chunks


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    audio = Audio.from_file(path)
    if audio.sampling_rate != target_sr:
        audio.resample(target_sr)
    return audio.audio_array.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser("mlx_audio.stt.voxtral")
    parser.add_argument("--model", required=True, help="HF repo or local path")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default="en")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--stream-strategy",
        default="stable",
        choices=["stable", "realtime"],
        help="Streaming backend strategy.",
    )
    parser.add_argument(
        "--stream-refresh-reads",
        type=int,
        default=50,
        help="For stable streaming, run a refresh every N buffer reads.",
    )
    parser.add_argument(
        "--realtime-carry-policy",
        default="exact",
        choices=["exact", "lexical"],
        help="For realtime strategy, choose how continuation token is carried across steps.",
    )
    parser.add_argument(
        "--realtime-auto-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically switch realtime streaming to stable refresh if token output degenerates.",
    )
    parser.add_argument(
        "--realtime-fallback-refresh-reads",
        type=int,
        default=50,
        help="When realtime fallback is active, run a stable refresh every N chunk reads.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant-bits", type=int, default=None, choices=[2, 3, 4, 8])
    parser.add_argument("--quant-group-size", type=int, default=32)
    args = parser.parse_args()

    model = VoxtralRealtime.load(
        args.model,
        dtype=args.dtype,
        quantize_bits=args.quant_bits,
        quantize_group_size=args.quant_group_size,
    )
    audio_cfg = model.tokenizer.instruct_tokenizer.audio_encoder.audio_config
    audio = _load_audio(args.audio, audio_cfg.sampling_rate)

    if args.stream:
        chunk_size = int(audio_cfg.sampling_rate / audio_cfg.frame_rate)
        for text in model.stream_transcribe(
            iter_chunks(audio, chunk_size),
            language=args.language,
            max_tokens=args.max_tokens,
            strategy=args.stream_strategy,
            offline_refresh_reads=args.stream_refresh_reads,
            realtime_carry_policy=args.realtime_carry_policy,
            realtime_auto_fallback=args.realtime_auto_fallback,
            realtime_fallback_refresh_reads=args.realtime_fallback_refresh_reads,
        ):
            print(text, end="", flush=True)
        print()
    else:
        out = model.transcribe(audio, language=args.language, max_tokens=args.max_tokens)
        print(out.text)


if __name__ == "__main__":
    main()
