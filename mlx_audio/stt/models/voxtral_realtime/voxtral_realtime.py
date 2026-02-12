"""Voxtral Realtime 4B - Main model orchestrator.

Inference pipeline:
1. Resample audio to 16kHz, pad (left silence + right silence)
2. Compute mel spectrogram
3. Run causal encoder -> 4x downsample -> adapter
4. Construct prompt: [BOS] + [STREAMING_PAD] * (n_left_pad + n_delay)
5. For each position: input = audio_embed + tok_embed(token_id)
6. Prefill decoder, then autoregressive generation until EOS
7. Decode tokens via Tekken tokenizer

Optimizations:
- Explicit mx.eval after prefill and periodically during decode to bound graph size
- Sampling uses logits directly (skips softmax+log round-trip)
"""

import math
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import STTOutput
from .audio import StreamingBuffer, compute_mel_filters, compute_mel_spectrogram
from .config import ModelConfig
from .decoder import Decoder, compute_time_embedding
from .encoder import AudioEncoder
from .tokenizer import TekkenTokenizer

try:
    from mistral_common.audio import Audio
    from mistral_common.protocol.transcription.request import (
        RawAudio,
        StreamingMode,
        TranscriptionRequest,
    )
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    Audio = None
    RawAudio = None
    StreamingMode = None
    TranscriptionRequest = None
    MistralTokenizer = None

# Derived streaming constants
SAMPLE_RATE = 16000
FRAME_RATE = 12.5
RAW_AUDIO_LENGTH_PER_TOK = int(SAMPLE_RATE // FRAME_RATE)  # 1280
HOP_LENGTH = 160
AUDIO_LENGTH_PER_TOK = RAW_AUDIO_LENGTH_PER_TOK // HOP_LENGTH  # 8


def _num_audio_tokens(audio_len):
    if audio_len % HOP_LENGTH != 0:
        audio_len = math.ceil(audio_len / HOP_LENGTH - 1)
    else:
        audio_len = audio_len // HOP_LENGTH
    return math.ceil(audio_len / AUDIO_LENGTH_PER_TOK)


def _num_delay_tokens(delay_ms):
    delay_len = int(delay_ms / 1000.0 * SAMPLE_RATE)
    return _num_audio_tokens(delay_len)


def _pad_audio_streaming(audio_array, n_left_pad_tokens, n_right_pad_tokens):
    """Pad audio for offline streaming mode.

    Left pad: n_left_pad_tokens * 1280 samples of silence
    Right pad: align to 1280 + n_right_pad_tokens * 1280 of silence
    """
    mult_of = RAW_AUDIO_LENGTH_PER_TOK
    n_samples = len(audio_array)
    align_pad = (mult_of - (n_samples % mult_of)) % mult_of
    right_pad = align_pad + n_right_pad_tokens * mult_of
    left_pad = n_left_pad_tokens * mult_of
    return np.pad(audio_array, (left_pad, right_pad))


def _fit_audio_embeddings_to_slots(audio_embeds: mx.array, num_slots: int) -> mx.array:
    slot_count = int(audio_embeds.shape[0])
    if slot_count == num_slots:
        return audio_embeds
    if slot_count > num_slots:
        return audio_embeds[:num_slots]
    pad = num_slots - slot_count
    return mx.concatenate(
        [audio_embeds, mx.zeros((pad, audio_embeds.shape[-1]), dtype=audio_embeds.dtype)],
        axis=0,
    )


def _load_mistral_tokenizer_compat(tekken_path: Path):
    if MistralTokenizer is None:
        return None
    try:
        return MistralTokenizer.from_file(str(tekken_path))
    except TypeError:
        payload = json.loads(tekken_path.read_text(encoding="utf-8"))
        audio_cfg = payload.get("audio")
        if not isinstance(audio_cfg, dict):
            raise

        allowed_keys = {
            "sampling_rate",
            "frame_rate",
            "audio_encoding_config",
            "chunk_length_s",
            "transcription_delay_ms",
            "transcription_format",
        }
        cleaned_audio = {k: v for k, v in audio_cfg.items() if k in allowed_keys}
        if cleaned_audio == audio_cfg:
            raise

        payload["audio"] = cleaned_audio
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="tekken.json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        try:
            return MistralTokenizer.from_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@dataclass(frozen=True)
class RealtimeTokenEvent:
    token_id: int
    text: str


@dataclass
class _RealtimeSession:
    model: "Model"
    max_tokens: int
    temperature: float
    transcription_delay_ms: Optional[int]
    realtime_chunk_multiple: int
    buffer: StreamingBuffer
    pending_chunk_remainder: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    stream_samples: int = 0
    is_first_segment: bool = True
    decoder_cache: Optional[List[tuple[mx.array, mx.array, int]]] = None
    decoder_position: int = 0
    last_token: Optional[int] = None
    generated_total: int = 0
    decoded_history: List[int] = field(default_factory=list)
    emitted_text: str = ""
    n_delay_tokens: int = 0
    n_right_pad_tokens: int = 0
    raw_audio_length_per_tok: int = int(RAW_AUDIO_LENGTH_PER_TOK)
    downsample: int = 0
    audio_position_frames: int = 0
    audio_caches: Optional[list] = None
    pending_conv: Optional[mx.array] = None

    def __post_init__(self) -> None:
        delay_ms = self.transcription_delay_ms or self.model.config.transcription_delay_ms
        self.n_delay_tokens = _num_delay_tokens(delay_ms)
        self.n_right_pad_tokens = int((self.n_delay_tokens + 1) + 10)
        tokenizer = getattr(self.model, "_mistral_tokenizer", None)
        if tokenizer is not None:
            audio_encoder = tokenizer.instruct_tokenizer.audio_encoder
            if audio_encoder is not None:
                audio_cfg = audio_encoder.audio_config
                self.n_delay_tokens = int(
                    getattr(audio_cfg, "num_delay_tokens", self.n_delay_tokens)
                )
                self.n_right_pad_tokens = int(
                    getattr(audio_cfg, "n_right_pad_tokens", self.n_right_pad_tokens)
                )
                self.raw_audio_length_per_tok = int(
                    getattr(
                        audio_cfg,
                        "raw_audio_length_per_tok",
                        self.raw_audio_length_per_tok,
                    )
                )
        self.model._ensure_ada_scales(delay_ms)
        self.downsample = int(self.model.encoder.config.downsample_factor)
        self.audio_caches = self.model.encoder.make_chunk_caches(self.downsample)
        self.pending_conv = mx.zeros((0, int(self.model.encoder.config.dim)), dtype=mx.float32)

    @property
    def exhausted(self) -> bool:
        if self.last_token == int(self.model.config.eos_token_id):
            return True
        return self.max_tokens > 0 and self.generated_total >= self.max_tokens

    def feed_chunk(self, chunk: np.ndarray) -> List[RealtimeTokenEvent]:
        if self.exhausted:
            return []
        chunk_f32 = chunk.astype(np.float32)
        if self.pending_chunk_remainder.size:
            chunk_f32 = np.concatenate([self.pending_chunk_remainder, chunk_f32], axis=0)
        usable = (len(chunk_f32) // self.realtime_chunk_multiple) * self.realtime_chunk_multiple
        self.pending_chunk_remainder = chunk_f32[usable:]
        if usable <= 0:
            return []

        self.buffer.write(chunk_f32[:usable])
        self.stream_samples += int(usable)

        out: List[RealtimeTokenEvent] = []
        while (segment := self.buffer.read()) is not None:
            out.extend(self._consume_segment(segment))
            if self.exhausted:
                break
        return out

    def flush(self) -> List[RealtimeTokenEvent]:
        if self.exhausted:
            return []

        if self.pending_chunk_remainder.size:
            pad = self.realtime_chunk_multiple - int(self.pending_chunk_remainder.size)
            tail = np.pad(self.pending_chunk_remainder, (0, pad)).astype(np.float32)
            self.pending_chunk_remainder = np.zeros((0,), dtype=np.float32)
            self.buffer.write(tail)
            self.stream_samples += int(tail.shape[0])

        raw_len = int(self.raw_audio_length_per_tok)
        right_tokens = int(self.n_right_pad_tokens)
        align_pad = (raw_len - (self.stream_samples % raw_len)) % raw_len
        total_pad = int(align_pad + raw_len * right_tokens)
        if total_pad > 0:
            mult = int(self.realtime_chunk_multiple)
            if mult > 0 and total_pad % mult:
                total_pad = ((total_pad + mult - 1) // mult) * mult
            self.buffer.write(np.zeros((total_pad,), dtype=np.float32))
            self.stream_samples += total_pad

        out: List[RealtimeTokenEvent] = []
        while (segment := self.buffer.read()) is not None:
            out.extend(self._consume_segment(segment))
            if self.exhausted:
                break
        return out

    def _emit(self, token: int) -> RealtimeTokenEvent:
        self.last_token = int(token)
        self.generated_total += 1
        self.decoded_history.append(int(token))
        text_so_far = self.model._tokenizer.decode(
            [t for t in self.decoded_history if t != self.model.config.eos_token_id]
        )
        if text_so_far.startswith(self.emitted_text):
            delta = text_so_far[len(self.emitted_text) :]
        else:
            delta = text_so_far
        self.emitted_text = text_so_far
        return RealtimeTokenEvent(token_id=int(token), text=delta)

    def _append_conv(self, audio_array: np.ndarray) -> None:
        conv = self.model._encode_audio_to_conv(audio_array)
        if int(conv.shape[0]) <= 0:
            return
        if self.pending_conv is None or int(self.pending_conv.shape[0]) == 0:
            self.pending_conv = conv
        else:
            self.pending_conv = mx.concatenate([self.pending_conv, conv], axis=0)

    def _next_audio_embed(self) -> Optional[mx.array]:
        if self.pending_conv is None:
            return None
        if int(self.pending_conv.shape[0]) < self.downsample:
            return None

        chunk = self.pending_conv[: self.downsample]
        self.pending_conv = self.pending_conv[self.downsample :]
        encoded = self.model.encoder.encode_incremental_chunk(
            chunk,
            caches=self.audio_caches,
            chunk_start=self.audio_position_frames,
        )
        self.audio_position_frames += self.downsample
        adapted = self.model.encoder.downsample_and_project(encoded)
        if int(adapted.shape[0]) <= 0:
            return None
        return adapted[0]

    def _consume_segment(self, segment: np.ndarray) -> List[RealtimeTokenEvent]:
        if self.exhausted:
            return []

        if self.is_first_segment:
            prompt_ids_mx, prepared_audio = self.model._prepare_realtime_first_inputs(
                segment
            )
            prompt_len = int(prompt_ids_mx.shape[0])
            self._append_conv(prepared_audio)

            audio_embeds: List[mx.array] = []
            for _ in range(prompt_len):
                emb = self._next_audio_embed()
                if emb is None:
                    break
                audio_embeds.append(emb)
            if len(audio_embeds) < prompt_len:
                self.is_first_segment = False
                return []
            adapter_slots = mx.stack(audio_embeds, axis=0)

            prompt_embeds = self.model.decoder.embed_tokens(prompt_ids_mx)
            prefix_embeds = adapter_slots + prompt_embeds

            h, self.decoder_cache = self.model.decoder.forward(prefix_embeds, start_pos=0, cache=None)
            logits = self.model.decoder.logits(h[-1])
            token = int(self.model._next_token_mx(logits, self.temperature).item())
            self.decoder_position = prompt_len
            self.is_first_segment = False
            return [self._emit(token)]

        if self.last_token is None:
            return []

        self._append_conv(segment)
        audio_embed = self._next_audio_embed()
        if audio_embed is None:
            return []

        token_embed = self.model.decoder.embed_token(int(self.last_token))
        step_embed = token_embed + audio_embed

        h, self.decoder_cache = self.model.decoder.forward(
            step_embed[None, :],
            start_pos=int(self.decoder_position),
            cache=self.decoder_cache,
        )
        logits = self.model.decoder.logits(h.squeeze(0))
        token = int(self.model._next_token_mx(logits, self.temperature).item())
        self.decoder_position += 1
        return [self._emit(token)]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = AudioEncoder(config.encoder_args)
        self.decoder = Decoder(config.decoder)

        # Will be set in post_load_hook
        self._tokenizer = None
        self._mistral_tokenizer = None
        self._mel_filters = None

    def _ensure_mel_filters(self):
        if self._mel_filters is None:
            aec = self.config.audio_encoding_args
            filters_np = compute_mel_filters(
                num_mel_bins=aec.num_mel_bins,
                window_size=aec.window_size,
                sample_rate=aec.sampling_rate,
            )
            self._mel_filters = mx.array(filters_np, dtype=mx.float32)
        return self._mel_filters

    def _load_audio(
        self, audio_input: Union[str, Path, List[mx.array], mx.array, np.ndarray]
    ) -> np.ndarray:
        """Load and resample audio from file path or array to 16kHz float32 numpy."""
        if isinstance(audio_input, (str, Path)):
            import soundfile as sf

            audio_np, sr = sf.read(str(audio_input), dtype="float32")
            audio_np = audio_np.flatten()
            if sr != SAMPLE_RATE:
                # Resample to 16kHz
                from scipy.signal import resample

                n_samples = int(len(audio_np) * SAMPLE_RATE / sr)
                audio_np = resample(audio_np, n_samples).astype(np.float32)
            return audio_np
        if isinstance(audio_input, list):
            audio_input = audio_input[0]
        return np.array(audio_input).flatten().astype(np.float32)

    def _prepare_realtime_first_inputs(
        self, segment: np.ndarray
    ) -> tuple[mx.array, np.ndarray]:
        if (
            self._mistral_tokenizer is None
            or Audio is None
            or RawAudio is None
            or TranscriptionRequest is None
            or StreamingMode is None
        ):
            raise RuntimeError(
                "Realtime transcription requires mistral-common[audio]. "
                "Install with: pip install 'mlx-audio[stt]'"
            )

        audio_obj = Audio(
            segment.astype(np.float32),
            int(self.config.audio_encoding_args.sampling_rate),
            format="wav",
        )
        req = TranscriptionRequest(
            model="voxtral",
            audio=RawAudio.from_audio(audio_obj),
            language=None,
            streaming=StreamingMode.ONLINE,
        )
        tokenized = self._mistral_tokenizer.instruct_tokenizer.encode_transcription(req)
        return (
            mx.array(tokenized.tokens),
            np.array(tokenized.audios[0].audio_array, dtype=np.float32),
        )

    def _prepare_mel(self, audio_np, transcription_delay_ms=None):
        """Prepare mel spectrogram from audio numpy array."""
        delay_ms = transcription_delay_ms or self.config.transcription_delay_ms
        n_delay = _num_delay_tokens(delay_ms)
        n_left = self.config.n_left_pad_tokens
        n_right = (n_delay + 1) + 10

        padded = _pad_audio_streaming(audio_np, n_left, n_right)

        aec = self.config.audio_encoding_args
        mel_filters = self._ensure_mel_filters()
        audio_mx = mx.array(padded, dtype=mx.float32)
        mel = compute_mel_spectrogram(
            audio_mx,
            mel_filters,
            window_size=aec.window_size,
            hop_length=aec.hop_length,
            global_log_mel_max=aec.global_log_mel_max,
        )

        if mel.shape[1] % 2 != 0:
            mel = mel[:, 1:]

        return mel, n_delay

    def _encode_segment_to_adapter(
        self,
        audio_np: np.ndarray,
        *,
        left_pad_tokens: int = 0,
        right_pad_tokens: int = 0,
    ) -> mx.array:
        if left_pad_tokens or right_pad_tokens:
            audio_np = _pad_audio_streaming(
                audio_np,
                int(left_pad_tokens),
                int(right_pad_tokens),
            )
        conv_out = self._encode_audio_to_conv(audio_np)
        if int(conv_out.shape[0]) <= 0:
            return mx.zeros((0, int(self.config.decoder.dim)), dtype=mx.float32)

        sw = int(self.encoder.config.sliding_window)
        if int(conv_out.shape[0]) <= sw:
            return self.encoder.encode_full(conv_out)

        encoded = mx.concatenate(list(self.encoder.encode_chunks(conv_out)), axis=0)
        return self.encoder.downsample_and_project(encoded)

    def _encode_audio_to_conv(self, audio_np: np.ndarray) -> mx.array:
        aec = self.config.audio_encoding_args
        mel_filters = self._ensure_mel_filters()
        audio_mx = mx.array(audio_np, dtype=mx.float32)
        mel = compute_mel_spectrogram(
            audio_mx,
            mel_filters,
            window_size=aec.window_size,
            hop_length=aec.hop_length,
            global_log_mel_max=aec.global_log_mel_max,
        )
        if mel.shape[1] % 2 != 0:
            mel = mel[:, 1:]
        return self.encoder.conv_stem(mel)

    def _ensure_ada_scales(self, transcription_delay_ms=None):
        """Ensure ada_scales match the given delay. Recomputes if needed."""
        delay_ms = transcription_delay_ms or self.config.transcription_delay_ms
        n_delay = _num_delay_tokens(delay_ms)
        if n_delay != self._ada_scale_delay:
            from .decoder import compute_time_embedding

            t_cond = compute_time_embedding(float(n_delay), self.config.decoder.dim)
            self.decoder.precompute_ada_scales(t_cond)
            for scale in self.decoder._ada_scales:
                if scale is not None:
                    mx.eval(scale)
            self._ada_scale_delay = n_delay

    def _encode_and_prefill(self, audio_np, verbose=False, transcription_delay_ms=None):
        """Shared encoder + prefill logic with incremental encoding.

        Only encodes the first audio chunk before prefill, deferring the
        rest to the autoregressive loop. Returns a chunk generator for
        on-demand encoding of remaining audio.

        Returns:
            (adapter_out, n_audio_total, prompt_len, logits, cache,
             enc_chunk_gen, start_time)
        """
        start_time = time.time()

        self._ensure_ada_scales(transcription_delay_ms)
        mel, n_delay = self._prepare_mel(audio_np, transcription_delay_ms)

        # Run conv stem and compute total audio token count
        conv_out = self.encoder.conv_stem(mel)
        ds = self.encoder.config.downsample_factor
        n_audio_total = conv_out.shape[0] // ds

        n_left = self.config.n_left_pad_tokens
        prompt_len = 1 + n_left + n_delay
        sw = self.encoder.config.sliding_window

        if conv_out.shape[0] <= sw:
            # Short audio: use non-chunked path with optimized causal attention
            # (SDPA Flash Attention kernel, no RotatingKVCache overhead)
            adapter_out = self.encoder.encode_full(conv_out)
            enc_chunk_gen = None
        else:
            # Long audio: chunked encoding with generator for incremental decode
            enc_chunk_gen = self.encoder.encode_chunks(conv_out)
            adapter_chunks = []
            adapter_len = 0
            while adapter_len < prompt_len:
                try:
                    chunk_out = next(enc_chunk_gen)
                    chunk_adapter = self.encoder.downsample_and_project(chunk_out)
                    adapter_chunks.append(chunk_adapter)
                    adapter_len += chunk_adapter.shape[0]
                except StopIteration:
                    enc_chunk_gen = None
                    break
            adapter_out = (
                mx.concatenate(adapter_chunks, axis=0) if adapter_chunks else None
            )

        if verbose:
            print(f"Audio: {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.1f}s)")
            print(
                f"Encoder: {'non-chunked (causal)' if conv_out.shape[0] <= sw else 'chunked'}, "
                f"{adapter_out.shape[0]} adapter tokens"
            )
            print(f"Total audio tokens: {n_audio_total}")

        # Build prompt embeddings
        prompt_ids = [self.config.bos_token_id] + [
            self.config.streaming_pad_token_id
        ] * (n_left + n_delay)

        prompt_ids_mx = mx.array(prompt_ids)
        prompt_text_embeds = self.decoder.embed_tokens(prompt_ids_mx)
        prefix_embeds = adapter_out[:prompt_len] + prompt_text_embeds

        if verbose:
            print(f"Prompt: {prompt_len} tokens, Audio span: {n_audio_total} tokens")

        # Prefill (single fused forward pass)
        prefill_start = time.time()
        h, cache = self.decoder.forward(prefix_embeds, start_pos=0)
        logits = self.decoder.logits(h[-1])
        cache_arrays = [t for lc in cache for t in lc[:2]]
        mx.eval(logits, *cache_arrays)

        if verbose:
            total_ttft = time.time() - start_time
            prefill_time = time.time() - prefill_start
            print(
                f"Prefill: {prompt_len} tokens in {prefill_time*1000:.0f}ms "
                f"({prompt_len / prefill_time:.0f} tok/s)"
            )
            print(f"Total TTFT: {total_ttft*1000:.0f}ms")

        return (
            adapter_out,
            n_audio_total,
            prompt_len,
            logits,
            cache,
            enc_chunk_gen,
            start_time,
        )

    def generate(
        self,
        audio: Union[str, Path, List[mx.array], mx.array],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        verbose: bool = False,
        stream: bool = False,
        transcription_delay_ms: Optional[int] = None,
        **kwargs,
    ):
        """Transcribe audio. Returns STTOutput, or yields text deltas if stream=True.

        Args:
            transcription_delay_ms: Override the model's default transcription delay
                (default 480ms). Lower values reduce latency but may hurt accuracy.
        """
        audio_np = self._load_audio(audio)

        if stream:
            return self._generate_stream(
                audio_np, max_tokens, temperature, verbose, transcription_delay_ms
            )

        adapter_out, n_audio, prompt_len, logits, cache, enc_chunk_gen, start_time = (
            self._encode_and_prefill(audio_np, verbose, transcription_delay_ms)
        )

        adapter_len = adapter_out.shape[0]
        generated = []

        # Double-buffered async eval: submit token computation to GPU,
        # read result at start of next iteration while GPU works on new ops
        next_tok = self._next_token_mx(logits, temperature)
        mx.async_eval(next_tok)

        decode_start = time.time()
        for pos in range(prompt_len, n_audio):
            token = int(next_tok.item())
            generated.append(token)
            if token == self.config.eos_token_id or len(generated) > max_tokens:
                break

            # Encode more audio chunks on demand
            if enc_chunk_gen is not None and pos >= adapter_len:
                try:
                    chunk_out = next(enc_chunk_gen)
                    chunk_adapter = self.encoder.downsample_and_project(chunk_out)
                    mx.eval(chunk_adapter)
                    adapter_out = mx.concatenate([adapter_out, chunk_adapter], axis=0)
                    adapter_len = adapter_out.shape[0]
                except StopIteration:
                    enc_chunk_gen = None

            if pos < adapter_len:
                embed = adapter_out[pos] + self.decoder.embed_token(token)
            else:
                embed = self.decoder.embed_token(token)

            h, cache = self.decoder.forward(embed[None, :], start_pos=pos, cache=cache)
            logits = self.decoder.logits(h.squeeze(0))
            next_tok = self._next_token_mx(logits, temperature)
            mx.async_eval(next_tok)

            if len(generated) % 256 == 0:
                mx.clear_cache()
        else:
            # Loop completed without break — read final pending token
            token = int(next_tok.item())
            generated.append(token)

        if generated and generated[-1] == self.config.eos_token_id:
            generated = generated[:-1]

        text = self._tokenizer.decode(generated).strip()
        end_time = time.time()
        total_time = end_time - start_time
        decode_time = end_time - decode_start

        if verbose:
            n_gen = len(generated)
            print(
                f"Decode: {n_gen} tokens in {decode_time:.3f}s "
                f"({n_gen / decode_time:.0f} tok/s, "
                f"{decode_time / max(n_gen, 1) * 1000:.1f} ms/tok)"
            )
            print(f"Total: {total_time:.3f}s")

        mx.clear_cache()

        return STTOutput(
            text=text,
            prompt_tokens=prompt_len,
            generation_tokens=len(generated),
            total_tokens=prompt_len + len(generated),
            total_time=total_time,
            prompt_tps=prompt_len / total_time if total_time > 0 else 0,
            generation_tps=len(generated) / decode_time if decode_time > 0 else 0,
        )

    def _generate_stream(
        self, audio_np, max_tokens, temperature, verbose, transcription_delay_ms=None
    ):
        """Generator that yields text deltas as tokens are decoded."""
        adapter_out, n_audio, prompt_len, logits, cache, enc_chunk_gen, start_time = (
            self._encode_and_prefill(audio_np, verbose, transcription_delay_ms)
        )

        adapter_len = adapter_out.shape[0]
        generated = []
        prev_text = ""

        next_tok = self._next_token_mx(logits, temperature)
        mx.async_eval(next_tok)

        for pos in range(prompt_len, n_audio):
            token = int(next_tok.item())
            generated.append(token)

            # Yield text delta (filter EOS from partial decode)
            text_so_far = self._tokenizer.decode(
                [t for t in generated if t != self.config.eos_token_id]
            )
            if text_so_far != prev_text:
                yield text_so_far[len(prev_text) :]
                prev_text = text_so_far

            if token == self.config.eos_token_id or len(generated) > max_tokens:
                break

            # Encode more audio chunks on demand
            if enc_chunk_gen is not None and pos >= adapter_len:
                try:
                    chunk_out = next(enc_chunk_gen)
                    chunk_adapter = self.encoder.downsample_and_project(chunk_out)
                    mx.eval(chunk_adapter)
                    adapter_out = mx.concatenate([adapter_out, chunk_adapter], axis=0)
                    adapter_len = adapter_out.shape[0]
                except StopIteration:
                    enc_chunk_gen = None

            if pos < adapter_len:
                embed = adapter_out[pos] + self.decoder.embed_token(token)
            else:
                embed = self.decoder.embed_token(token)

            h, cache = self.decoder.forward(embed[None, :], start_pos=pos, cache=cache)
            logits = self.decoder.logits(h.squeeze(0))
            next_tok = self._next_token_mx(logits, temperature)
            mx.async_eval(next_tok)

            if len(generated) % 256 == 0:
                mx.clear_cache()
        else:
            # Loop completed without break — read final pending token
            token = int(next_tok.item())
            generated.append(token)
            text_so_far = self._tokenizer.decode(
                [t for t in generated if t != self.config.eos_token_id]
            )
            if text_so_far != prev_text:
                yield text_so_far[len(prev_text) :]

        mx.clear_cache()

    def stream_realtime_tokens(
        self,
        audio_iter: Iterable[np.ndarray],
        *,
        max_tokens: int = 0,
        temperature: float = 0.0,
        transcription_delay_ms: Optional[int] = None,
    ) -> Generator[RealtimeTokenEvent, None, None]:
        try:
            if self._mistral_tokenizer is None:
                raise RuntimeError(
                    "Realtime transcription requires mistral-common[audio]. "
                    "Install with: pip install 'mlx-audio[stt]'"
                )
            aec = self.config.audio_encoding_args
            realtime_chunk_multiple = int(abs((int(aec.window_size) // 2) - int(aec.hop_length)))
            if realtime_chunk_multiple <= 0:
                raise ValueError("Realtime chunk multiple must be > 0")

            sampling_rate = int(aec.sampling_rate)
            frame_rate = float(aec.frame_rate)
            delay_ms = float(
                transcription_delay_ms
                if transcription_delay_ms is not None
                else self.config.transcription_delay_ms
            )
            look_ahead_ms = float(self.config.streaming_look_ahead_ms)
            look_back_ms = float(self.config.streaming_look_back_ms)

            tokenizer = getattr(self, "_mistral_tokenizer", None)
            if tokenizer is not None:
                audio_encoder = tokenizer.instruct_tokenizer.audio_encoder
                if audio_encoder is not None:
                    audio_cfg = audio_encoder.audio_config
                    sampling_rate = int(getattr(audio_cfg, "sampling_rate", sampling_rate))
                    frame_rate = float(getattr(audio_cfg, "frame_rate", frame_rate))
                    delay_ms = float(getattr(audio_cfg, "transcription_delay_ms", delay_ms))
                    look_ahead_ms = float(
                        getattr(audio_cfg, "streaming_look_ahead_ms", look_ahead_ms)
                    )
                    look_back_ms = float(
                        getattr(audio_cfg, "streaming_look_back_ms", look_back_ms)
                    )

            session = _RealtimeSession(
                model=self,
                max_tokens=max_tokens,
                temperature=temperature,
                transcription_delay_ms=transcription_delay_ms,
                realtime_chunk_multiple=realtime_chunk_multiple,
                buffer=StreamingBuffer(
                    sampling_rate=sampling_rate,
                    frame_rate=frame_rate,
                    transcription_delay_ms=delay_ms,
                    streaming_look_ahead_ms=look_ahead_ms,
                    streaming_look_back_ms=look_back_ms,
                ),
            )

            for chunk in audio_iter:
                for event in session.feed_chunk(chunk):
                    yield event
                if session.exhausted:
                    break

            if not session.exhausted:
                for event in session.flush():
                    yield event
        finally:
            mx.clear_cache()

    def stream_realtime(
        self,
        audio_iter: Iterable[np.ndarray],
        *,
        max_tokens: int = 0,
        temperature: float = 0.0,
        transcription_delay_ms: Optional[int] = None,
    ) -> Generator[str, None, None]:
        for event in self.stream_realtime_tokens(
            audio_iter,
            max_tokens=max_tokens,
            temperature=temperature,
            transcription_delay_ms=transcription_delay_ms,
        ):
            if event.text:
                yield event.text

    def transcribe_realtime(
        self,
        audio_iter: Iterable[np.ndarray],
        *,
        max_tokens: int = 0,
        temperature: float = 0.0,
        transcription_delay_ms: Optional[int] = None,
    ) -> STTOutput:
        start = time.time()
        events = list(
            self.stream_realtime_tokens(
                audio_iter,
                max_tokens=max_tokens,
                temperature=temperature,
                transcription_delay_ms=transcription_delay_ms,
            )
        )
        eos = int(self.config.eos_token_id)
        generated = [int(event.token_id) for event in events if int(event.token_id) != eos]
        text = self._tokenizer.decode(generated).strip()
        total_time = max(time.time() - start, 1e-9)

        delay_ms = transcription_delay_ms or self.config.transcription_delay_ms
        prompt_tokens = int(1 + self.config.n_left_pad_tokens + _num_delay_tokens(delay_ms))
        return STTOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(generated),
            total_tokens=prompt_tokens + len(generated),
            total_time=total_time,
            prompt_tps=prompt_tokens / total_time,
            generation_tps=len(generated) / total_time,
        )

    def _next_token_mx(self, logits, temperature):
        """Compute next token as lazy mx.array (for async eval pipelining)."""
        if temperature == 0:
            return mx.argmax(logits)
        return mx.random.categorical(logits * (1.0 / temperature))

    def _sample(self, logits, temperature):
        # categorical accepts unnormalized logits directly — no softmax+log needed
        return int(self._next_token_mx(logits, temperature).item())

    def sanitize(self, weights):
        """Map weight names from consolidated.safetensors to our module structure."""
        new_weights = {}

        enc_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
        adapter_prefix = "mm_streams_embeddings.embedding_module"
        tok_emb_key = "mm_streams_embeddings.embedding_module.tok_embeddings.weight"

        for k, v in weights.items():
            new_key = None

            if k == tok_emb_key:
                new_key = "decoder.tok_embeddings.weight"

            elif k == "norm.weight":
                new_key = "decoder.norm.weight"

            elif k.startswith(f"{enc_prefix}.conv_layers."):
                # e.g., ...conv_layers.0.conv.weight -> encoder.conv_layers_0_conv.conv.weight
                rest = k[len(f"{enc_prefix}.conv_layers.") :]
                # rest: "0.conv.weight" or "0.conv.bias"
                parts = rest.split(".", 2)  # ['0', 'conv', 'weight']
                layer_idx = parts[0]
                param = parts[2]  # 'weight' or 'bias'
                new_key = f"encoder.conv_layers_{layer_idx}_conv.conv.{param}"

                # Transpose conv weights from PyTorch [out, in, k] to MLX [out, k, in]
                if param == "weight" and v.ndim == 3:
                    v = v.transpose(0, 2, 1)

            elif k.startswith(f"{enc_prefix}.transformer.layers."):
                rest = k[len(f"{enc_prefix}.transformer.layers.") :]
                # e.g., "0.attention.wq.weight"
                parts = rest.split(".", 1)
                layer_idx = parts[0]
                param_path = parts[1]

                # Map FFN weights
                param_path = param_path.replace("feed_forward.w1.", "feed_forward_w1.")
                param_path = param_path.replace("feed_forward.w2.", "feed_forward_w2.")
                param_path = param_path.replace("feed_forward.w3.", "feed_forward_w3.")

                new_key = f"encoder.transformer_layers.{layer_idx}.{param_path}"

            elif k.startswith(f"{enc_prefix}.transformer.norm."):
                rest = k[len(f"{enc_prefix}.transformer.norm.") :]
                new_key = f"encoder.transformer_norm.{rest}"

            elif k.startswith(f"{adapter_prefix}.audio_language_projection."):
                rest = k[len(f"{adapter_prefix}.audio_language_projection.") :]
                # "0.weight" -> "audio_language_projection_0.weight"
                # "2.weight" -> "audio_language_projection_2.weight"
                parts = rest.split(".", 1)
                idx = parts[0]
                param = parts[1]
                new_key = f"encoder.audio_language_projection_{idx}.{param}"

            elif k.startswith("layers."):
                rest = k[len("layers.") :]
                # e.g., "0.attention.wq.weight"
                parts = rest.split(".", 1)
                layer_idx = parts[0]
                param_path = parts[1]

                # Map FFN
                param_path = param_path.replace("feed_forward.w1.", "feed_forward_w1.")
                param_path = param_path.replace("feed_forward.w2.", "feed_forward_w2.")
                param_path = param_path.replace("feed_forward.w3.", "feed_forward_w3.")
                # Map ada norm: ada_rms_norm_t_cond.0.weight -> ada_rms_norm_t_cond.ada_down.weight
                param_path = param_path.replace(
                    "ada_rms_norm_t_cond.0.", "ada_rms_norm_t_cond.ada_down."
                )
                param_path = param_path.replace(
                    "ada_rms_norm_t_cond.2.", "ada_rms_norm_t_cond.ada_up."
                )

                new_key = f"decoder.layers.{layer_idx}.{param_path}"

            if new_key is not None:
                new_weights[new_key] = v
            else:
                # Pass through any unrecognized weights as-is
                new_weights[k] = v

        return new_weights

    def model_quant_predicate(self, p, m):
        """Skip quantization on encoder norms, ada norms, embeddings."""
        skip_patterns = [
            "norm",
            "ada_rms_norm",
            "tok_embeddings",
            "conv_layers",
            "audio_language_projection",
        ]
        return not any(pat in p for pat in skip_patterns)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Initialize tokenizer and precompute ada scales after weight loading."""
        model_path = Path(model_path)

        # Load Tekken tokenizer
        model._tokenizer = TekkenTokenizer.from_model_path(model_path)
        tekken_path = model_path / "tekken.json"
        if MistralTokenizer is not None and tekken_path.exists():
            model._mistral_tokenizer = _load_mistral_tokenizer_compat(tekken_path)

        # Precompute mel filters
        model._ensure_mel_filters()

        # Precompute ada scales from time conditioning (tracked for runtime override)
        n_delay = _num_delay_tokens(model.config.transcription_delay_ms)
        t_cond = compute_time_embedding(float(n_delay), model.config.decoder.dim)
        model.decoder.precompute_ada_scales(t_cond)
        model._ada_scale_delay = n_delay
        # Evaluate ada scales eagerly
        for scale in model.decoder._ada_scales:
            if scale is not None:
                mx.eval(scale)

        return model
