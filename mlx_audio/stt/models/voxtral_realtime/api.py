from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, cast

import mlx.core as mx
import numpy as np
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.utils import apply_quantization, get_model_path, load_config

try:
    from mistral_common.audio import Audio
    from mistral_common.protocol.transcription.request import (
        RawAudio,
        StreamingMode,
        TranscriptionRequest,
    )
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    Audio = None
    RawAudio = None
    StreamingMode = None
    TranscriptionRequest = None
    SpecialTokenPolicy = None
    MistralTokenizer = Any

from .audio import StreamingBuffer, compute_log_mel
from .config import VoxtralRealtimeConfig
from .model import VoxtralModel


def _require_mistral_common() -> None:
    if Audio is None or TranscriptionRequest is None or MistralTokenizer is Any:
        raise ImportError(
            "Voxtral requires mistral-common[audio]. Install with: pip install 'mlx-audio[stt]'"
        )


def _get_streaming_mode(name: str):
    _require_mistral_common()
    return getattr(StreamingMode, name)


@dataclass(frozen=True)
class RealtimeTokenEvent:
    token_id: int
    text: str


@dataclass
class _RealtimeSession:
    runtime: "VoxtralRealtime"
    language: Optional[str]
    max_tokens: int
    realtime_chunk_multiple: int
    buffer: StreamingBuffer
    text_cache: List[Any]
    audio_transformer_cache: Any
    downsample: int
    pending_chunk_remainder: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    stream_samples: int = 0
    first_iteration: bool = True
    last_token: Optional[int] = None
    prompt_dtype: Any = mx.int32
    generated_total: int = 0
    decoded_history: List[int] = field(default_factory=list)
    emitted_text: str = ""

    @property
    def exhausted(self) -> bool:
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

        # When the audio stream ends, add the offline right padding tokens the
        # tokenizer would have applied in OFFLINE mode. Without this, realtime
        # decoding can miss the delayed tail of the transcript.
        audio_encoder = self.runtime.tokenizer.instruct_tokenizer.audio_encoder
        if audio_encoder is not None:
            audio_cfg = audio_encoder.audio_config
            raw_len = int(getattr(audio_cfg, "raw_audio_length_per_tok", 0) or 0)
            right_tokens = int(getattr(audio_cfg, "n_right_pad_tokens", 0) or 0)
            if raw_len > 0:
                align_pad = (raw_len - (self.stream_samples % raw_len)) % raw_len
                extra_pad = raw_len * right_tokens
                total_pad = int(align_pad + extra_pad)
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

    def _consume_segment(self, segment: np.ndarray) -> List[RealtimeTokenEvent]:
        if self.exhausted:
            return []

        if self.first_iteration:
            input_ids, audio_arrays = self.runtime._prepare_inputs(
                segment,
                self.language,
                streaming_mode=_get_streaming_mode("ONLINE"),
            )
            step_audio = audio_arrays[0]
            self.prompt_dtype = input_ids.dtype
            if int(input_ids.shape[0]) > 0:
                self.last_token = int(input_ids[-1].item())
        else:
            continuation_token = self.last_token
            if continuation_token is None:
                return []
            step_audio = segment.astype(np.float32)
            # Realtime continuation consumes only the previous generated token.
            input_ids = mx.array([int(continuation_token)], dtype=self.prompt_dtype)

        conv_features = self.runtime._prepare_audio_conv_features(
            [step_audio],
            center=True,
            truncate_left_for_realtime=True,
        )
        segment_token_budget = self.runtime._resolve_max_tokens(
            1,
            prompt_len=int(input_ids.shape[0]),
            audio_samples=int(step_audio.shape[0]),
            conv_length=int(conv_features.shape[0]),
        )
        if self.max_tokens > 0:
            segment_token_budget = min(segment_token_budget, self.max_tokens - self.generated_total)
        if segment_token_budget <= 0:
            self.first_iteration = False
            return []

        tokens = self.runtime._generate_audio_conditioned(
            input_ids,
            conv_features=conv_features,
            max_tokens=segment_token_budget,
            text_cache=self.text_cache,
            audio_cache=self.audio_transformer_cache,
            use_stateful_audio_cache=True,
        )
        self.first_iteration = False

        events: List[RealtimeTokenEvent] = []
        for token in tokens:
            self.last_token = token
            self.generated_total += 1
            self.decoded_history.append(token)
            raw_text = self.runtime.tokenizer.decode(
                self.decoded_history,
                special_token_policy=self.runtime._decode_policy,
            )
            if raw_text.startswith(self.emitted_text):
                delta = raw_text[len(self.emitted_text) :]
            else:
                delta = raw_text
            self.emitted_text = raw_text
            events.append(RealtimeTokenEvent(token_id=token, text=delta))
        return events


class VoxtralRealtime:
    def __init__(
        self,
        model: VoxtralModel,
        tokenizer: MistralTokenizer,
        config: VoxtralRealtimeConfig,
    ):
        _require_mistral_common()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        audio_encoder = tokenizer.instruct_tokenizer.audio_encoder
        if audio_encoder is None:
            raise ValueError("Tokenizer is missing audio encoder configuration")
        num_delay_tokens = getattr(audio_encoder.audio_config, "num_delay_tokens", None)
        delay_tokens = int(num_delay_tokens) if num_delay_tokens is not None else 6
        self.audio_token_id = audio_encoder.special_ids.audio
        eos_ids = getattr(tokenizer.instruct_tokenizer.tokenizer, "eos_token_ids", None)
        if not eos_ids:
            eos_id = getattr(tokenizer.instruct_tokenizer.tokenizer, "eos_id", 2)
            eos_ids = [int(eos_id)]
        self._eos_token_ids = [int(x) for x in eos_ids]
        self._decode_policy = (
            SpecialTokenPolicy.IGNORE if SpecialTokenPolicy is not None else None
        )
        if config.text.ada_rms_norm_t_cond:
            embed_weight = self.model.language_model.model.embed_tokens.weight
            self.model._t_cond = _time_embedding(
                mx.array([delay_tokens], dtype=mx.float32),
                dim=config.text.hidden_size,
            ).astype(embed_weight.dtype)

    @classmethod
    def load(
        cls,
        model_id_or_path: str,
        dtype: str = "fp16",
        quantize_bits: Optional[int] = None,
        quantize_group_size: int = 32,
        revision: Optional[str] = None,
    ) -> "VoxtralRealtime":
        return cls._load_uncached(
            model_id_or_path=model_id_or_path,
            dtype=dtype,
            quantize_bits=quantize_bits,
            quantize_group_size=quantize_group_size,
            revision=revision,
        )

    @classmethod
    def clear_cache(cls) -> None:
        mx.clear_cache()

    @classmethod
    def cache_info(cls) -> dict[str, Any]:
        return {"size": 0, "max_size": 0, "keys": []}

    @classmethod
    def _load_uncached(
        cls,
        model_id_or_path: str,
        dtype: str = "fp16",
        quantize_bits: Optional[int] = None,
        quantize_group_size: int = 32,
        revision: Optional[str] = None,
    ) -> "VoxtralRealtime":
        from mlx_audio.stt.utils import load as load_stt_model

        model: Optional[Any]
        load_error: Optional[Exception] = None
        try:
            model = load_stt_model(model_id_or_path, revision=revision)
        except Exception as exc:  # pragma: no cover - exercised via fallback test
            model = None
            load_error = exc

        if model is None or not isinstance(model, VoxtralModel):
            model_path = get_model_path(model_id_or_path, revision=revision)
            raw_cfg = load_config(model_path)
            if str(raw_cfg.get("model_type", "")).lower() == "voxtral":
                model = _load_with_forced_voxtral_realtime_tag(
                    model_path, revision=revision
                )
            elif load_error is not None:
                raise load_error

        if not isinstance(model, VoxtralModel):
            raise TypeError(
                f"Expected a VoxtralRealtime-compatible model, got {type(model).__name__}"
            )

        needs_runtime_refresh = False
        if dtype != "fp16" or quantize_bits is not None:
            _maybe_cast_model(model, dtype)
            needs_runtime_refresh = True
        if quantize_bits is not None:
            _maybe_quantize_model(
                model,
                bits=quantize_bits,
                group_size=quantize_group_size,
            )
            needs_runtime_refresh = True

        if needs_runtime_refresh:
            # Match shared loader behavior: materialize updated params after
            # cast/quantize so lazy buffers from prior weights do not linger.
            mx.eval(model.parameters())
            if quantize_bits is not None:
                mx.clear_cache()

        runtime_getter = getattr(model, "_get_runtime", None)
        if not callable(runtime_getter):
            raise RuntimeError(
                "Loaded VoxtralRealtime model is missing runtime loader. "
                "Use mlx_audio.stt.load() compatible converted weights."
            )

        if needs_runtime_refresh and hasattr(model, "_runtime"):
            setattr(model, "_runtime", None)

        runtime = runtime_getter()
        if isinstance(runtime, cls):
            return runtime

        tokenizer = getattr(model, "_tokenizer", None)
        runtime_cfg = getattr(getattr(model, "config", None), "runtime", None)
        if tokenizer is None or runtime_cfg is None:
            raise RuntimeError("Voxtral runtime is missing tokenizer or runtime config")

        runtime = cls(model=model, tokenizer=tokenizer, config=runtime_cfg)
        if hasattr(model, "_runtime"):
            setattr(model, "_runtime", runtime)
        return runtime

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        streaming: bool = False,
        max_tokens: int = 0,
    ) -> STTOutput:
        del streaming
        try:
            input_ids, audio_arrays = self._prepare_inputs(
                audio,
                language,
                streaming_mode=_get_streaming_mode("OFFLINE"),
            )
            conv_features = self._prepare_audio_conv_features(audio_arrays, center=True)
            max_tokens = self._resolve_max_tokens(
                max_tokens,
                prompt_len=int(input_ids.shape[0]),
                audio_samples=int(audio_arrays[0].shape[0]),
                conv_length=int(conv_features.shape[0]),
            )
            tokens = self._generate_audio_conditioned(
                input_ids, conv_features=conv_features, max_tokens=max_tokens
            )
            text = self.tokenizer.decode(tokens, special_token_policy=self._decode_policy)
            return STTOutput(
                text=text,
                prompt_tokens=len(input_ids),
                generation_tokens=len(tokens),
            )
        finally:
            # Large temporary attention buffers can remain in MLX's memory cache
            # after decode. Clear allocator cache between requests/sessions.
            mx.clear_cache()

    def stream_transcribe(
        self,
        audio_iter: Iterable[np.ndarray],
        language: Optional[str] = None,
        max_tokens: int = 0,
        strategy: str = "realtime",
    ) -> Generator[str, None, None]:
        for event in self.stream_transcribe_tokens(
            audio_iter,
            language=language,
            max_tokens=max_tokens,
            strategy=strategy,
        ):
            if event.text:
                yield event.text

    def stream_transcribe_tokens(
        self,
        audio_iter: Iterable[np.ndarray],
        language: Optional[str] = None,
        max_tokens: int = 0,
        strategy: str = "realtime",
    ) -> Generator[RealtimeTokenEvent, None, None]:
        try:
            audio_encoder = self.tokenizer.instruct_tokenizer.audio_encoder
            if audio_encoder is None:
                raise ValueError("Tokenizer is missing audio encoder configuration")
            if strategy != "realtime":
                raise ValueError(
                    f"Unsupported streaming strategy: {strategy}. "
                    "VoxtralRealtime supports only 'realtime' for vLLM parity."
                )

            from mlx_lm.models import cache as lm_cache

            audio_config = audio_encoder.audio_config
            realtime_chunk_multiple = int(
                abs((int(self.config.audio.window_size) // 2) - int(self.config.audio.hop_length))
            )
            if realtime_chunk_multiple <= 0:
                raise ValueError("Realtime chunk multiple must be > 0")
            downsample = int(self.config.audio.downsample_factor)
            if downsample <= 0:
                raise ValueError("downsample_factor must be > 0")

            session = _RealtimeSession(
                runtime=self,
                language=language,
                max_tokens=max_tokens,
                realtime_chunk_multiple=realtime_chunk_multiple,
                buffer=StreamingBuffer(
                    sampling_rate=audio_config.sampling_rate,
                    frame_rate=audio_config.frame_rate,
                    transcription_delay_ms=float(audio_config.transcription_delay_ms or 0.0),
                    streaming_look_ahead_ms=float(
                        getattr(audio_config, "streaming_look_ahead_ms", 0.0) or 0.0
                    ),
                    streaming_look_back_ms=float(
                        getattr(audio_config, "streaming_look_back_ms", 0.0) or 0.0
                    ),
                ),
                text_cache=lm_cache.make_prompt_cache(self.model),
                audio_transformer_cache=self.model.audio_encoder.make_cache(),
                downsample=downsample,
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

    def _prepare_inputs(
        self,
        audio: np.ndarray,
        language: Optional[str],
        streaming_mode: Any = None,
    ) -> Tuple[mx.array, List[np.ndarray]]:
        _require_mistral_common()
        if streaming_mode is None:
            streaming_mode = _get_streaming_mode("OFFLINE")
        audio_obj = Audio(
            audio.astype(np.float32), self.config.audio.sampling_rate, format="wav"
        )
        req = TranscriptionRequest(
            model=self.config.raw.get("model", "voxtral"),
            audio=RawAudio.from_audio(audio_obj),
            language=cast(Any, language),
            streaming=streaming_mode,
        )
        tokenized = self.tokenizer.instruct_tokenizer.encode_transcription(req)
        input_ids = mx.array(tokenized.tokens)
        audio_arrays = [tokenized.audios[0].audio_array]
        return input_ids, audio_arrays

    def _prepare_audio_conv_features(
        self,
        audio_arrays: List[np.ndarray],
        center: bool,
        truncate_left_for_realtime: bool = False,
        padding_cache: Optional[Any] = None,
    ) -> mx.array:
        mel = compute_log_mel(audio_arrays[0], self.config.audio, center=center)
        if padding_cache is None:
            conv_stride = int(self.model.audio_encoder.total_stride)
            mel_remainder = int(mel.shape[1] % conv_stride)
            if mel_remainder:
                mel = mel[:, mel_remainder:]

            mel_frames = int(mel.shape[1])
            if mel_frames <= 0:
                return mx.zeros((0, int(self.config.audio.d_model)), dtype=mx.float32)

            # Match vLLM's long-input handling: process conv features in bounded
            # chunks instead of one giant pass to reduce peak workspace usage.
            chunk_size = max(
                1, int(self.config.audio.max_source_positions) * conv_stride
            )
            conv_chunks: List[mx.array] = []
            for start in range(0, mel_frames, chunk_size):
                mel_chunk = mel[:, start : start + chunk_size]
                if mel_chunk.shape[1] <= 0:
                    continue
                conv_chunk, _ = self.model.audio_encoder.forward_conv_features(
                    mx.array(mel_chunk)
                )
                mx.eval(conv_chunk)
                conv_chunks.append(conv_chunk)

            if not conv_chunks:
                return mx.zeros((0, int(self.config.audio.d_model)), dtype=mx.float32)
            if len(conv_chunks) == 1:
                conv = conv_chunks[0]
            else:
                conv = mx.concatenate(conv_chunks, axis=0)

            downsample = int(self.config.audio.downsample_factor)
            conv_remainder = int(conv.shape[0] % downsample)
            if conv_remainder:
                conv = conv[conv_remainder:]
            return conv

        conv, _ = self.model.audio_encoder.forward_conv_features(
            mx.array(mel), padding_cache=padding_cache
        )
        if truncate_left_for_realtime:
            downsample = int(self.config.audio.downsample_factor)
            conv_remainder = int(conv.shape[0] % downsample)
            if conv_remainder:
                conv = conv[conv_remainder:]
        return conv

    def _resolve_max_tokens(
        self,
        max_tokens: int,
        *,
        prompt_len: int,
        audio_samples: int,
        conv_length: Optional[int] = None,
    ) -> int:
        downsample = int(self.config.audio.downsample_factor)
        if downsample <= 0:
            raise ValueError("downsample_factor must be > 0")

        audio_encoder = self.tokenizer.instruct_tokenizer.audio_encoder
        if audio_encoder is None:
            raise ValueError("Tokenizer is missing audio encoder configuration")
        audio_cfg = audio_encoder.audio_config
        audio_steps = max(int(audio_cfg.num_audio_tokens(int(audio_samples))), 0)
        if conv_length is not None:
            conv_steps = max(int(conv_length) // downsample, 0)
            audio_steps = min(audio_steps, conv_steps)

        # Our decode emits the next token based on the final prompt position, so the
        # number of audio-conditioned decoding steps available for generation is:
        #   audio_steps - prompt_len + 1
        model_limit = max(audio_steps - int(prompt_len) + 1, 0)
        if max_tokens <= 0:
            return model_limit
        return min(max_tokens, model_limit)

    def _generate_audio_conditioned(
        self,
        prompt_ids: mx.array,
        conv_features: mx.array,
        max_tokens: int,
        text_cache: Optional[List[Any]] = None,
        audio_cache: Optional[List[Any]] = None,
        use_stateful_audio_cache: bool = False,
    ) -> List[int]:
        from mlx_lm.models import cache as lm_cache
        prompt_cache = (
            text_cache
            if text_cache is not None
            else lm_cache.make_prompt_cache(self.model)
        )
        audio_cache = (
            audio_cache
            if audio_cache is not None
            else self.model.audio_encoder.make_cache()
        )
        prompt_cache_offset_tokens = _cache_offset_tokens(prompt_cache)
        if use_stateful_audio_cache:
            audio_position_offset_frames = 0
        else:
            audio_position_offset_frames = int(
                prompt_cache_offset_tokens * self.config.audio.downsample_factor
            )
        prompt_len = int(prompt_ids.shape[0])
        if prompt_len <= 0:
            return []
        t_cond = getattr(self.model, "_t_cond", None)
        try:
            prefill_eval_interval = max(
                1, int(os.getenv("MLX_AUDIO_VOXTRAL_PREFILL_EVAL_INTERVAL", "8"))
            )
        except ValueError:
            prefill_eval_interval = 8

        def _audio_embed_at(position: int) -> Optional[mx.array]:
            step = self.config.audio.downsample_factor
            start = position * step
            end = start + step
            if start >= conv_features.shape[0]:
                return None
            chunk = conv_features[start:end]
            if chunk.shape[0] < step:
                pad = step - chunk.shape[0]
                chunk = mx.concatenate(
                    [chunk, mx.zeros((pad, chunk.shape[1]), dtype=chunk.dtype)],
                    axis=0,
                )
            transformed, _ = self.model.audio_encoder.forward_transformer(
                chunk,
                cache=audio_cache,
                position_offset=audio_position_offset_frames,
            )
            transformed = transformed.reshape(1, self.config.audio.encoder_ffn_dim)
            return self.model.audio_language_adapter(transformed)

        def _prefill_step(input_token: mx.array, input_embedding: mx.array) -> mx.array:
            return self.model.language_model.model(
                input_token[None],
                cache=prompt_cache,
                input_embeddings=input_embedding[None],
                t_cond=t_cond,
            )

        def _model_logits(input_token: mx.array, input_embedding: mx.array) -> mx.array:
            hidden_states = self.model.language_model.model(
                input_token[None],
                cache=prompt_cache,
                input_embeddings=input_embedding[None],
                t_cond=t_cond,
            )
            logits = self.model.language_model.model.embed_tokens.as_linear(
                hidden_states[:, -1, :]
            )
            return logits

        for idx in range(max(prompt_len - 1, 0)):
            tok = prompt_ids[idx : idx + 1]
            emb = self.model.language_model.embed_input_ids(tok)
            audio_embed = _audio_embed_at(idx)
            if audio_embed is None:
                return []
            emb = emb + audio_embed
            hidden_states = _prefill_step(tok, emb)
            if idx % prefill_eval_interval == prefill_eval_interval - 1:
                mx.eval(hidden_states)
        if prompt_len > 1:
            mx.eval(hidden_states)

        last_tok_idx = prompt_len - 1
        last_tok = prompt_ids[last_tok_idx : last_tok_idx + 1]
        last_emb = self.model.language_model.embed_input_ids(last_tok)
        audio_embed = _audio_embed_at(last_tok_idx)
        if audio_embed is None:
            return []
        last_emb = last_emb + audio_embed
        logits = _model_logits(last_tok, last_emb)
        token = int(mx.argmax(logits, axis=-1).item())

        generated: List[int] = []
        position = prompt_len
        for step in range(max_tokens):
            if token in self._eos_token_ids:
                break
            generated.append(token)
            # Avoid an unnecessary extra decode step on the final iteration.
            # Keeping cache growth aligned with emitted tokens is required for
            # cross-call streaming parity.
            if step == max_tokens - 1:
                break
            input_token = mx.array([token], dtype=prompt_ids.dtype)
            token_embed = self.model.language_model.embed_input_ids(input_token)
            audio_embed = _audio_embed_at(position)
            if audio_embed is None:
                break
            token_embed = token_embed + audio_embed
            logits = _model_logits(input_token, token_embed)
            token = int(mx.argmax(logits, axis=-1).item())
            position += 1

        return generated


def load_tokenizer(model_id_or_path: str, revision: Optional[str]) -> MistralTokenizer:
    _require_mistral_common()
    model_path = get_model_path(
        model_id_or_path,
        revision=revision,
        allow_patterns=[
            "tekken*.json",
            "tokenizer*.json",
            "config.json",
            "params.json",
        ],
    )
    for candidate in model_path.rglob("tekken*.json"):
        return MistralTokenizer.from_file(str(candidate))
    if Path(model_id_or_path).exists():
        raise FileNotFoundError("No tekken*.json found in local model path")
    return MistralTokenizer.from_hf_hub(model_id_or_path, revision=revision)


def _maybe_cast_model(model: VoxtralModel, dtype: str) -> None:
    if dtype == "fp16":
        target = mx.float16
    elif dtype == "bf16":
        target = mx.bfloat16
    elif dtype == "fp32":
        target = mx.float32
    else:
        return
    from mlx.utils import tree_flatten, tree_unflatten

    flat = tree_flatten(model.parameters())
    casted = [(name, value.astype(target)) for name, value in flat]
    model.update(tree_unflatten(casted))


def _maybe_quantize_model(
    model: VoxtralModel,
    bits: Optional[int],
    group_size: int,
) -> None:
    if bits is None:
        return

    from mlx.utils import tree_flatten

    # Reuse shared quantization path by providing pseudo scale entries for
    # modules we want to quantize at runtime.
    fake_quant_scales = {}
    for name, weight in tree_flatten(model.parameters()):
        if not (name.startswith("language_model.") and name.endswith(".weight")):
            continue
        if group_size > 0 and int(weight.shape[-1]) % group_size != 0:
            continue
        module_path = name[: -len(".weight")]
        fake_quant_scales[f"{module_path}.scales"] = mx.array(0.0, dtype=mx.float32)

    apply_quantization(
        model=model,
        config={"quantization": {"bits": int(bits), "group_size": int(group_size)}},
        weights=fake_quant_scales,
        model_quant_predicate=getattr(model, "model_quant_predicate", None),
    )


def _time_embedding(t: mx.array, dim: int, theta: float = 10000.0) -> mx.array:
    half = dim // 2
    inv_freq = mx.exp(-math.log(theta) * mx.arange(half) / half)
    emb = t[:, None] * inv_freq[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


def _cache_offset_tokens(cache: Optional[List[Any]]) -> int:
    if not cache:
        return 0
    first = cache[0]
    if first is None:
        return 0
    try:
        return int(getattr(first, "offset", 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _replace_audio_placeholders(
    input_ids: mx.array,
    text_embeds: mx.array,
    audio_embeds: mx.array,
    audio_token_id: int,
) -> mx.array:
    audio_mask = cast(mx.array, input_ids == int(audio_token_id))
    num_slots = int(mx.sum(audio_mask.astype(mx.int32)).item())
    if num_slots <= 0:
        return text_embeds

    audio_embeds = _fit_audio_embeddings_to_slots(audio_embeds, num_slots)

    slot_idx = mx.cumsum(audio_mask.astype(mx.int32), axis=0) - 1
    slot_idx = mx.maximum(slot_idx, 0)
    slot_idx = mx.minimum(slot_idx, num_slots - 1)
    expanded_audio = audio_embeds[slot_idx]
    return mx.where(audio_mask[:, None], expanded_audio, text_embeds)


def _fit_audio_embeddings_to_slots(audio_embeds: mx.array, num_slots: int) -> mx.array:
    slot_count = int(audio_embeds.shape[0])
    if slot_count == num_slots:
        return audio_embeds
    if slot_count > num_slots:
        return audio_embeds[:num_slots]

    pad = num_slots - slot_count
    return mx.concatenate(
        [
            audio_embeds,
            mx.zeros((pad, audio_embeds.shape[-1]), dtype=audio_embeds.dtype),
        ],
        axis=0,
    )


def _build_audio_slot_embeddings(
    num_slots: int,
    embed_dim: int,
    dtype: Any,
    get_slot_embed: Callable[[int], Optional[mx.array]],
) -> mx.array:
    if num_slots <= 0:
        return mx.zeros((0, embed_dim), dtype=dtype)

    slot_embeds: List[mx.array] = []
    for slot_idx in range(num_slots):
        slot_embed = get_slot_embed(slot_idx)
        if slot_embed is None:
            slot_embed = mx.zeros((1, embed_dim), dtype=dtype)
        slot_embeds.append(slot_embed)
    return mx.concatenate(slot_embeds, axis=0)


def _should_fallback_realtime(
    recent_tokens: List[int], control_token_ids: set[int]
) -> bool:
    window = 24
    if len(recent_tokens) < window:
        return False

    tokens = recent_tokens[-window:]
    control = sum(1 for tok in tokens if tok in control_token_ids)
    lexical_tokens = [tok for tok in tokens if tok not in control_token_ids]
    lexical_count = len(lexical_tokens)

    if lexical_count == 0:
        return control / float(window) >= 0.85

    unique_lexical = len(set(lexical_tokens))
    longest_lexical_run = 0
    run = 0
    prev: Optional[int] = None
    for tok in tokens:
        if tok in control_token_ids:
            prev = None
            run = 0
            continue
        if tok == prev:
            run += 1
        else:
            run = 1
            prev = tok
        longest_lexical_run = max(longest_lexical_run, run)

    control_ratio = control / float(window)
    return (
        (control_ratio >= 0.8 and unique_lexical <= 2)
        or (lexical_count >= 8 and unique_lexical <= 2)
        or longest_lexical_run >= 4
    )


def _load_with_forced_voxtral_realtime_tag(
    model_path: Path, revision: Optional[str]
) -> VoxtralModel:
    from mlx_audio.stt.utils import load as load_stt_model

    raw_cfg = load_config(model_path)
    patched_cfg = dict(raw_cfg)
    patched_cfg["model_type"] = "voxtral_realtime"

    with TemporaryDirectory(prefix="voxtral-realtime-load-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "config.json").write_text(
            json.dumps(patched_cfg), encoding="utf-8"
        )

        for filename in ("consolidated.safetensors", "tekken.json", "params.json"):
            src = model_path / filename
            if src.exists():
                (tmp_path / filename).symlink_to(src.resolve())

        loaded = load_stt_model(str(tmp_path), revision=revision)
        if not isinstance(loaded, VoxtralModel):
            raise TypeError(
                "Forced voxtral_realtime load did not return a VoxtralModel "
                f"(got {type(loaded).__name__})"
            )
        return loaded
