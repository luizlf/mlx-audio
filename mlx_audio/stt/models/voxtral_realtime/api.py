from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.utils import get_model_path

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
from .config import VoxtralConfig
from .model import VoxtralModel


@dataclass
class STTOutput:
    text: str
    prompt_tokens: int
    generation_tokens: int


def _require_mistral_common() -> None:
    if Audio is None or TranscriptionRequest is None or MistralTokenizer is Any:
        raise ImportError(
            "Voxtral requires mistral-common[audio]. Install with: pip install 'mlx-audio[stt]'"
        )


def _get_streaming_mode(name: str):
    _require_mistral_common()
    return getattr(StreamingMode, name)


class VoxtralRealtime:
    def __init__(
        self,
        model: VoxtralModel,
        tokenizer: MistralTokenizer,
        config: VoxtralConfig,
    ):
        _require_mistral_common()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        audio_encoder = tokenizer.instruct_tokenizer.audio_encoder
        if audio_encoder is None:
            raise ValueError("Tokenizer is missing audio encoder configuration")
        num_delay_tokens = getattr(audio_encoder.audio_config, "num_delay_tokens", None)
        self._num_delay_tokens = int(num_delay_tokens) if num_delay_tokens is not None else 6
        self.audio_token_id = audio_encoder.special_ids.audio
        self.begin_audio_token_id = audio_encoder.special_ids.begin_audio
        self.streaming_pad_token_id = getattr(audio_encoder.special_ids, "streaming_pad", None)
        self.streaming_word_token_id = (
            int(self.streaming_pad_token_id) + 1 if self.streaming_pad_token_id is not None else None
        )
        self.repeat_audio_text_token_id = (
            int(self.streaming_pad_token_id) + 2 if self.streaming_pad_token_id is not None else None
        )
        self._stream_control_token_ids = {
            int(token_id)
            for token_id in (
                self.streaming_pad_token_id,
                self.streaming_word_token_id,
                self.repeat_audio_text_token_id,
            )
            if token_id is not None
        }
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
                mx.array([self._num_delay_tokens], dtype=mx.float32),
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
        from mlx_audio.stt.utils import load as load_stt_model

        model = load_stt_model(model_id_or_path, revision=revision)
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
        language: str = "en",
        streaming: bool = False,
        max_tokens: int = 256,
    ) -> STTOutput:
        del streaming
        input_ids, audio_arrays = self._prepare_inputs(
            audio,
            language,
            streaming_mode=_get_streaming_mode("OFFLINE"),
        )
        conv_features = self._prepare_audio_conv_features(audio_arrays, center=True)
        max_tokens = self._resolve_max_tokens(max_tokens, conv_features.shape[0])
        tokens = self._generate_audio_conditioned(
            input_ids, conv_features=conv_features, max_tokens=max_tokens
        )
        text = self.tokenizer.decode(tokens, special_token_policy=self._decode_policy)
        return STTOutput(
            text=text,
            prompt_tokens=len(input_ids),
            generation_tokens=len(tokens),
        )

    def stream_transcribe(
        self,
        audio_iter: Iterable[np.ndarray],
        language: str = "en",
        max_tokens: int = 256,
        strategy: str = "stable",
        offline_refresh_reads: int = 50,
        realtime_carry_policy: str = "exact",
        realtime_auto_fallback: bool = True,
        realtime_fallback_refresh_reads: int = 50,
    ) -> Generator[str, None, None]:
        audio_encoder = self.tokenizer.instruct_tokenizer.audio_encoder
        if audio_encoder is None:
            raise ValueError("Tokenizer is missing audio encoder configuration")
        audio_config = audio_encoder.audio_config
        if strategy not in {"stable", "realtime"}:
            raise ValueError(f"Unsupported streaming strategy: {strategy}")
        if realtime_carry_policy not in {"exact", "lexical"}:
            raise ValueError(
                f"Unsupported realtime carry policy: {realtime_carry_policy}"
            )
        if realtime_fallback_refresh_reads <= 0:
            raise ValueError("realtime_fallback_refresh_reads must be > 0")

        if strategy == "stable":
            if offline_refresh_reads <= 0:
                raise ValueError("offline_refresh_reads must be > 0")
            buffer = StreamingBuffer(
                sampling_rate=audio_config.sampling_rate,
                frame_rate=audio_config.frame_rate,
                transcription_delay_ms=float(audio_config.transcription_delay_ms or 0.0),
                streaming_look_ahead_ms=float(audio_config.streaming_look_ahead_ms or 0.0),
                streaming_look_back_ms=float(audio_config.streaming_look_back_ms or 0.0),
            )
            committed_text = ""
            best_text = ""
            accumulated_chunks: List[np.ndarray] = []
            read_count = 0
            refresh_count = 0
            warmup_refreshes = 1

            for chunk in audio_iter:
                chunk_f32 = chunk.astype(np.float32)
                accumulated_chunks.append(chunk_f32)
                buffer.write(chunk_f32)
                while buffer.read() is not None:
                    read_count += 1
                    if read_count % offline_refresh_reads != 0:
                        continue
                    full_audio = np.concatenate(accumulated_chunks)
                    out = self.transcribe(full_audio, language=language, max_tokens=max_tokens)
                    refresh_count += 1
                    if len(out.text) >= len(best_text):
                        best_text = out.text
                    if refresh_count <= warmup_refreshes:
                        continue
                    if best_text.startswith(committed_text):
                        delta = best_text[len(committed_text) :]
                        if delta:
                            committed_text = best_text
                            yield delta

            if accumulated_chunks:
                full_audio = np.concatenate(accumulated_chunks)
                out = self.transcribe(full_audio, language=language, max_tokens=max_tokens)
                if len(out.text) >= len(best_text):
                    best_text = out.text
                if best_text.startswith(committed_text):
                    delta = best_text[len(committed_text) :]
                else:
                    committed_text, delta = _merge_stream_text(committed_text, best_text)
                if delta:
                    yield delta
            return

        buffer = StreamingBuffer(
            sampling_rate=audio_config.sampling_rate,
            frame_rate=audio_config.frame_rate,
            transcription_delay_ms=float(audio_config.transcription_delay_ms or 0.0),
            streaming_look_ahead_ms=float(audio_config.streaming_look_ahead_ms or 0.0),
            streaming_look_back_ms=float(audio_config.streaming_look_back_ms or 0.0),
        )
        from mlx_lm.models import cache as lm_cache

        text_cache = lm_cache.make_prompt_cache(self.model)
        first_iteration = True
        last_token: Optional[int] = None
        carry_token: Optional[int] = None
        prompt_dtype = mx.int32
        generated_total = 0
        decoded_history: List[int] = []
        emitted = ""
        recent_tokens: List[int] = []
        accumulated_chunks: List[np.ndarray] = []
        fallback_mode = False
        fallback_read_count = 0
        fallback_refresh_count = 0
        fallback_best_text = ""
        fallback_warmup_refreshes = 1
        for chunk in audio_iter:
            chunk_f32 = chunk.astype(np.float32)
            accumulated_chunks.append(chunk_f32)
            if fallback_mode:
                fallback_read_count += 1
                if fallback_read_count % realtime_fallback_refresh_reads != 0:
                    continue
                full_audio = np.concatenate(accumulated_chunks)
                out = self.transcribe(full_audio, language=language, max_tokens=max_tokens)
                fallback_refresh_count += 1
                if len(out.text) >= len(fallback_best_text):
                    fallback_best_text = out.text
                if fallback_refresh_count <= fallback_warmup_refreshes:
                    continue
                if fallback_best_text.startswith(emitted):
                    delta = fallback_best_text[len(emitted) :]
                    if delta:
                        emitted = fallback_best_text
                        yield delta
                else:
                    emitted, delta = _merge_stream_text(emitted, fallback_best_text)
                    if delta:
                        yield delta
                continue

            if max_tokens > 0 and generated_total >= max_tokens:
                break
            buffer.write(chunk_f32)
            while (segment := buffer.read()) is not None:
                if max_tokens > 0 and generated_total >= max_tokens:
                    break
                if first_iteration:
                    input_ids, audio_arrays = self._prepare_inputs(
                        segment,
                        language,
                        streaming_mode=_get_streaming_mode("ONLINE"),
                    )
                    step_audio = audio_arrays[0]
                    prompt_dtype = input_ids.dtype
                else:
                    if realtime_carry_policy == "lexical":
                        continuation_token = carry_token if carry_token is not None else last_token
                    else:
                        continuation_token = last_token
                    if continuation_token is None:
                        continue
                    step_audio = segment.astype(np.float32)
                    # vLLM realtime continuation uses only the last generated token.
                    input_ids = mx.array([int(continuation_token)], dtype=prompt_dtype)
                conv_features = self._prepare_audio_conv_features(
                    [step_audio],
                    center=True,
                    padding_cache=None,
                    truncate_left_for_realtime=True,
                )

                # vLLM realtime generates a single token per streaming step.
                segment_token_budget = 1
                if max_tokens > 0:
                    segment_token_budget = min(segment_token_budget, max_tokens - generated_total)
                if segment_token_budget <= 0:
                    first_iteration = False
                    continue

                tokens = self._generate_audio_conditioned(
                    input_ids,
                    conv_features=conv_features,
                    max_tokens=segment_token_budget,
                    text_cache=text_cache,
                    realtime_expand_single_token=True,
                )
                if tokens:
                    last_token = tokens[-1]
                    if realtime_carry_policy == "lexical":
                        if last_token not in self._stream_control_token_ids:
                            carry_token = last_token
                    else:
                        carry_token = last_token
                    generated_total += len(tokens)
                    decoded_history.extend(tokens)
                    recent_tokens.extend(int(t) for t in tokens)
                    if len(recent_tokens) > 128:
                        recent_tokens = recent_tokens[-128:]
                elif first_iteration:
                    last_token = int(input_ids[-1].item())
                first_iteration = False
                if realtime_auto_fallback and _should_fallback_realtime(
                    recent_tokens, self._stream_control_token_ids
                ):
                    fallback_mode = True
                    fallback_best_text = emitted
                    full_audio = np.concatenate(accumulated_chunks)
                    out = self.transcribe(full_audio, language=language, max_tokens=max_tokens)
                    if len(out.text) >= len(fallback_best_text):
                        fallback_best_text = out.text
                    emitted, delta = _merge_stream_text(emitted, fallback_best_text)
                    if delta:
                        yield delta
                    break

        if fallback_mode and accumulated_chunks:
            full_audio = np.concatenate(accumulated_chunks)
            out = self.transcribe(full_audio, language=language, max_tokens=max_tokens)
            if len(out.text) >= len(fallback_best_text):
                fallback_best_text = out.text
            emitted, delta = _merge_stream_text(emitted, fallback_best_text)
            if delta:
                yield delta
        elif accumulated_chunks:
            decode_fn = getattr(self.tokenizer, "decode", None)
            raw_text = ""
            if callable(decode_fn):
                raw_text = decode_fn(
                    decoded_history, special_token_policy=self._decode_policy
                )

            if _should_emit_realtime_text(
                raw_text, recent_tokens, self._stream_control_token_ids
            ):
                emitted, delta = _merge_stream_text(emitted, raw_text)
                if delta:
                    yield delta
            else:
                # Raw realtime can remain unstable for some clips; emit one final
                # coherent transcript instead of low-quality control-loop artifacts.
                full_audio = np.concatenate(accumulated_chunks)
                out = self.transcribe(full_audio, language=language, max_tokens=max_tokens)
                emitted, delta = _merge_stream_text(emitted, out.text)
                if delta:
                    yield delta

    def _prepare_inputs(
        self,
        audio: np.ndarray,
        language: str,
        streaming_mode: Any = None,
    ) -> Tuple[mx.array, List[np.ndarray]]:
        _require_mistral_common()
        if streaming_mode is None:
            streaming_mode = _get_streaming_mode("OFFLINE")
        audio_obj = Audio(audio.astype(np.float32), self.config.audio.sampling_rate, format="wav")
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

    def _merge_input_embeddings(
        self, input_ids: mx.array, audio_arrays: List[np.ndarray]
    ) -> mx.array:
        if not audio_arrays:
            return self.model.language_model.embed_input_ids(input_ids)

        audio_embeds = self._encode_audio(audio_arrays, center=True)
        return self._fuse_embeddings(input_ids, audio_embeds)

    def _prepare_realtime_embeddings(
        self,
        input_ids: mx.array,
        audio_arrays: List[np.ndarray],
        is_first_iteration: bool,
    ) -> mx.array:
        if not audio_arrays:
            return self.model.language_model.embed_input_ids(input_ids)

        audio_text_embeds = self._encode_audio(audio_arrays, center=is_first_iteration)
        return self._fuse_embeddings(input_ids, audio_text_embeds)

    def _encode_audio(self, audio_arrays: List[np.ndarray], center: bool) -> mx.array:
        mel_features = [
            compute_log_mel(audio, self.config.audio, center=center) for audio in audio_arrays
        ]
        mel_mx = [mx.array(mel) for mel in mel_features]
        hidden = self.model.audio_encoder(mel_mx)[0]
        downsample = self.config.audio.downsample_factor
        seq_len, dim = hidden.shape
        if seq_len % downsample != 0:
            pad = downsample - (seq_len % downsample)
            hidden = mx.concatenate([hidden, mx.zeros((pad, dim))], axis=0)
        hidden = hidden.reshape(-1, self.config.audio.encoder_ffn_dim)
        return self.model.audio_language_adapter(hidden)

    def _prepare_audio_conv_features(
        self,
        audio_arrays: List[np.ndarray],
        center: bool,
        padding_cache: Optional[Any] = None,
        truncate_left_for_realtime: bool = False,
    ) -> mx.array:
        mel = compute_log_mel(audio_arrays[0], self.config.audio, center=center)
        if truncate_left_for_realtime:
            conv_stride = int(self.model.audio_encoder.total_stride)
            mel_remainder = int(mel.shape[1] % conv_stride)
            if mel_remainder:
                mel = mel[:, mel_remainder:]
        conv, _ = self.model.audio_encoder.forward_conv_features(
            mx.array(mel), padding_cache=padding_cache
        )
        if truncate_left_for_realtime:
            downsample = int(self.config.audio.downsample_factor)
            conv_remainder = int(conv.shape[0] % downsample)
            if conv_remainder:
                conv = conv[conv_remainder:]
        return conv

    def _fuse_embeddings(self, input_ids: mx.array, audio_embeds: mx.array) -> mx.array:
        text_embeds = self.model.language_model.embed_input_ids(input_ids)
        return _replace_audio_placeholders(
            input_ids=input_ids,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            audio_token_id=int(self.audio_token_id),
        )

    def _resolve_max_tokens(self, max_tokens: int, audio_length: int) -> int:
        model_limit = int(
            math.ceil(
                (audio_length * self.config.audio.downsample_factor)
                / self.config.audio_length_per_tok
            )
        )
        if max_tokens <= 0:
            return model_limit
        return min(max_tokens, model_limit)

    def _generate_audio_conditioned(
        self,
        prompt_ids: mx.array,
        conv_features: mx.array,
        max_tokens: int,
        text_cache: Optional[List[Any]] = None,
        encoder_cache: Optional[List[Any]] = None,
        realtime_expand_single_token: bool = False,
        debug_steps: Optional[List[Dict[str, Any]]] = None,
        debug_top_k: int = 5,
    ) -> List[int]:
        from mlx_lm.models import cache as lm_cache
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(0.0, top_p=1.0, top_k=0)
        prompt_cache = (
            text_cache if text_cache is not None else lm_cache.make_prompt_cache(self.model)
        )
        audio_cache = (
            encoder_cache if encoder_cache is not None else self.model.audio_encoder.make_cache()
        )
        prompt_cache_offset_tokens = _cache_offset_tokens(prompt_cache)
        audio_position_offset_frames = int(
            prompt_cache_offset_tokens * self.config.audio.downsample_factor
        )
        prompt_len = int(prompt_ids.shape[0])
        if prompt_len <= 0:
            return []
        base_prompt_embeds = self.model.language_model.embed_input_ids(prompt_ids)
        audio_placeholder_mask = cast(mx.array, prompt_ids == int(self.audio_token_id))
        num_audio_slots = int(mx.sum(audio_placeholder_mask.astype(mx.int32)).item())

        audio_embeds_full: Optional[mx.array] = None
        if realtime_expand_single_token:
            transformed, _ = self.model.audio_encoder.forward_transformer(
                conv_features,
                cache=None,
                position_offset=audio_position_offset_frames,
            )
            downsample = int(self.config.audio.downsample_factor)
            seq_len, dim = transformed.shape
            if seq_len % downsample != 0:
                pad = downsample - (seq_len % downsample)
                transformed = mx.concatenate(
                    [transformed, mx.zeros((pad, dim), dtype=transformed.dtype)], axis=0
                )
            transformed = transformed.reshape(-1, self.config.audio.encoder_ffn_dim)
            audio_embeds_full = self.model.audio_language_adapter(transformed)

        effective_prompt_len = prompt_len
        # For single-token continuation with audio context, expand effective_prompt_len
        # to iterate through all audio positions during prefill (vLLM realtime parity).
        # Keep at least one prefill step even when audio collapses to a single pooled
        # position, otherwise generation stays locked to position 0 and emits pad tokens.
        if realtime_expand_single_token and audio_embeds_full is not None and prompt_len == 1:
            audio_steps = int(audio_embeds_full.shape[0])
            if audio_steps > 0:
                effective_prompt_len = max(audio_steps, 2)

        audio_position_shift = 0
        if audio_embeds_full is not None and effective_prompt_len > int(audio_embeds_full.shape[0]):
            audio_position_shift = effective_prompt_len - int(audio_embeds_full.shape[0])

        def _audio_embed_at(position: int) -> Optional[mx.array]:
            if audio_embeds_full is not None:
                audio_position = position - audio_position_shift
                if audio_position < 0 or audio_position >= audio_embeds_full.shape[0]:
                    return None
                return audio_embeds_full[audio_position : audio_position + 1]
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

        def _model_logits(input_token: mx.array, input_embedding: mx.array) -> mx.array:
            logits = self.model(
                input_token[None], cache=prompt_cache, input_embeddings=input_embedding[None]
            )
            return logits[:, -1, :]

        def _capture_debug(position: int, chosen_token: int, logprobs: mx.array) -> None:
            if debug_steps is None:
                return
            values = np.array(logprobs[0], dtype=np.float64)
            if values.ndim != 1 or values.size == 0:
                return
            top_k = max(1, min(int(debug_top_k), values.shape[0]))
            top_idx = np.argsort(values)[-top_k:][::-1]
            debug_steps.append(
                {
                    "position": int(position),
                    "chosen_token": int(chosen_token),
                    "top_ids": [int(i) for i in top_idx.tolist()],
                    "top_logprobs": [float(values[i]) for i in top_idx.tolist()],
                }
            )

        use_placeholder_fusion = num_audio_slots > 0
        prompt_embeds: Optional[mx.array] = None
        if use_placeholder_fusion:
            audio_for_slots: List[mx.array] = []
            for slot_idx in range(num_audio_slots):
                slot_audio_embed = _audio_embed_at(slot_idx)
                if slot_audio_embed is None:
                    slot_audio_embed = mx.zeros(
                        (1, base_prompt_embeds.shape[-1]), dtype=base_prompt_embeds.dtype
                    )
                audio_for_slots.append(slot_audio_embed)
            if audio_for_slots:
                packed_audio = mx.concatenate(audio_for_slots, axis=0)
            else:
                packed_audio = mx.zeros((0, base_prompt_embeds.shape[-1]), dtype=base_prompt_embeds.dtype)
            prompt_embeds = _replace_audio_placeholders(
                input_ids=prompt_ids,
                text_embeds=base_prompt_embeds,
                audio_embeds=packed_audio,
                audio_token_id=int(self.audio_token_id),
            )

        for idx in range(max(effective_prompt_len - 1, 0)):
            tok_idx = idx if prompt_len > 1 else 0
            tok = prompt_ids[tok_idx : tok_idx + 1]
            if prompt_embeds is not None and idx < prompt_len:
                emb = prompt_embeds[idx : idx + 1]
            else:
                emb = self.model.language_model.embed_input_ids(tok)
                audio_embed = _audio_embed_at(idx)
                if audio_embed is not None:
                    emb = emb + audio_embed
            _ = self.model(tok[None], cache=prompt_cache, input_embeddings=emb[None])

        last_tok_idx = min(effective_prompt_len - 1, prompt_len - 1)
        last_tok = prompt_ids[last_tok_idx : last_tok_idx + 1]
        if prompt_embeds is not None and last_tok_idx < prompt_len:
            last_emb = prompt_embeds[last_tok_idx : last_tok_idx + 1]
        else:
            last_emb = self.model.language_model.embed_input_ids(last_tok)
            audio_embed = _audio_embed_at(effective_prompt_len - 1)
            if audio_embed is not None:
                last_emb = last_emb + audio_embed
        logits = _model_logits(last_tok, last_emb)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = int(sampler(logprobs).item())
        token_logprobs = logprobs

        generated: List[int] = []
        position = effective_prompt_len
        for step in range(max_tokens):
            if token in self._eos_token_ids:
                break
            generated.append(token)
            _capture_debug(position, token, token_logprobs)
            # Avoid an unnecessary extra decode step on the final iteration.
            # Keeping cache growth aligned with emitted tokens is required for
            # cross-call streaming parity.
            if step == max_tokens - 1:
                break
            input_token = mx.array([token], dtype=prompt_ids.dtype)
            token_embed = self.model.language_model.embed_input_ids(input_token)
            if not use_placeholder_fusion:
                audio_embed = _audio_embed_at(position)
                if audio_embed is not None:
                    token_embed = token_embed + audio_embed
            logits = _model_logits(input_token, token_embed)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            token_logprobs = logprobs
            token = int(sampler(logprobs).item())
            position += 1

        return generated

    def _generate(
        self, input_ids: mx.array, input_embeddings: mx.array, max_tokens: int
    ) -> List[int]:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(0.0, top_p=0.95, top_k=0)
        generated: List[int] = []
        for token, _ in generate_step(
            prompt=input_ids,
            model=self.model,
            input_embeddings=input_embeddings,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            if int(token) in self._eos_token_ids:
                break
            generated.append(int(token))
        return generated


def load_tokenizer(model_id_or_path: str, revision: Optional[str]) -> MistralTokenizer:
    _require_mistral_common()
    model_path = get_model_path(
        model_id_or_path,
        revision=revision,
        allow_patterns=["tekken*.json", "tokenizer*.json", "config.json", "params.json"],
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

    def _quantize_predicate(name: str, module: Any) -> bool:
        if not (name.startswith("language_model") and isinstance(module, nn.Linear)):
            return False
        if group_size > 0:
            input_dim = int(module.weight.shape[-1])
            if input_dim % group_size != 0:
                return False
        return True

    nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        class_predicate=_quantize_predicate,
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
    except Exception:
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

    if audio_embeds.shape[0] < num_slots:
        pad = num_slots - audio_embeds.shape[0]
        audio_embeds = mx.concatenate(
            [audio_embeds, mx.zeros((pad, audio_embeds.shape[-1]), dtype=audio_embeds.dtype)],
            axis=0,
        )
    elif audio_embeds.shape[0] > num_slots:
        audio_embeds = audio_embeds[:num_slots]

    slot_idx = mx.cumsum(audio_mask.astype(mx.int32), axis=0) - 1
    slot_idx = mx.maximum(slot_idx, 0)
    slot_idx = mx.minimum(slot_idx, num_slots - 1)
    expanded_audio = audio_embeds[slot_idx]
    return mx.where(audio_mask[:, None], expanded_audio, text_embeds)


def _merge_stream_text(existing: str, current: str) -> tuple[str, str]:
    if not current:
        return existing, ""

    max_overlap = min(len(existing), len(current))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if existing.endswith(current[:size]):
            overlap = size
            break

    merged = existing + current[overlap:]
    return merged, merged[len(existing) :]


def _should_emit_realtime_text(
    raw_text: str,
    recent_tokens: List[int],
    control_token_ids: set[int],
) -> bool:
    stripped = raw_text.strip()
    if not stripped:
        return False
    if len(stripped) < 16:
        return False
    return not _should_fallback_realtime(recent_tokens, control_token_ids)


def _should_fallback_realtime(recent_tokens: List[int], control_token_ids: set[int]) -> bool:
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
    if control_ratio >= 0.8 and unique_lexical <= 2:
        return True
    if lexical_count >= 8 and unique_lexical <= 2:
        return True
    if longest_lexical_run >= 4:
        return True
    return False
