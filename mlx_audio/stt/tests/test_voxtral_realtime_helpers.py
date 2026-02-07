from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mistral_common")

from mlx_audio.stt.models.voxtral_realtime import api as voxtral_api
from mlx_audio.stt.models.voxtral_realtime.api import (
    _replace_audio_placeholders,
    _should_fallback_realtime,
)
from mlx_audio.stt.models.voxtral_realtime.config import AudioConfig, TextConfig, VoxtralConfig
from mlx_audio.stt.models.voxtral_realtime.converter import _build_output_config
from mlx_audio.stt.models.voxtral_realtime.model_config import ModelConfig


def test_replace_audio_placeholders_replaces_only_audio_tokens() -> None:
    input_ids = mx.array([1, 24, 7, 24], dtype=mx.int32)
    text_embeds = mx.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=mx.float32,
    )
    audio_embeds = mx.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
        ],
        dtype=mx.float32,
    )

    out = _replace_audio_placeholders(
        input_ids=input_ids,
        text_embeds=text_embeds,
        audio_embeds=audio_embeds,
        audio_token_id=24,
    )

    expected = np.array(
        [
            [1.0, 1.0],
            [10.0, 10.0],
            [3.0, 3.0],
            [20.0, 20.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(out), expected)


def test_replace_audio_placeholders_pads_missing_audio_embeddings() -> None:
    input_ids = mx.array([24, 5, 24, 24], dtype=mx.int32)
    text_embeds = mx.zeros((4, 2), dtype=mx.float32)
    audio_embeds = mx.array([[9.0, 9.0]], dtype=mx.float32)

    out = _replace_audio_placeholders(
        input_ids=input_ids,
        text_embeds=text_embeds,
        audio_embeds=audio_embeds,
        audio_token_id=24,
    )

    expected = np.array(
        [
            [9.0, 9.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(out), expected)


def test_replace_audio_placeholders_truncates_extra_audio_embeddings() -> None:
    input_ids = mx.array([8, 24, 24], dtype=mx.int32)
    text_embeds = mx.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=mx.float32,
    )
    audio_embeds = mx.array(
        [
            [7.0, 7.0],
            [8.0, 8.0],
            [9.0, 9.0],
        ],
        dtype=mx.float32,
    )

    out = _replace_audio_placeholders(
        input_ids=input_ids,
        text_embeds=text_embeds,
        audio_embeds=audio_embeds,
        audio_token_id=24,
    )

    expected = np.array(
        [
            [1.0, 1.0],
            [7.0, 7.0],
            [8.0, 8.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(out), expected)


def test_should_fallback_realtime_detects_control_loops() -> None:
    recent = [32] * 48
    assert _should_fallback_realtime(recent, {32, 33, 34})


def test_should_fallback_realtime_detects_repetitive_lexical_runs() -> None:
    recent = ([32] * 20) + ([1455] * 28)
    assert _should_fallback_realtime(recent, {32, 33, 34})


def test_should_fallback_realtime_keeps_varied_tokens() -> None:
    recent = [32, 1455, 72, 33, 11, 985, 34, 45] * 6
    assert not _should_fallback_realtime(recent, {32, 33, 34})


def test_stream_transcribe_realtime_stays_realtime_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyBuffer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._pending = []

        def write(self, chunk: np.ndarray) -> None:
            self._pending.append(chunk)

        def read(self):
            if self._pending:
                return self._pending.pop(0)
            return None

    monkeypatch.setattr(voxtral_api, "StreamingBuffer", _DummyBuffer)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _model: None)

    runtime = object.__new__(voxtral_api.VoxtralRealtime)
    audio_cfg = type(
        "_AudioCfg",
        (),
        {
            "sampling_rate": 16000,
            "frame_rate": 50,
            "transcription_delay_ms": 0.0,
            "streaming_look_ahead_ms": 0.0,
            "streaming_look_back_ms": 0.0,
        },
    )()
    runtime.tokenizer = type(
        "_Tokenizer",
        (),
        {
            "instruct_tokenizer": type(
                "_InstructTokenizer",
                (),
                {"audio_encoder": type("_AudioEncoder", (), {"audio_config": audio_cfg})()},
            )()
        },
    )()
    runtime.model = object()
    runtime._stream_control_token_ids = {32, 33, 34}
    runtime._decode_policy = None

    prepare_calls = {"count": 0}
    transcribe_calls = {"count": 0}

    def _prepare_inputs(audio, language, streaming_mode=None):
        del language, streaming_mode
        prepare_calls["count"] += 1
        return mx.array([7], dtype=mx.int32), [audio.astype(np.float32)]

    def _prepare_audio_conv_features(audio_arrays, **kwargs):
        del audio_arrays, kwargs
        return mx.zeros((1, 1, 1), dtype=mx.float32)

    def _generate_audio_conditioned(*args, **kwargs):
        del args, kwargs
        return []

    def _transcribe(audio, language="en", max_tokens=256):
        del audio, language, max_tokens
        transcribe_calls["count"] += 1
        return voxtral_api.STTOutput(text="final transcript", prompt_tokens=0, generation_tokens=0)

    runtime._prepare_inputs = _prepare_inputs
    runtime._prepare_audio_conv_features = _prepare_audio_conv_features
    runtime._generate_audio_conditioned = _generate_audio_conditioned
    runtime.transcribe = _transcribe

    chunks = [np.ones(320, dtype=np.float32) for _ in range(3)]
    list(
        runtime.stream_transcribe(
            chunks,
            strategy="realtime",
            offline_refresh_reads=1,
            realtime_auto_fallback=True,
            realtime_fallback_refresh_reads=1,
        )
    )

    assert prepare_calls["count"] > 0
    assert transcribe_calls["count"] == 1


def test_voxtral_config_from_multimodal_dict() -> None:
    cfg = {
        "hidden_size": 3072,
        "vocab_size": 32000,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "intermediate_size": 8192,
        "rms_norm_eps": 1e-5,
        "multimodal": {
            "whisper_model_args": {
                "encoder_args": {
                    "audio_encoding_args": {
                        "num_mel_bins": 80,
                        "window_size": 400,
                        "sampling_rate": 16000,
                        "hop_length": 160,
                    },
                    "dim": 256,
                    "n_layers": 4,
                    "hidden_dim": 1024,
                    "n_heads": 4,
                    "head_dim": 64,
                    "vocab_size": 123,
                    "max_source_positions": 1500,
                    "causal": True,
                    "sliding_window": 1000,
                },
                "downsample_args": {"downsample_factor": 2},
            }
        },
        "max_position_embeddings": 2048,
    }

    vox = VoxtralConfig.from_dict(cfg)
    assert vox.audio.num_mel_bins == 80
    assert vox.audio.is_causal is True
    assert vox.text.hidden_size == 3072


def test_voxtral_config_remaps_legacy_split_audio_config() -> None:
    cfg = {
        "model_type": "voxtral_realtime",
        "text_config": {
            "hidden_size": 3072,
            "num_hidden_layers": 30,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "rms_norm_eps": 1e-5,
            "vocab_size": 131072,
        },
        "audio_config": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "intermediate_size": 5120,
            "num_mel_bins": 128,
            "max_source_positions": 1500,
        },
    }

    vox = VoxtralConfig.from_dict(cfg)
    assert vox.audio.d_model == 1280
    assert vox.audio.encoder_layers == 32
    assert vox.audio.encoder_ffn_dim == 5120
    assert vox.audio.encoder_attention_heads == 20
    assert vox.audio.sampling_rate == 16000
    assert vox.audio.hop_length == 160
    assert vox.audio.window_size == 400


def test_converter_output_config_forces_voxtral_model_type() -> None:
    cfg = VoxtralConfig(
        text=TextConfig({"hidden_size": 1, "vocab_size": 1}),
        audio=AudioConfig({}),
        raw={"foo": "bar", "model_type": "mistral"},
    )

    out_cfg = _build_output_config(cfg)
    assert out_cfg["model_type"] == "voxtral_realtime"
    assert out_cfg["foo"] == "bar"
    assert cfg.raw["model_type"] == "mistral"


def test_model_config_wraps_runtime_config() -> None:
    cfg = {
        "text_config": {
            "model_type": "mistral",
            "hidden_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32000,
            "head_dim": 128,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        "audio_config": {
            "num_mel_bins": 80,
            "window_size": 400,
            "sampling_rate": 16000,
            "hop_length": 160,
            "downsample_factor": 2,
            "d_model": 256,
            "encoder_layers": 4,
            "encoder_ffn_dim": 1024,
            "encoder_attention_heads": 4,
            "max_source_positions": 1500,
        },
    }

    model_cfg = ModelConfig.from_dict(cfg)
    assert model_cfg.model_type == "voxtral_realtime"
    assert model_cfg.audio_config.num_mel_bins == 80
    assert model_cfg.text_config.hidden_size == 3072
