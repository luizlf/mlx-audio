from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mistral_common")

from mlx_audio.stt.models.voxtral.config import ModelConfig
from mlx_audio.stt.voxtral.api import _replace_audio_placeholders, _should_fallback_realtime
from mlx_audio.stt.voxtral.config import VoxtralConfig


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
    assert model_cfg.model_type == "voxtral"
    assert model_cfg.audio_config.num_mel_bins == 80
    assert model_cfg.text_config.hidden_size == 3072
