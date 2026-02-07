from __future__ import annotations

from copy import deepcopy

from mlx_audio.stt.models.voxtral_realtime.config import VoxtralConfig
from mlx_audio.stt.models.voxtral_realtime.model_config import ModelConfig


def _legacy_split_config(model_type: str = "voxtral_realtime") -> dict:
    return {
        "model_type": model_type,
        "text_config": {
            "dim": 3072,
            "n_layers": 30,
            "n_heads": 32,
            "n_kv_heads": 8,
            "hidden_dim": 8192,
            "norm_eps": 1e-5,
            "vocab_size": 131072,
            "model_max_length": 65536,
            "tied_embeddings": False,
        },
        "audio_config": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "intermediate_size": 5120,
            "downsample_factor": 2,
            "causal": True,
            "audio_encoding_args": {
                "num_mel_bins": 128,
                "window_size": 400,
                "sampling_rate": 16000,
                "hop_length": 160,
            },
        },
    }


def test_model_config_from_dict_accepts_nested_runtime_split_config() -> None:
    wrapped = {"runtime": _legacy_split_config(), "model_path": "/tmp/voxtral-rt"}

    model_cfg = ModelConfig.from_dict(wrapped)

    assert model_cfg.model_type == "voxtral_realtime"
    assert model_cfg.model_path == "/tmp/voxtral-rt"
    assert model_cfg.text_config.hidden_size == 3072
    assert model_cfg.audio_config.d_model == 1280


def test_model_config_does_not_map_voxtral_to_voxtral_realtime() -> None:
    model_cfg = ModelConfig.from_dict({"runtime": _legacy_split_config(model_type="voxtral")})

    assert model_cfg.model_type == "voxtral"
    assert model_cfg.raw["model_type"] == "voxtral"


def test_voxtral_config_preserves_legacy_split_config_remapping() -> None:
    vox_cfg = VoxtralConfig.from_dict(_legacy_split_config())

    assert vox_cfg.text.hidden_size == 3072
    assert vox_cfg.text.num_hidden_layers == 30
    assert vox_cfg.text.num_attention_heads == 32
    assert vox_cfg.text.num_key_value_heads == 8
    assert vox_cfg.text.max_position_embeddings == 65536
    assert vox_cfg.text.tie_word_embeddings is False

    assert vox_cfg.audio.num_mel_bins == 128
    assert vox_cfg.audio.window_size == 400
    assert vox_cfg.audio.n_fft == 400
    assert vox_cfg.audio.sampling_rate == 16000
    assert vox_cfg.audio.hop_length == 160
    assert vox_cfg.audio.downsample_factor == 2
    assert vox_cfg.audio.d_model == 1280
    assert vox_cfg.audio.encoder_layers == 32
    assert vox_cfg.audio.encoder_ffn_dim == 5120
    assert vox_cfg.audio.encoder_attention_heads == 20
    assert vox_cfg.audio.encoder_head_dim == 64
    assert vox_cfg.audio.is_causal is True
    assert vox_cfg.audio.max_source_positions == 1500


def test_voxtral_config_from_dict_does_not_mutate_input() -> None:
    cfg = _legacy_split_config()
    cfg_before = deepcopy(cfg)

    VoxtralConfig.from_dict(cfg)

    assert cfg == cfg_before
