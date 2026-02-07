from __future__ import annotations

from copy import deepcopy
import json

from mlx_audio.stt.models.voxtral_realtime import config as voxtral_config_module
from mlx_audio.stt.models.voxtral_realtime.config import ModelConfig, VoxtralRealtimeConfig
from mlx_audio.utils import load_config


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
    vox_cfg = VoxtralRealtimeConfig.from_dict(_legacy_split_config())

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
    assert vox_cfg.audio.max_source_positions == 1500


def test_voxtral_config_from_dict_does_not_mutate_input() -> None:
    cfg = _legacy_split_config()
    cfg_before = deepcopy(cfg)

    VoxtralRealtimeConfig.from_dict(cfg)

    assert cfg == cfg_before


def test_shared_load_config_falls_back_to_params_json(tmp_path) -> None:
    expected = _legacy_split_config()
    (tmp_path / "params.json").write_text(json.dumps(expected), encoding="utf-8")

    loaded = load_config(tmp_path)

    assert loaded == expected


def test_voxtral_config_from_pretrained_loads_params_json_when_config_missing(tmp_path) -> None:
    expected = _legacy_split_config()
    (tmp_path / "params.json").write_text(json.dumps(expected), encoding="utf-8")

    vox_cfg, model_path = VoxtralRealtimeConfig.from_pretrained(str(tmp_path))

    assert model_path == tmp_path
    assert vox_cfg.text.hidden_size == 3072
    assert vox_cfg.audio.d_model == 1280


def test_voxtral_config_from_pretrained_uses_shared_loaders(monkeypatch, tmp_path) -> None:
    events = []
    expected = _legacy_split_config()

    def fake_get_model_path(model_id_or_path: str, revision=None, **kwargs):
        events.append(("get_model_path", model_id_or_path, revision, kwargs))
        return tmp_path

    def fake_load_config(model_path, **kwargs):
        events.append(("load_config", model_path, kwargs))
        return expected

    monkeypatch.setattr(voxtral_config_module, "get_model_path", fake_get_model_path)
    monkeypatch.setattr(voxtral_config_module, "load_config", fake_load_config)

    vox_cfg, model_path = VoxtralRealtimeConfig.from_pretrained("mlx-community/voxtral-rt", revision="main")

    assert events == [
        ("get_model_path", "mlx-community/voxtral-rt", "main", {}),
        ("load_config", tmp_path, {}),
    ]
    assert model_path == tmp_path
    assert vox_cfg.audio.encoder_layers == 32


def test_voxtral_config_uses_max_seq_len_fallback() -> None:
    cfg = {
        "max_seq_len": 4096,
        "dim": 3072,
        "n_layers": 2,
        "n_heads": 8,
        "n_kv_heads": 8,
        "hidden_dim": 8192,
        "norm_eps": 1e-5,
        "vocab_size": 32000,
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
                    "vocab_size": 123,
                    "max_source_positions": 1500,
                    "causal": True,
                },
                "downsample_args": {"downsample_factor": 2},
            }
        },
    }

    vox_cfg = VoxtralRealtimeConfig.from_dict(cfg)

    assert vox_cfg.text.max_position_embeddings == 4096
