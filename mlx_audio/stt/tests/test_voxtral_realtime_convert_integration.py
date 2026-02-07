from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx_lm.utils as mlx_lm_utils
import pytest

import mlx_audio.convert as convert_module
from mlx_audio.convert import Domain
from mlx_audio.stt.models.voxtral_realtime.voxtral_realtime import (
    align_weights,
    remap_weights,
)


@pytest.fixture(autouse=True)
def _clear_convert_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    convert_module._model_types_cache.clear()
    convert_module._detection_hints_cache.clear()
    monkeypatch.setattr(
        convert_module,
        "get_model_types",
        lambda domain: {"voxtral", "voxtral_realtime"} if domain is Domain.STT else set(),
    )
    yield
    convert_module._model_types_cache.clear()
    convert_module._detection_hints_cache.clear()


def test_voxtral_detects_stt_domain_and_keeps_voxtral_type(tmp_path: Path) -> None:
    model_path = tmp_path / "hf-cache"
    model_path.mkdir()
    config = {
        "model_type": "voxtral",
        "multimodal": {"whisper_model_args": {}},
    }

    domain = convert_module.detect_model_domain(config, model_path)
    model_type = convert_module.get_model_type(config, model_path, domain)

    assert domain is Domain.STT
    assert model_type == "voxtral"


def test_official_realtime_params_without_model_type_resolve_to_voxtral_realtime(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models--mistralai--voxtral-mini-4b-realtime-2602"
    model_path.mkdir()
    config = {
        "dim": 3072,
        "n_layers": 26,
        "n_heads": 32,
        "n_kv_heads": 8,
        "hidden_dim": 9216,
        "norm_eps": 1e-5,
        "vocab_size": 131072,
        "rope_theta": 1000000.0,
        "model_max_length": 131072,
        "sliding_window": 8192,
        "use_biases": False,
        "causal": True,
        "ada_rms_norm_t_cond": True,
        "ada_rms_norm_t_cond_dim": 32,
        "tied_embeddings": False,
        "model_parallel": {"tp_size": 1},
        "multimodal": {"whisper_model_args": {}},
    }

    domain = convert_module.detect_model_domain(config, model_path)
    model_type = convert_module.get_model_type(config, model_path, domain)

    assert domain is Domain.STT
    assert model_type == "voxtral_realtime"


def test_official_voxtral_params_without_model_type_stay_voxtral(tmp_path: Path) -> None:
    model_path = tmp_path / "models--mistralai--voxtral-mini-3b-2507"
    model_path.mkdir()
    config = {
        "dim": 3072,
        "n_layers": 30,
        "n_heads": 32,
        "n_kv_heads": 8,
        "hidden_dim": 8192,
        "norm_eps": 1e-5,
        "vocab_size": 131072,
        "rope_theta": 100000000.0,
        "max_position_embeddings": 131072,
        "multimodal": {"whisper_model_args": {}},
    }

    domain = convert_module.detect_model_domain(config, model_path)
    model_type = convert_module.get_model_type(config, model_path, domain)

    assert domain is Domain.STT
    assert model_type == "voxtral"


def test_convert_uses_shared_pipeline_for_explicit_voxtral_realtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict = {}
    model_path = tmp_path / "models--mistralai--voxtral-realtime-mini"
    model_path.mkdir(parents=True)

    class DummyModelConfig:
        def __init__(self) -> None:
            self.model_path: Path | None = None

        @classmethod
        def from_dict(cls, cfg: dict) -> "DummyModelConfig":
            calls["input_config"] = dict(cfg)
            return cls()

    class DummyModel:
        def __init__(self, model_config: DummyModelConfig) -> None:
            self.model_config = model_config
            self.sanitize_calls = 0
            self.load_weights_calls = 0
            self.loaded_weights: dict = {}

        def sanitize(self, weights: dict) -> dict:
            self.sanitize_calls += 1
            calls["sanitize_input_keys"] = sorted(weights.keys())
            return {"sanitized.weight": weights["raw.weight"]}

        def load_weights(self, items) -> None:
            self.load_weights_calls += 1
            self.loaded_weights = dict(items)

        def parameters(self):
            return {"sanitized.weight": mx.array([1.0], dtype=mx.float32)}

    dummy_module = SimpleNamespace(ModelConfig=DummyModelConfig, Model=DummyModel)
    saved: dict = {}

    monkeypatch.setattr(convert_module, "get_model_path", lambda *_args, **_kwargs: model_path)
    monkeypatch.setattr(
        convert_module,
        "load_config",
        lambda _path: {
            "model_type": "voxtral_realtime",
            "multimodal": {"whisper_model_args": {}},
        },
    )
    monkeypatch.setattr(
        convert_module,
        "load_weights",
        lambda _path: {"raw.weight": mx.array([1.0], dtype=mx.float32)},
    )
    monkeypatch.setattr(convert_module, "copy_model_files", lambda *_args, **_kwargs: None)

    def _fake_get_model_class(model_type: str, domain: Domain):
        calls["model_type"] = model_type
        calls["domain"] = domain
        return dummy_module

    monkeypatch.setattr(convert_module, "get_model_class", _fake_get_model_class)
    monkeypatch.setattr(
        mlx_lm_utils,
        "save_model",
        lambda path, model, donate_model=True: saved.update(
            {"path": Path(path), "model": model, "donate_model": donate_model}
        ),
    )
    monkeypatch.setattr(
        mlx_lm_utils,
        "save_config",
        lambda cfg, config_path: saved.update(
            {"config": dict(cfg), "config_path": Path(config_path)}
        ),
    )

    out_path = tmp_path / "out"
    convert_module.convert(
        hf_path="mistralai/Voxtral-Realtime-Mini-3B-2507",
        mlx_path=str(out_path),
    )

    model = saved["model"]
    assert calls["domain"] is Domain.STT
    assert calls["model_type"] == "voxtral_realtime"
    assert isinstance(model, DummyModel)
    assert model.sanitize_calls == 1
    assert model.load_weights_calls == 1
    assert sorted(model.loaded_weights.keys()) == ["sanitized.weight"]
    assert model.model_config.model_path == model_path
    assert saved["config"]["model_type"] == "voxtral_realtime"
    assert saved["config_path"] == out_path / "config.json"


def test_voxtral_weight_remap_helpers_live_in_runtime_module() -> None:
    assert callable(remap_weights)
    assert callable(align_weights)


def test_voxtral_weight_remap_supports_mm_whisper_prefix() -> None:
    weights = {
        "mm_whisper_embeddings.whisper_encoder.conv_layers.0.conv.weight": mx.array(
            [[[1.0]]], dtype=mx.float32
        ),
        "mm_whisper_embeddings.tok_embeddings.weight": mx.array([[1.0]], dtype=mx.float32),
    }

    remapped = remap_weights(weights)

    assert "audio_encoder.conv1.conv.weight" in remapped
    assert "language_model.model.embed_tokens.weight" in remapped
