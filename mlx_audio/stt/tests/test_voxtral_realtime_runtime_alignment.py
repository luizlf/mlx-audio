from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import mlx.core as mx
import numpy as np
import pytest

if "mistral_common.audio" not in sys.modules:
    mistral_pkg = types.ModuleType("mistral_common")
    mistral_pkg.__path__ = []
    mistral_pkg.__spec__ = ModuleSpec("mistral_common", loader=None, is_package=True)
    audio_mod = types.ModuleType("mistral_common.audio")
    audio_mod.__spec__ = ModuleSpec("mistral_common.audio", loader=None)

    def _mel_filter_bank(
        *,
        num_frequency_bins: int,
        num_mel_bins: int,
        min_frequency: float,
        max_frequency: float,
        sampling_rate: int,
    ) -> np.ndarray:
        del min_frequency, max_frequency, sampling_rate
        return np.zeros((num_frequency_bins, num_mel_bins), dtype=np.float32)

    audio_mod.mel_filter_bank = _mel_filter_bank
    sys.modules["mistral_common"] = mistral_pkg
    sys.modules["mistral_common.audio"] = audio_mod

from mlx_audio.stt.models.voxtral_realtime import api as voxtral_api


def test_voxtral_realtime_load_uses_shared_stt_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLoadedModel(voxtral_api.VoxtralModel):
        def __init__(self):
            pass

        def _get_runtime(self):
            return self._runtime

    runtime = object.__new__(voxtral_api.VoxtralRealtime)
    model = _FakeLoadedModel()
    model._runtime = runtime

    load_calls = {}

    def _fake_shared_load(model_path: str, **kwargs):
        load_calls["model_path"] = model_path
        load_calls["kwargs"] = kwargs
        return model

    monkeypatch.setattr("mlx_audio.stt.utils.load", _fake_shared_load)
    monkeypatch.setattr(
        voxtral_api,
        "_maybe_cast_model",
        lambda *_args, **_kwargs: pytest.fail("default load should not cast runtime model"),
    )
    monkeypatch.setattr(
        voxtral_api,
        "_maybe_quantize_model",
        lambda *_args, **_kwargs: pytest.fail("default load should not quantize runtime model"),
    )

    out = voxtral_api.VoxtralRealtime.load(
        "mlx-community/voxtral-realtime",
        revision="main",
    )

    assert out is runtime
    assert load_calls == {
        "model_path": "mlx-community/voxtral-realtime",
        "kwargs": {"revision": "main"},
    }


def test_voxtral_realtime_load_refreshes_runtime_for_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLoadedModel(voxtral_api.VoxtralModel):
        def __init__(self):
            pass

        def _get_runtime(self):
            self.runtime_calls += 1
            if self._runtime is None:
                self._runtime = self._replacement_runtime
            return self._runtime

    initial_runtime = object.__new__(voxtral_api.VoxtralRealtime)
    refreshed_runtime = object.__new__(voxtral_api.VoxtralRealtime)
    model = _FakeLoadedModel()
    model._runtime = initial_runtime
    model._replacement_runtime = refreshed_runtime
    model.runtime_calls = 0

    cast_calls = []
    quant_calls = []

    monkeypatch.setattr("mlx_audio.stt.utils.load", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        voxtral_api,
        "_maybe_cast_model",
        lambda _model, dtype: cast_calls.append((_model, dtype)),
    )
    monkeypatch.setattr(
        voxtral_api,
        "_maybe_quantize_model",
        lambda _model, bits, group_size: quant_calls.append((_model, bits, group_size)),
    )

    out = voxtral_api.VoxtralRealtime.load(
        "mlx-community/voxtral-realtime",
        dtype="bf16",
        quantize_bits=4,
        quantize_group_size=64,
    )

    assert out is refreshed_runtime
    assert model.runtime_calls == 1
    assert cast_calls == [(model, "bf16")]
    assert quant_calls == [(model, 4, 64)]


def test_stream_transcribe_realtime_keeps_realtime_path_when_not_degenerate(
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
    monkeypatch.setattr(voxtral_api, "_get_streaming_mode", lambda _name: _name)

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
            "decode": lambda *_args, **_kwargs: "this is a healthy realtime transcript",
            "instruct_tokenizer": type(
                "_InstructTokenizer",
                (),
                {"audio_encoder": type("_AudioEncoder", (), {"audio_config": audio_cfg})()},
            )(),
        },
    )()
    runtime.model = object()
    runtime._stream_control_token_ids = {32, 33, 34}
    runtime._decode_policy = None

    prepare_calls = {"count": 0}
    generate_calls = {"count": 0}
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
        generate_calls["count"] += 1
        return [1455]

    def _transcribe(audio, language="en", max_tokens=256):
        del audio, language, max_tokens
        transcribe_calls["count"] += 1
        return voxtral_api.STTOutput(text="fallback transcript", prompt_tokens=0, generation_tokens=0)

    runtime._prepare_inputs = _prepare_inputs
    runtime._prepare_audio_conv_features = _prepare_audio_conv_features
    runtime._generate_audio_conditioned = _generate_audio_conditioned
    runtime.transcribe = _transcribe

    chunks = [np.ones(320, dtype=np.float32) for _ in range(4)]
    out = list(
        runtime.stream_transcribe(
            chunks,
            strategy="realtime",
            realtime_auto_fallback=True,
            realtime_fallback_refresh_reads=1,
        )
    )

    assert out == ["this is a healthy realtime transcript"]
    assert prepare_calls["count"] > 0
    assert generate_calls["count"] > 0
    assert transcribe_calls["count"] == 0


def test_stream_transcribe_realtime_falls_back_only_after_degenerate_detection(
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
    monkeypatch.setattr(voxtral_api, "_get_streaming_mode", lambda _name: _name)

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
            "decode": lambda *_args, **_kwargs: "realtime text",
            "instruct_tokenizer": type(
                "_InstructTokenizer",
                (),
                {"audio_encoder": type("_AudioEncoder", (), {"audio_config": audio_cfg})()},
            )(),
        },
    )()
    runtime.model = object()
    runtime._stream_control_token_ids = {32, 33, 34}
    runtime._decode_policy = None

    events = []
    transcribe_calls = {"count": 0}

    def _prepare_inputs(audio, language, streaming_mode=None):
        del language, streaming_mode
        return mx.array([7], dtype=mx.int32), [audio.astype(np.float32)]

    def _prepare_audio_conv_features(audio_arrays, **kwargs):
        del audio_arrays, kwargs
        return mx.zeros((1, 1, 1), dtype=mx.float32)

    def _generate_audio_conditioned(*args, **kwargs):
        del args, kwargs
        events.append("generate")
        return [32]

    def _transcribe(audio, language="en", max_tokens=256):
        del audio, language, max_tokens
        transcribe_calls["count"] += 1
        events.append("transcribe")
        return voxtral_api.STTOutput(text="fallback transcript", prompt_tokens=0, generation_tokens=0)

    runtime._prepare_inputs = _prepare_inputs
    runtime._prepare_audio_conv_features = _prepare_audio_conv_features
    runtime._generate_audio_conditioned = _generate_audio_conditioned
    runtime.transcribe = _transcribe

    chunks = [np.ones(320, dtype=np.float32) for _ in range(30)]
    out = list(
        runtime.stream_transcribe(
            chunks,
            strategy="realtime",
            realtime_auto_fallback=True,
            realtime_fallback_refresh_reads=1,
        )
    )

    assert transcribe_calls["count"] > 0
    assert "generate" in events
    assert "transcribe" in events
    assert events.index("transcribe") > events.index("generate")
    assert any("fallback transcript" in piece for piece in out)
