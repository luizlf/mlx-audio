from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

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
from mlx_audio import utils as mlx_utils


@pytest.fixture(autouse=True)
def _reset_voxtral_runtime_cache() -> None:
    voxtral_api.VoxtralRealtime.clear_cache()
    yield
    voxtral_api.VoxtralRealtime.clear_cache()


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


def test_loader_forces_voxtral_realtime_on_streaming_signature() -> None:
    cfg = {
        "model_type": "voxtral",
        "multimodal": {
            "whisper_model_args": {
                "encoder_args": {
                    "audio_encoding_args": {"transcription_format": "streaming"}
                }
            }
        },
    }
    assert mlx_utils._should_force_voxtral_realtime_model_type(cfg, "stt")


def test_loader_keeps_voxtral_for_non_streaming_signature() -> None:
    cfg = {
        "model_type": "voxtral",
        "multimodal": {
            "whisper_model_args": {
                "encoder_args": {
                    "audio_encoding_args": {"transcription_format": "chunked"}
                }
            }
        },
    }
    assert not mlx_utils._should_force_voxtral_realtime_model_type(cfg, "stt")


def test_voxtral_realtime_load_recovers_from_voxtral_model_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WrongType:
        pass

    class _RecoveredModel(voxtral_api.VoxtralModel):
        def __init__(self):
            pass

        def _get_runtime(self):
            return self._runtime

    runtime = object.__new__(voxtral_api.VoxtralRealtime)
    recovered = _RecoveredModel()
    recovered._runtime = runtime

    monkeypatch.setattr("mlx_audio.stt.utils.load", lambda *_args, **_kwargs: _WrongType())
    monkeypatch.setattr(
        voxtral_api,
        "get_model_path",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        voxtral_api,
        "load_config",
        lambda _path: {"model_type": "voxtral"},
    )
    force_calls = {}
    monkeypatch.setattr(
        voxtral_api,
        "_load_with_forced_voxtral_realtime_tag",
        lambda _path, revision=None: (
            force_calls.update({"path": _path, "revision": revision}) or recovered
        ),
    )

    out = voxtral_api.VoxtralRealtime.load("../voxtral-mlx/model_mlx_audio")

    assert out is runtime
    assert force_calls["revision"] is None


def test_voxtral_realtime_load_does_not_reuse_process_runtime_cache(
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

    calls = {"count": 0}

    def _fake_shared_load(*_args, **_kwargs):
        calls["count"] += 1
        return model

    monkeypatch.setattr("mlx_audio.stt.utils.load", _fake_shared_load)

    out1 = voxtral_api.VoxtralRealtime.load("../voxtral-mlx/model_mlx_audio")
    out2 = voxtral_api.VoxtralRealtime.load("../voxtral-mlx/model_mlx_audio")

    assert out1 is out2
    assert calls["count"] == 2


def test_voxtral_realtime_load_cache_normalizes_local_path_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FakeLoadedModel(voxtral_api.VoxtralModel):
        def __init__(self):
            pass

        def _get_runtime(self):
            return self._runtime

    runtime = object.__new__(voxtral_api.VoxtralRealtime)
    model = _FakeLoadedModel()
    model._runtime = runtime

    calls = {"count": 0}

    def _fake_shared_load(*_args, **_kwargs):
        calls["count"] += 1
        return model

    monkeypatch.setattr("mlx_audio.stt.utils.load", _fake_shared_load)

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    out1 = voxtral_api.VoxtralRealtime.load(str(model_dir))
    out2 = voxtral_api.VoxtralRealtime.load(str(model_dir / "."))

    assert out1 is out2
    assert calls["count"] == 2


def test_voxtral_realtime_load_recovers_when_base_loader_raises_for_voxtral_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecoveredModel(voxtral_api.VoxtralModel):
        def __init__(self):
            pass

        def _get_runtime(self):
            return self._runtime

    runtime = object.__new__(voxtral_api.VoxtralRealtime)
    recovered = _RecoveredModel()
    recovered._runtime = runtime

    monkeypatch.setattr(
        "mlx_audio.stt.utils.load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad voxtral tag")),
    )
    monkeypatch.setattr(voxtral_api, "get_model_path", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(voxtral_api, "load_config", lambda _path: {"model_type": "voxtral"})
    monkeypatch.setattr(
        voxtral_api,
        "_load_with_forced_voxtral_realtime_tag",
        lambda _path, revision=None: recovered,
    )

    out = voxtral_api.VoxtralRealtime.load("../voxtral-mlx/model_mlx_audio")

    assert out is runtime


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
            "num_audio_tokens": staticmethod(lambda _n: 1),
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
    runtime.config = type(
        "_RuntimeCfg",
        (),
        {"audio": type("_RuntimeAudioCfg", (), {"window_size": 400, "hop_length": 160, "downsample_factor": 1})()},
    )()
    runtime.model = type(
        "_Model",
        (),
        {
            "audio_encoder": type(
                "_AudioEncoderRuntime",
                (),
                {
                    "make_padding_cache": staticmethod(lambda: object()),
                    "make_cache": staticmethod(lambda: object()),
                },
            )()
        },
    )()
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
        )
    )

    assert out == ["this is a healthy realtime transcript"]
    assert prepare_calls["count"] > 0
    assert generate_calls["count"] > 0
    assert transcribe_calls["count"] == 0


def test_stream_transcribe_realtime_does_not_use_offline_fallback_on_degenerate_tokens(
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
            "num_audio_tokens": staticmethod(lambda _n: 1),
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
    runtime.config = type(
        "_RuntimeCfg",
        (),
        {"audio": type("_RuntimeAudioCfg", (), {"window_size": 400, "hop_length": 160, "downsample_factor": 1})()},
    )()
    runtime.model = type(
        "_Model",
        (),
        {
            "audio_encoder": type(
                "_AudioEncoderRuntime",
                (),
                {
                    "make_padding_cache": staticmethod(lambda: object()),
                    "make_cache": staticmethod(lambda: object()),
                },
            )()
        },
    )()
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

    def _transcribe(audio, language="en", max_tokens=0):
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
        )
    )

    assert transcribe_calls["count"] == 0
    assert "generate" in events
    assert "transcribe" not in events
    assert any("realtime text" in piece for piece in out)


def test_stream_transcribe_realtime_continues_after_empty_first_step(
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
            "num_audio_tokens": staticmethod(lambda _n: 1),
        },
    )()
    runtime.tokenizer = type(
        "_Tokenizer",
        (),
        {
            "decode": staticmethod(
                lambda token_ids, **_kwargs: "continued output" if token_ids else ""
            ),
            "instruct_tokenizer": type(
                "_InstructTokenizer",
                (),
                {"audio_encoder": type("_AudioEncoder", (), {"audio_config": audio_cfg})()},
            )(),
        },
    )()
    runtime.config = type(
        "_RuntimeCfg",
        (),
        {
            "audio": type(
                "_RuntimeAudioCfg",
                (),
                {"window_size": 400, "hop_length": 160, "downsample_factor": 1},
            )()
        },
    )()
    runtime.model = type(
        "_Model",
        (),
        {
            "audio_encoder": type(
                "_AudioEncoderRuntime",
                (),
                {
                    "make_padding_cache": staticmethod(lambda: object()),
                    "make_cache": staticmethod(lambda: object()),
                },
            )()
        },
    )()
    runtime._stream_control_token_ids = {32, 33, 34}
    runtime._decode_policy = None

    generate_calls = {"count": 0}

    def _prepare_inputs(audio, language, streaming_mode=None):
        del audio, language, streaming_mode
        return mx.array([7], dtype=mx.int32), [np.ones(320, dtype=np.float32)]

    def _prepare_audio_conv_features(audio_arrays, **kwargs):
        del audio_arrays, kwargs
        return mx.zeros((1, 1, 1), dtype=mx.float32)

    def _generate_audio_conditioned(*args, **kwargs):
        del args, kwargs
        generate_calls["count"] += 1
        if generate_calls["count"] == 1:
            return []
        return [1455]

    runtime._prepare_inputs = _prepare_inputs
    runtime._prepare_audio_conv_features = _prepare_audio_conv_features
    runtime._generate_audio_conditioned = _generate_audio_conditioned

    chunks = [np.ones(320, dtype=np.float32), np.ones(320, dtype=np.float32)]
    out = list(runtime.stream_transcribe(chunks))

    assert generate_calls["count"] >= 2
    assert out == ["continued output"]


def test_stream_transcribe_rejects_stable_strategy() -> None:
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
            "num_audio_tokens": staticmethod(lambda _n: 1),
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

    with pytest.raises(ValueError, match="supports only 'realtime'"):
        list(runtime.stream_transcribe([np.ones(320, dtype=np.float32)], strategy="stable"))


def test_wrapper_generate_stream_rejects_stable_strategy() -> None:
    from mlx_audio.stt.models.voxtral_realtime import voxtral_realtime as voxtral_wrapper

    model = voxtral_wrapper.Model.__new__(voxtral_wrapper.Model)
    model.config = type(
        "_Cfg",
        (),
        {
            "audio_config": type("_AudioCfg", (), {"sampling_rate": 16000})(),
        },
    )()
    model._load_audio_input = lambda _audio: np.ones(3200, dtype=np.float32)

    class _Runtime:
        pass

    model._get_runtime = lambda: _Runtime()

    with pytest.raises(ValueError, match="stream_strategy='realtime'"):
        list(
            model.generate(
                np.ones(3200, dtype=np.float32),
                stream=True,
                stream_strategy="stable",
            )
        )


def test_wrapper_generate_stream_passes_realtime_strategy_to_runtime() -> None:
    from mlx_audio.stt.models.voxtral_realtime import voxtral_realtime as voxtral_wrapper

    model = voxtral_wrapper.Model.__new__(voxtral_wrapper.Model)
    model.config = type(
        "_Cfg",
        (),
        {
            "audio_config": type("_AudioCfg", (), {"sampling_rate": 16000})(),
        },
    )()
    model._load_audio_input = lambda _audio: np.ones(3200, dtype=np.float32)

    captured = {}

    class _Runtime:
        tokenizer = type(
            "_Tokenizer",
            (),
            {
                "instruct_tokenizer": type(
                    "_InstructTokenizer",
                    (),
                    {
                        "audio_encoder": type(
                            "_AudioEncoder",
                            (),
                            {"audio_config": type("_AudioCfg", (), {"sampling_rate": 16000, "frame_rate": 50})()},
                        )()
                    },
                )()
            },
        )()

        def stream_transcribe_tokens(self, _audio_iter, **kwargs):
            captured.update(kwargs)
            yield voxtral_api.RealtimeTokenEvent(token_id=1455, text="realtime output")

    model._get_runtime = lambda: _Runtime()

    out = list(
        model.generate(
            np.ones(3200, dtype=np.float32),
            stream=True,
            stream_strategy="realtime",
        )
    )

    assert captured["strategy"] == "realtime"
    assert out[0].text == "realtime output"
    assert out[0].token_id == 1455
    assert out[0].is_final is False
    assert out[-1].is_final is True


def test_stream_transcribe_tokens_emits_token_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "num_audio_tokens": staticmethod(lambda _n: 1),
        },
    )()
    runtime.tokenizer = type(
        "_Tokenizer",
        (),
        {
            "decode": staticmethod(
                lambda token_ids, **_kwargs: "hello" if token_ids else ""
            ),
            "instruct_tokenizer": type(
                "_InstructTokenizer",
                (),
                {
                    "audio_encoder": type(
                        "_AudioEncoder", (), {"audio_config": audio_cfg}
                    )()
                },
            )(),
        },
    )()
    runtime.config = type(
        "_RuntimeCfg",
        (),
        {
            "audio": type(
                "_RuntimeAudioCfg",
                (),
                {"window_size": 400, "hop_length": 160, "downsample_factor": 1},
            )()
        },
    )()
    runtime.model = type(
        "_Model",
        (),
        {
            "audio_encoder": type(
                "_AudioEncoderRuntime",
                (),
                {
                    "make_padding_cache": staticmethod(lambda: object()),
                    "make_cache": staticmethod(lambda: object()),
                },
            )()
        },
    )()
    runtime._decode_policy = None

    runtime._prepare_inputs = lambda audio, language, streaming_mode=None: (
        mx.array([7], dtype=mx.int32),
        [audio.astype(np.float32)],
    )
    runtime._prepare_audio_conv_features = (
        lambda _audio_arrays, **_kwargs: mx.zeros((1, 1, 1), dtype=mx.float32)
    )
    runtime._generate_audio_conditioned = lambda *_args, **_kwargs: [1455]

    events = list(runtime.stream_transcribe_tokens([np.ones(320, dtype=np.float32)]))

    assert len(events) == 1
    assert events[0].token_id == 1455
    assert events[0].text == "hello"
