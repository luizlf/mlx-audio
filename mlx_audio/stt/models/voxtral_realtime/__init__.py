from typing import Any

from .config import AudioConfig, TextConfig, VoxtralConfig
from .model_config import ModelConfig
from .model import VoxtralModel

_MISTRAL_COMMON_ERROR: ModuleNotFoundError | None = None


def _raise_missing_mistral_common(*_args: Any, **_kwargs: Any):
    message = (
        "Voxtral Realtime requires mistral-common[audio]. "
        "Install with: pip install 'mlx-audio[stt]'"
    )
    if _MISTRAL_COMMON_ERROR is not None:
        raise ModuleNotFoundError(message) from _MISTRAL_COMMON_ERROR
    raise ModuleNotFoundError(message)


try:
    from .api import STTOutput, VoxtralRealtime, load_tokenizer
except ModuleNotFoundError as exc:
    _MISTRAL_COMMON_ERROR = exc
    STTOutput = Any
    VoxtralRealtime = Any
    load_tokenizer = _raise_missing_mistral_common

try:
    from .voxtral_realtime import Model, StreamingResult
except ModuleNotFoundError as exc:
    _MISTRAL_COMMON_ERROR = _MISTRAL_COMMON_ERROR or exc
    class _MissingDependencyModel:
        def __init__(self, *_args: Any, **_kwargs: Any):
            _raise_missing_mistral_common()

    Model = _MissingDependencyModel
    StreamingResult = Any

DETECTION_HINTS = {
    "config_keys": {
        "audio_length_per_tok",
        "multimodal",
        "text_config",
        "audio_config",
        "whisper_model_args",
    },
    "path_patterns": {
        "voxtral",
        "voxtral_realtime",
        "voxtralrealtime",
    },
    "aliases": {
        "voxtral": "voxtral_realtime",
    },
}

__all__ = [
    "STTOutput",
    "VoxtralRealtime",
    "load_tokenizer",
    "VoxtralConfig",
    "AudioConfig",
    "TextConfig",
    "VoxtralModel",
    "ModelConfig",
    "Model",
    "StreamingResult",
]
