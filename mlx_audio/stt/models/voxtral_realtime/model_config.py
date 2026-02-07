from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import AudioConfig, TextConfig, VoxtralConfig


@dataclass
class ModelConfig:
    runtime: VoxtralConfig
    model_path: Optional[str] = None
    model_type: str = "voxtral_realtime"

    def __post_init__(self) -> None:
        if isinstance(self.runtime, dict):
            self.runtime = VoxtralConfig.from_dict(self.runtime)
        if not isinstance(self.runtime, VoxtralConfig):
            raise TypeError("runtime must be a VoxtralConfig or a dict-like config payload")
        self.model_type = str(self.model_type or "voxtral_realtime")

    @property
    def raw(self) -> Dict[str, Any]:
        return self.runtime.raw

    @property
    def text_config(self) -> TextConfig:
        return self.runtime.text

    @property
    def audio_config(self) -> AudioConfig:
        return self.runtime.audio

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        cfg = params.copy()
        runtime_cfg = cfg.pop("runtime", None)
        model_path = cfg.pop("model_path", None)
        if runtime_cfg is None:
            runtime_cfg = cfg

        model_type_value = cfg.get("model_type")
        if model_type_value is None and isinstance(runtime_cfg, VoxtralConfig):
            model_type_value = runtime_cfg.raw.get("model_type")
        if model_type_value is None and isinstance(runtime_cfg, dict):
            model_type_value = runtime_cfg.get("model_type")
        model_type = str(model_type_value or "voxtral_realtime")

        if isinstance(runtime_cfg, dict):
            runtime = VoxtralConfig.from_dict(runtime_cfg.copy())
        elif isinstance(runtime_cfg, VoxtralConfig):
            runtime = runtime_cfg
        else:
            raise TypeError("runtime must be a VoxtralConfig or a dict-like config payload")

        return cls(runtime=runtime, model_path=model_path, model_type=model_type)
