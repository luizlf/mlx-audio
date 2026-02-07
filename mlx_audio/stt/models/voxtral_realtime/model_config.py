from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import AudioConfig, TextConfig, VoxtralConfig


@dataclass
class ModelConfig:
    runtime: VoxtralConfig
    model_path: Optional[str] = None

    @property
    def model_type(self) -> str:
        model_type = str(self.runtime.raw.get("model_type", "voxtral_realtime"))
        if model_type == "voxtral":
            return "voxtral_realtime"
        return model_type

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
        model_path = cfg.get("model_path")
        runtime = VoxtralConfig.from_dict(cfg)
        return cls(runtime=runtime, model_path=model_path)
