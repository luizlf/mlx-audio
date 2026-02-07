from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mlx_audio.utils import get_model_path, load_config


@dataclass
class TextConfig:
    data: Dict[str, Any]

    @property
    def hidden_size(self) -> int:
        return int(self.data.get("hidden_size", self.data.get("dim")))

    @property
    def vocab_size(self) -> int:
        return int(self.data["vocab_size"])

    @property
    def num_hidden_layers(self) -> int:
        return int(self.data.get("num_hidden_layers", self.data.get("n_layers")))

    @property
    def num_attention_heads(self) -> int:
        return int(self.data.get("num_attention_heads", self.data.get("n_heads")))

    @property
    def num_key_value_heads(self) -> int:
        return int(
            self.data.get(
                "num_key_value_heads", self.data.get("n_kv_heads", self.num_attention_heads)
            )
        )

    @property
    def intermediate_size(self) -> int:
        return int(self.data.get("intermediate_size", self.data.get("hidden_dim")))

    @property
    def rms_norm_eps(self) -> float:
        return float(self.data.get("rms_norm_eps", self.data.get("norm_eps", 1e-5)))

    @property
    def rope_theta(self) -> float:
        return float(self.data.get("rope_theta", 10000.0))

    @property
    def max_position_embeddings(self) -> int:
        value = self.data.get("max_position_embeddings")
        if value is None:
            value = self.data.get("model_max_length", self.data.get("max_seq_len", 131072))
        return int(value)

    @property
    def sliding_window(self) -> Optional[int]:
        value = self.data.get("sliding_window")
        return int(value) if value is not None else None

    @property
    def layer_types(self) -> Optional[list[str]]:
        return self.data.get("layer_types")

    @property
    def tie_word_embeddings(self) -> bool:
        return bool(self.data.get("tie_word_embeddings", self.data.get("tied_embeddings", True)))

    @property
    def ada_rms_norm_t_cond(self) -> bool:
        return bool(self.data.get("ada_rms_norm_t_cond", False))

    @property
    def ada_rms_norm_t_cond_dim(self) -> Optional[int]:
        value = self.data.get("ada_rms_norm_t_cond_dim")
        return int(value) if value is not None else None


@dataclass
class AudioConfig:
    data: Dict[str, Any]

    @property
    def num_mel_bins(self) -> int:
        return int(self.data["num_mel_bins"])

    @property
    def window_size(self) -> int:
        return int(self.data["window_size"])

    @property
    def n_fft(self) -> int:
        return int(self.data.get("n_fft", self.window_size))

    @property
    def sampling_rate(self) -> int:
        return int(self.data["sampling_rate"])

    @property
    def hop_length(self) -> int:
        return int(self.data["hop_length"])

    @property
    def downsample_factor(self) -> int:
        return int(self.data.get("downsample_factor", 2))

    @property
    def d_model(self) -> int:
        return int(self.data["d_model"])

    @property
    def encoder_layers(self) -> int:
        return int(self.data["encoder_layers"])

    @property
    def encoder_ffn_dim(self) -> int:
        return int(self.data["encoder_ffn_dim"])

    @property
    def encoder_attention_heads(self) -> int:
        return int(self.data["encoder_attention_heads"])

    @property
    def encoder_head_dim(self) -> int:
        return int(self.data.get("encoder_head_dim", self.d_model // self.encoder_attention_heads))

    @property
    def max_source_positions(self) -> int:
        value = self.data.get("max_source_positions")
        if value is None:
            return 1500
        return int(value)

    @property
    def sliding_window(self) -> Optional[int]:
        value = self.data.get("sliding_window")
        return int(value) if value is not None else None

    @property
    def global_log_mel_max(self) -> Optional[float]:
        value = self.data.get("global_log_mel_max")
        return float(value) if value is not None else None

    @property
    def max_position_embeddings(self) -> int:
        value = self.data.get("max_position_embeddings")
        if value is None:
            return self.max_source_positions
        return int(value)


@dataclass
class VoxtralRealtimeConfig:
    text: TextConfig
    audio: AudioConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_length_per_tok(self) -> int:
        return int(self.raw.get("audio_length_per_tok", 8))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "VoxtralRealtimeConfig":
        cfg_copy = cfg.copy()
        text_cfg, audio_cfg = _split_config(cfg_copy)
        return cls(
            text=TextConfig(data=text_cfg.copy()),
            audio=AudioConfig(data=audio_cfg.copy()),
            raw=cfg_copy,
        )

    @classmethod
    def from_pretrained(
        cls, model_id_or_path: str, revision: Optional[str] = None
    ) -> Tuple["VoxtralRealtimeConfig", Path]:
        model_path = get_model_path(model_id_or_path, revision=revision)
        cfg = load_config(model_path)
        return cls.from_dict(cfg), model_path


@dataclass
class ModelConfig:
    runtime: VoxtralRealtimeConfig
    model_path: Optional[str] = None
    model_type: str = "voxtral_realtime"

    def __post_init__(self) -> None:
        self.runtime = _coerce_runtime_config(self.runtime)
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
        if model_type_value is None and isinstance(runtime_cfg, VoxtralRealtimeConfig):
            model_type_value = runtime_cfg.raw.get("model_type")
        if model_type_value is None and isinstance(runtime_cfg, dict):
            model_type_value = runtime_cfg.get("model_type")
        model_type = str(model_type_value or "voxtral_realtime")

        runtime = _coerce_runtime_config(runtime_cfg)
        return cls(runtime=runtime, model_path=model_path, model_type=model_type)


def _coerce_runtime_config(
    runtime_cfg: VoxtralRealtimeConfig | Dict[str, Any],
) -> VoxtralRealtimeConfig:
    if isinstance(runtime_cfg, dict):
        return VoxtralRealtimeConfig.from_dict(runtime_cfg)
    if isinstance(runtime_cfg, VoxtralRealtimeConfig):
        return runtime_cfg
    raise TypeError("runtime must be a VoxtralRealtimeConfig or a dict-like config payload")


def _split_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # If already split
    if "text_config" in cfg and "audio_config" in cfg:
        return dict(cfg["text_config"]), _normalize_split_audio_config(cfg["audio_config"])

    # Mistral multimodal format
    if "multimodal" in cfg and "whisper_model_args" in cfg["multimodal"]:
        return _remap_mistral_audio_args(cfg)

    raise ValueError("Unsupported Voxtral config format. Expected multimodal or audio/text config.")


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


_REQUIRED_AUDIO_CONFIG_KEYS = (
    "num_mel_bins",
    "window_size",
    "sampling_rate",
    "hop_length",
    "d_model",
    "encoder_layers",
    "encoder_ffn_dim",
    "encoder_attention_heads",
)


def _normalize_split_audio_config(audio_cfg: Dict[str, Any]) -> Dict[str, Any]:
    remapped = dict(audio_cfg)
    encoding_args = remapped.get("audio_encoding_args")
    if not isinstance(encoding_args, dict):
        encoding_args = {}

    window_size = _first_not_none(
        remapped.get("window_size"),
        encoding_args.get("window_size"),
        400,
    )

    normalized = {
        "num_mel_bins": _first_not_none(
            remapped.get("num_mel_bins"),
            encoding_args.get("num_mel_bins"),
            128,
        ),
        "window_size": window_size,
        "n_fft": _first_not_none(
            remapped.get("n_fft"),
            encoding_args.get("n_fft"),
            window_size,
        ),
        "sampling_rate": _first_not_none(
            remapped.get("sampling_rate"),
            encoding_args.get("sampling_rate"),
            16000,
        ),
        "hop_length": _first_not_none(
            remapped.get("hop_length"),
            encoding_args.get("hop_length"),
            160,
        ),
        "downsample_factor": _first_not_none(remapped.get("downsample_factor"), 2),
        "d_model": _first_not_none(
            remapped.get("d_model"),
            remapped.get("hidden_size"),
            remapped.get("dim"),
        ),
        "encoder_layers": _first_not_none(
            remapped.get("encoder_layers"),
            remapped.get("num_hidden_layers"),
            remapped.get("n_layers"),
        ),
        "encoder_ffn_dim": _first_not_none(
            remapped.get("encoder_ffn_dim"),
            remapped.get("intermediate_size"),
            remapped.get("hidden_dim"),
        ),
        "encoder_attention_heads": _first_not_none(
            remapped.get("encoder_attention_heads"),
            remapped.get("num_attention_heads"),
            remapped.get("n_heads"),
        ),
        "encoder_head_dim": _first_not_none(
            remapped.get("encoder_head_dim"),
            remapped.get("head_dim"),
        ),
        "max_source_positions": _first_not_none(remapped.get("max_source_positions"), 1500),
        "sliding_window": remapped.get("sliding_window"),
        "global_log_mel_max": _first_not_none(
            remapped.get("global_log_mel_max"),
            encoding_args.get("global_log_mel_max"),
        ),
        "max_position_embeddings": remapped.get("max_position_embeddings"),
    }

    remapped.update({k: v for k, v in normalized.items() if v is not None})
    missing = [key for key in _REQUIRED_AUDIO_CONFIG_KEYS if remapped.get(key) is None]
    if missing:
        raise ValueError(
            "Unsupported split Voxtral audio_config: missing required keys after remapping: "
            + ", ".join(missing)
        )
    return remapped


def _remap_mistral_audio_args(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    whisper_args = cfg["multimodal"]["whisper_model_args"]
    encoder_args = whisper_args["encoder_args"]
    audio_encoding_args = encoder_args.get("audio_encoding_args", {})
    if not isinstance(audio_encoding_args, dict):
        audio_encoding_args = {}

    downsample_args = whisper_args.get("downsample_args", {})
    downsample_factor = _first_not_none(downsample_args.get("downsample_factor"), 2)
    text_max_position_embeddings = _first_not_none(
        cfg.get("max_position_embeddings"),
        cfg.get("model_max_length"),
        cfg.get("max_seq_len"),
        131072,
    )
    audio_max_position_embeddings = int(text_max_position_embeddings)
    if bool(encoder_args.get("causal")):
        audio_max_position_embeddings = int(downsample_factor) * audio_max_position_embeddings

    audio_cfg = _normalize_split_audio_config(
        {
            "audio_encoding_args": audio_encoding_args,
            "num_mel_bins": audio_encoding_args.get("num_mel_bins"),
            "window_size": audio_encoding_args.get("window_size"),
            "n_fft": audio_encoding_args.get("n_fft"),
            "sampling_rate": audio_encoding_args.get("sampling_rate"),
            "hop_length": audio_encoding_args.get("hop_length"),
            "downsample_factor": downsample_factor,
            "d_model": encoder_args.get("d_model"),
            "dim": encoder_args.get("dim"),
            "encoder_layers": encoder_args.get("encoder_layers"),
            "n_layers": encoder_args.get("n_layers"),
            "encoder_ffn_dim": encoder_args.get("encoder_ffn_dim"),
            "hidden_dim": encoder_args.get("hidden_dim"),
            "encoder_attention_heads": encoder_args.get("encoder_attention_heads"),
            "n_heads": encoder_args.get("n_heads"),
            "encoder_head_dim": encoder_args.get("encoder_head_dim"),
            "head_dim": encoder_args.get("head_dim"),
            "max_source_positions": encoder_args.get("max_source_positions"),
            "sliding_window": encoder_args.get("sliding_window"),
            "global_log_mel_max": audio_encoding_args.get("global_log_mel_max"),
            "max_position_embeddings": audio_max_position_embeddings,
        }
    )

    text_cfg = {
        "model_type": cfg.get("model_type", "mistral"),
        "hidden_size": cfg.get("hidden_size", cfg.get("dim")),
        "num_hidden_layers": cfg.get("num_hidden_layers", cfg.get("n_layers")),
        "num_attention_heads": cfg.get("num_attention_heads", cfg.get("n_heads")),
        "num_key_value_heads": cfg.get("num_key_value_heads", cfg.get("n_kv_heads")),
        "intermediate_size": cfg.get("intermediate_size", cfg.get("hidden_dim")),
        "rms_norm_eps": cfg.get("rms_norm_eps", cfg.get("norm_eps", 1e-5)),
        "vocab_size": cfg.get("vocab_size"),
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "head_dim": cfg.get("head_dim"),
        "sliding_window": cfg.get("sliding_window"),
        "max_position_embeddings": text_max_position_embeddings,
        "tie_word_embeddings": cfg.get("tie_word_embeddings", cfg.get("tied_embeddings", True)),
        "attention_bias": cfg.get("attention_bias", cfg.get("use_biases", False)),
        "mlp_bias": cfg.get("mlp_bias", cfg.get("use_biases", False)),
        "layer_types": cfg.get("layer_types"),
        "ada_rms_norm_t_cond": cfg.get("ada_rms_norm_t_cond", False),
        "ada_rms_norm_t_cond_dim": cfg.get("ada_rms_norm_t_cond_dim"),
    }
    return text_cfg, audio_cfg
