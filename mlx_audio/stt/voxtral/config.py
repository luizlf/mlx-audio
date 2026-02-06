from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from huggingface_hub import hf_hub_download


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
            value = self.data.get("model_max_length", 131072)
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
    def is_causal(self) -> bool:
        return bool(self.data.get("is_causal", False))

    @property
    def sliding_window(self) -> Optional[int]:
        value = self.data.get("sliding_window")
        return int(value) if value is not None else None

    @property
    def block_pool_size(self) -> int:
        return int(self.data.get("block_pool_size", 1))

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
class VoxtralConfig:
    text: TextConfig
    audio: AudioConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_length_per_tok(self) -> int:
        return int(self.raw.get("audio_length_per_tok", 8))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "VoxtralConfig":
        text_cfg, audio_cfg = _split_config(cfg)
        return cls(text=TextConfig(text_cfg), audio=AudioConfig(audio_cfg), raw=cfg)

    @classmethod
    def from_pretrained(
        cls, model_id_or_path: str, revision: Optional[str] = None
    ) -> Tuple["VoxtralConfig", Path]:
        path = Path(model_id_or_path)
        if path.exists():
            cfg_path = _resolve_config_path(path)
        else:
            cfg_path = _download_config(model_id_or_path, revision=revision)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg), cfg_path.parent


def _download_config(model_id: str, revision: Optional[str]) -> Path:
    # Prefer params.json for Mistral weights, fallback to config.json
    for filename in ("params.json", "config.json"):
        try:
            return Path(hf_hub_download(model_id, filename=filename, revision=revision))
        except Exception:
            continue
    raise FileNotFoundError("Could not download params.json or config.json from Hugging Face")


def _resolve_config_path(path: Path) -> Path:
    for name in ("params.json", "config.json"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No params.json or config.json under {path}")


def _split_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # If already split
    if "text_config" in cfg and "audio_config" in cfg:
        return _remap_split_configs(cfg)

    # Mistral multimodal format
    if "multimodal" in cfg and "whisper_model_args" in cfg["multimodal"]:
        return _remap_mistral_audio_args(cfg)

    raise ValueError("Unsupported Voxtral config format. Expected multimodal or audio/text config.")


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _remap_split_configs(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    text_cfg = _normalize_split_text_config(cfg["text_config"])
    audio_cfg = _normalize_split_audio_config(cfg["audio_config"])
    return text_cfg, audio_cfg


def _normalize_split_text_config(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    remapped = dict(text_cfg)

    alias_pairs = (
        ("hidden_size", ("hidden_size", "dim")),
        ("num_hidden_layers", ("num_hidden_layers", "n_layers")),
        ("num_attention_heads", ("num_attention_heads", "n_heads")),
        ("num_key_value_heads", ("num_key_value_heads", "n_kv_heads")),
        ("intermediate_size", ("intermediate_size", "hidden_dim")),
        ("rms_norm_eps", ("rms_norm_eps", "norm_eps")),
        ("max_position_embeddings", ("max_position_embeddings", "model_max_length")),
        ("tie_word_embeddings", ("tie_word_embeddings", "tied_embeddings")),
    )
    for out_key, in_keys in alias_pairs:
        value = _first_not_none(*(remapped.get(k) for k in in_keys))
        if value is not None:
            remapped[out_key] = value

    if remapped.get("num_key_value_heads") is None:
        remapped["num_key_value_heads"] = remapped.get("num_attention_heads")

    return remapped


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
        "vocab_size": remapped.get("vocab_size"),
        "max_source_positions": _first_not_none(remapped.get("max_source_positions"), 1500),
        "is_causal": _first_not_none(remapped.get("is_causal"), remapped.get("causal"), False),
        "sliding_window": remapped.get("sliding_window"),
        "block_pool_size": _first_not_none(remapped.get("block_pool_size"), 1),
        "pos_embed": _first_not_none(remapped.get("pos_embed"), "sinusoidal"),
        "global_log_mel_max": _first_not_none(
            remapped.get("global_log_mel_max"),
            encoding_args.get("global_log_mel_max"),
        ),
        "max_position_embeddings": remapped.get("max_position_embeddings"),
    }

    remapped.update({k: v for k, v in normalized.items() if v is not None})

    required_keys = (
        "num_mel_bins",
        "window_size",
        "sampling_rate",
        "hop_length",
        "d_model",
        "encoder_layers",
        "encoder_ffn_dim",
        "encoder_attention_heads",
    )
    missing = [key for key in required_keys if remapped.get(key) is None]
    if missing:
        raise ValueError(
            "Unsupported split Voxtral audio_config: missing required keys after remapping: "
            + ", ".join(missing)
        )

    return remapped


def _remap_mistral_audio_args(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    whisper_args = cfg["multimodal"]["whisper_model_args"]
    encoder_args = whisper_args["encoder_args"]
    downsample_factor = whisper_args["downsample_args"]["downsample_factor"]

    block_pool_size = downsample_factor if encoder_args.get("causal") else 1

    audio_cfg = {
        "num_mel_bins": encoder_args["audio_encoding_args"]["num_mel_bins"],
        "window_size": encoder_args["audio_encoding_args"]["window_size"],
        "n_fft": encoder_args["audio_encoding_args"].get(
            "n_fft", encoder_args["audio_encoding_args"]["window_size"]
        ),
        "sampling_rate": encoder_args["audio_encoding_args"]["sampling_rate"],
        "hop_length": encoder_args["audio_encoding_args"]["hop_length"],
        "downsample_factor": downsample_factor,
        "d_model": encoder_args["dim"],
        "encoder_layers": encoder_args["n_layers"],
        "encoder_ffn_dim": encoder_args["hidden_dim"],
        "encoder_attention_heads": encoder_args["n_heads"],
        "encoder_head_dim": encoder_args.get(
            "head_dim", encoder_args["dim"] // encoder_args["n_heads"]
        ),
        "vocab_size": encoder_args["vocab_size"],
        "max_source_positions": encoder_args["max_source_positions"],
        "is_causal": encoder_args.get("causal", False),
        "sliding_window": encoder_args.get("sliding_window"),
        "block_pool_size": block_pool_size,
        "pos_embed": encoder_args.get("pos_embed", "sinusoidal"),
        "global_log_mel_max": encoder_args["audio_encoding_args"].get("global_log_mel_max"),
        "max_position_embeddings": block_pool_size * cfg.get("max_position_embeddings", 131072),
    }

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
        "max_position_embeddings": cfg.get("max_position_embeddings", cfg.get("model_max_length")),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", cfg.get("tied_embeddings", True)),
        "attention_bias": cfg.get("attention_bias", cfg.get("use_biases", False)),
        "mlp_bias": cfg.get("mlp_bias", cfg.get("use_biases", False)),
        "layer_types": cfg.get("layer_types"),
        "ada_rms_norm_t_cond": cfg.get("ada_rms_norm_t_cond", False),
        "ada_rms_norm_t_cond_dim": cfg.get("ada_rms_norm_t_cond_dim"),
    }
    return text_cfg, audio_cfg
