from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download

from .config import TextConfig, VoxtralConfig
from .model import VoxtralModel

AUDIO_REMAP = [
    (r"mm_streams_embeddings\.embedding_module\.(.*)", r"\1"),
    (r"mm_whisper_embeddings\.(.*)", r"\1"),
    (r"whisper_encoder\.conv_layers\.0\.(weight|bias)", r"audio_encoder.conv1.conv.\1"),
    (r"whisper_encoder\.conv_layers\.1\.(weight|bias)", r"audio_encoder.conv2.conv.\1"),
    (r"whisper_encoder\.conv_layers\.0\.conv\.(weight|bias)", r"audio_encoder.conv1.conv.\1"),
    (r"whisper_encoder\.conv_layers\.1\.conv\.(weight|bias)", r"audio_encoder.conv2.conv.\1"),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wq\.(weight|bias)",
        r"audio_encoder.layers.\1.self_attn.q_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wk\.(weight|bias)",
        r"audio_encoder.layers.\1.self_attn.k_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wv\.(weight|bias)",
        r"audio_encoder.layers.\1.self_attn.v_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(weight|bias)",
        r"audio_encoder.layers.\1.self_attn.out_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(weight|bias)",
        r"audio_encoder.layers.\1.self_attn_layer_norm.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(weight|bias)",
        r"audio_encoder.layers.\1.mlp.gate_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(weight|bias)",
        r"audio_encoder.layers.\1.mlp.up_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(weight|bias)",
        r"audio_encoder.layers.\1.mlp.down_proj.\2",
    ),
    (
        r"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(weight|bias)",
        r"audio_encoder.layers.\1.final_layer_norm.\2",
    ),
    (r"whisper_encoder\.transformer\.norm\.(weight|bias)", r"audio_encoder.layer_norm.\1"),
    (r"audio_language_projection\.0\.(weight|bias)", r"audio_language_adapter.w_in.\1"),
    (r"audio_language_projection\.2\.(weight|bias)", r"audio_language_adapter.w_out.\1"),
    (
        r"mm_whisper_embeddings\.audio_language_projection\.0\.(weight|bias)",
        r"audio_language_adapter.w_in.\1",
    ),
    (
        r"mm_whisper_embeddings\.audio_language_projection\.2\.(weight|bias)",
        r"audio_language_adapter.w_out.\1",
    ),
]

LLM_REMAP = [
    (r"tok_embeddings\.(weight|bias)", r"language_model.model.embed_tokens.\1"),
    (
        r"layers\.(\d+)\.attention\.wq\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.q_proj.\2",
    ),
    (
        r"layers\.(\d+)\.attention\.wk\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.k_proj.\2",
    ),
    (
        r"layers\.(\d+)\.attention\.wv\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.v_proj.\2",
    ),
    (
        r"layers\.(\d+)\.attention\.wo\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.o_proj.\2",
    ),
    (
        r"layers\.(\d+)\.attention_norm\.(weight|bias)",
        r"language_model.model.layers.\1.input_layernorm.\2",
    ),
    (
        r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)",
        r"language_model.model.layers.\1.mlp.gate_proj.\2",
    ),
    (
        r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)",
        r"language_model.model.layers.\1.mlp.up_proj.\2",
    ),
    (
        r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)",
        r"language_model.model.layers.\1.mlp.down_proj.\2",
    ),
    (
        r"layers\.(\d+)\.ffn_norm\.(weight|bias)",
        r"language_model.model.layers.\1.post_attention_layernorm.\2",
    ),
    (
        r"layers\.(\d+)\.ada_rms_norm_t_cond\.(\d+)\.(weight|bias)",
        r"language_model.model.layers.\1.ada_rms_norm_t_cond.layers.\2.\3",
    ),
    (r"norm\.(weight|bias)", r"language_model.model.norm.\1"),
    (r"output\.(weight|bias)", r"language_model.lm_head.\1"),
    # Transformers format
    (r"model\.embed_tokens\.(weight|bias)", r"language_model.model.embed_tokens.\1"),
    (
        r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.q_proj.\2",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.k_proj.\2",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.v_proj.\2",
    ),
    (
        r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)",
        r"language_model.model.layers.\1.self_attn.o_proj.\2",
    ),
    (
        r"model\.layers\.(\d+)\.input_layernorm\.(weight|bias)",
        r"language_model.model.layers.\1.input_layernorm.\2",
    ),
    (
        r"model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)",
        r"language_model.model.layers.\1.post_attention_layernorm.\2",
    ),
    (
        r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)",
        r"language_model.model.layers.\1.mlp.gate_proj.\2",
    ),
    (
        r"model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)",
        r"language_model.model.layers.\1.mlp.up_proj.\2",
    ),
    (
        r"model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)",
        r"language_model.model.layers.\1.mlp.down_proj.\2",
    ),
    (r"model\.norm\.(weight|bias)", r"language_model.model.norm.\1"),
    (r"lm_head\.(weight|bias)", r"language_model.lm_head.\1"),
]


def remap_weights(
    weights: Dict[str, mx.array], text_cfg: Optional[TextConfig] = None
) -> Dict[str, mx.array]:
    remapped: Dict[str, mx.array] = {}
    for name, value in weights.items():
        if text_cfg is not None:
            value = _maybe_permute_rope(name, value, text_cfg)
        new_name = _apply_rules_until_fixed(name, AUDIO_REMAP)
        new_name = _apply_rules_until_fixed(new_name, LLM_REMAP)
        remapped[new_name] = value
    return remapped


def align_weights(model: VoxtralModel, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    aligned: Dict[str, mx.array] = {}
    for name, value in weights.items():
        if name not in params:
            continue
        param = params[name]
        if value.shape != param.shape and value.ndim == 3 and param.ndim == 3:
            if value.transpose(0, 2, 1).shape == param.shape:
                value = value.transpose(0, 2, 1)
        aligned[name] = value
    return aligned


def _apply_rules(name: str, rules: Iterable[Tuple[str, str]]) -> str:
    for pattern, repl in rules:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, repl, name)
    return name


def _apply_rules_until_fixed(name: str, rules: Iterable[Tuple[str, str]]) -> str:
    prev = None
    curr = name
    while curr != prev:
        prev = curr
        curr = _apply_rules(curr, rules)
    return curr


def _permute_for_rope(tensor: mx.array, n_heads: int, dim1: int, dim2: int) -> mx.array:
    tensor = tensor.reshape(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(0, 2, 1, 3)
    return tensor.reshape(dim1, dim2)


def _maybe_permute_rope(name: str, value: mx.array, text_cfg: TextConfig) -> mx.array:
    if value.ndim != 2:
        return value

    head_dim = int(
        text_cfg.data.get("head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    )
    hidden_size = text_cfg.hidden_size

    if re.fullmatch(r"layers\.\d+\.attention\.wq\.weight", name):
        query_dim = text_cfg.num_attention_heads * head_dim
        if value.shape != (query_dim, hidden_size):
            return value
        return _permute_for_rope(value, text_cfg.num_attention_heads, query_dim, hidden_size)

    if re.fullmatch(r"layers\.\d+\.attention\.wk\.weight", name):
        key_dim = text_cfg.num_key_value_heads * head_dim
        if value.shape != (key_dim, hidden_size):
            return value
        return _permute_for_rope(value, text_cfg.num_key_value_heads, key_dim, hidden_size)

    return value


def convert_to_mlx(model_id_or_path: str, output_dir: str, revision: str | None = None) -> Path:
    config, _ = VoxtralConfig.from_pretrained(model_id_or_path, revision=revision)
    model = VoxtralModel(config.text, config.audio)

    weights = _load_raw_weights(model_id_or_path, revision)
    weights = remap_weights(weights, text_cfg=config.text)
    weights = align_weights(model, weights)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.raw, f, indent=2)
    return out


def _load_raw_weights(model_id_or_path: str, revision: str | None) -> Dict[str, mx.array]:
    path = Path(model_id_or_path)
    if path.exists():
        return _load_safetensors(path)
    snapshot_dir = snapshot_download(
        model_id_or_path,
        revision=revision,
        allow_patterns=["*.safetensors", "params.json", "config.json"],
    )
    return _load_safetensors(Path(snapshot_dir))


def _load_safetensors(path: Path) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    for file in sorted(path.glob("*.safetensors")):
        arrays = mx.load(str(file))
        for name, value in arrays.items():
            weights[name] = value
    return weights


def main() -> None:
    parser = argparse.ArgumentParser("mlx_audio.stt.voxtral.converter")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    convert_to_mlx(args.model, args.output, revision=args.revision)


if __name__ == "__main__":
    main()
