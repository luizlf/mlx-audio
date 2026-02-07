import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput
from .api import VoxtralRealtime, load_tokenizer
from .audio import iter_chunks
from .config import ModelConfig, TextConfig
from .model import VoxtralModel


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


AUDIO_REMAP = [
    (r"mm_streams_embeddings\.embedding_module\.(.*)", r"\1"),
    (r"mm_whisper_embeddings\.whisper_encoder\.(.*)", r"whisper_encoder.\1"),
    (
        r"mm_whisper_embeddings\.audio_language_projection\.(.*)",
        r"audio_language_projection.\1",
    ),
    (
        r"whisper_encoder\.conv_layers\.0\.conv\.(weight|bias)",
        r"audio_encoder.conv1.conv.\1",
    ),
    (
        r"whisper_encoder\.conv_layers\.1\.conv\.(weight|bias)",
        r"audio_encoder.conv2.conv.\1",
    ),
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
    (
        r"whisper_encoder\.transformer\.norm\.(weight|bias)",
        r"audio_encoder.layer_norm.\1",
    ),
    (r"audio_language_projection\.0\.(weight|bias)", r"audio_language_adapter.w_in.\1"),
    (
        r"audio_language_projection\.2\.(weight|bias)",
        r"audio_language_adapter.w_out.\1",
    ),
]

LLM_REMAP = [
    (r"mm_whisper_embeddings\.tok_embeddings\.(weight|bias)", r"tok_embeddings.\1"),
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
]


def remap_weights(
    weights: Dict[str, mx.array], text_cfg: Optional[TextConfig] = None
) -> Dict[str, mx.array]:
    remapped: Dict[str, mx.array] = {}
    for name, value in weights.items():
        if text_cfg is not None:
            value = _maybe_permute_source_qk_for_rope(name, value, text_cfg)
        new_name = _apply_rules_until_fixed(name, AUDIO_REMAP)
        new_name = _apply_rules_until_fixed(new_name, LLM_REMAP)
        remapped[new_name] = value
    return remapped


def align_weights(
    model: VoxtralModel, weights: Dict[str, mx.array]
) -> Dict[str, mx.array]:
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    aligned: Dict[str, mx.array] = {}
    for name, value in weights.items():
        if name not in params:
            continue
        param = params[name]
        if value.shape != param.shape and value.ndim == 3 and param.ndim == 3:
            transposed = value.transpose(0, 2, 1)
            if transposed.shape == param.shape:
                value = transposed
        aligned[name] = value
    return aligned


def _apply_rules_until_fixed(name: str, rules: Iterable[Tuple[str, str]]) -> str:
    prev = None
    curr = name
    while curr != prev:
        prev = curr
        for pattern, repl in rules:
            if re.fullmatch(pattern, curr):
                curr = re.sub(pattern, repl, curr)
                break
    return curr


def _permute_source_qk_for_rope(
    tensor: mx.array, n_heads: int, dim1: int, dim2: int
) -> mx.array:
    tensor = tensor.reshape(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(0, 2, 1, 3)
    return tensor.reshape(dim1, dim2)


# Voxtral Realtime source checkpoints store attention wq/wk weights in a
# pre-RoPE layout. Before remapping to q_proj/k_proj names, reorder those
# weights to the layout expected by MLX-LM attention modules.
def _maybe_permute_source_qk_for_rope(
    name: str, value: mx.array, text_cfg: TextConfig
) -> mx.array:
    if value.ndim != 2:
        return value

    head_dim = int(
        text_cfg.data.get(
            "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads
        )
    )
    hidden_size = text_cfg.hidden_size

    if re.fullmatch(r"layers\.\d+\.attention\.wq\.weight", name):
        query_dim = text_cfg.num_attention_heads * head_dim
        if value.shape != (query_dim, hidden_size):
            return value
        return _permute_source_qk_for_rope(
            value, text_cfg.num_attention_heads, query_dim, hidden_size
        )

    if re.fullmatch(r"layers\.\d+\.attention\.wk\.weight", name):
        key_dim = text_cfg.num_key_value_heads * head_dim
        if value.shape != (key_dim, hidden_size):
            return value
        return _permute_source_qk_for_rope(
            value, text_cfg.num_key_value_heads, key_dim, hidden_size
        )

    return value


class Model(VoxtralModel):
    def __init__(self, config: ModelConfig):
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        super().__init__(config.text_config, config.audio_config)
        self.config = config
        self._tokenizer = None
        self._runtime: Optional[VoxtralRealtime] = None

    @property
    def sample_rate(self) -> int:
        return int(self.config.audio_config.sampling_rate)

    def _build_runtime(self) -> VoxtralRealtime:
        if self._tokenizer is None:
            raise RuntimeError(
                "Voxtral tokenizer not initialized. Use mlx_audio.stt.load() to load this model."
            )
        return VoxtralRealtime(
            model=self,
            tokenizer=self._tokenizer,
            config=self.config.runtime,
        )

    def _get_runtime(self) -> VoxtralRealtime:
        if self._runtime is None:
            self._runtime = self._build_runtime()
        return self._runtime

    def sanitize(self, weights):
        remapped = remap_weights(weights, text_cfg=self.config.text_config)
        aligned = align_weights(self, remapped)
        quant_suffixes = (".scales", ".biases", ".zeros", ".qweight")
        for name, value in remapped.items():
            if name in aligned:
                continue
            if name.endswith(quant_suffixes):
                aligned[name] = value
        return aligned

    def model_quant_predicate(self, path, module):
        return path.startswith("language_model") and isinstance(module, nn.Linear)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        tokenizer = load_tokenizer(str(model_path), revision=None)
        model._tokenizer = tokenizer
        model._runtime = model._build_runtime()
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        del config
        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from mlx_audio.stt.utils import load

        return load(model_path, **kwargs)

    def _load_audio_input(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
    ) -> np.ndarray:
        from mlx_audio.stt.utils import load_audio

        audio_input = audio[0] if isinstance(audio, list) else audio
        if isinstance(audio_input, str):
            audio_input = load_audio(audio_input, sr=self.sample_rate)
        if isinstance(audio_input, mx.array):
            audio_input = np.array(audio_input)
        if not isinstance(audio_input, np.ndarray):
            raise TypeError(f"Unsupported audio type: {type(audio_input)}")
        return audio_input.astype(np.float32)

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int = 256,
        language: str = "en",
        verbose: bool = False,
        stream: bool = False,
        stream_strategy: str = "stable",
        stream_refresh_reads: int = 50,
        realtime_carry_policy: str = "exact",
        realtime_auto_fallback: bool = True,
        realtime_fallback_refresh_reads: int = 50,
        generation_stream: Optional[mx.Stream] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        **kwargs,
    ):
        del (
            generation_stream,
            temperature,
            top_p,
            top_k,
            min_p,
            min_tokens_to_keep,
            kwargs,
            verbose,
        )

        runtime = self._get_runtime()
        audio_np = self._load_audio_input(audio)

        if stream:
            audio_encoder = runtime.tokenizer.instruct_tokenizer.audio_encoder
            if audio_encoder is None:
                raise ValueError("Tokenizer is missing audio encoder configuration")
            audio_cfg = audio_encoder.audio_config
            chunk_size = int(audio_cfg.sampling_rate / audio_cfg.frame_rate)

            def _stream() -> Generator[StreamingResult, None, None]:
                for text in runtime.stream_transcribe(
                    iter_chunks(audio_np, chunk_size),
                    language=language,
                    max_tokens=max_tokens,
                    strategy=stream_strategy,
                    offline_refresh_reads=stream_refresh_reads,
                    realtime_carry_policy=realtime_carry_policy,
                    realtime_auto_fallback=realtime_auto_fallback,
                    realtime_fallback_refresh_reads=realtime_fallback_refresh_reads,
                ):
                    if text:
                        yield StreamingResult(
                            text=text, is_final=False, language=language
                        )
                yield StreamingResult(text="", is_final=True, language=language)

            return _stream()

        start_time = time.time()
        out = runtime.transcribe(audio_np, language=language, max_tokens=max_tokens)
        total_time = time.time() - start_time
        duration = len(audio_np) / float(self.sample_rate)
        return STTOutput(
            text=out.text,
            segments=[
                {
                    "text": out.text,
                    "start": 0.0,
                    "end": duration,
                }
            ],
            language=language,
            prompt_tokens=out.prompt_tokens,
            generation_tokens=out.generation_tokens,
            total_tokens=out.prompt_tokens + out.generation_tokens,
            total_time=total_time,
            prompt_tps=out.prompt_tokens / total_time if total_time > 0 else 0.0,
            generation_tps=(
                out.generation_tokens / total_time if total_time > 0 else 0.0
            ),
        )

    def stream_transcribe(
        self, audio, **kwargs
    ) -> Generator[StreamingResult, None, None]:
        return self.generate(audio, stream=True, **kwargs)
