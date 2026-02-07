from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.rope_utils import initialize_rope

from .config import AudioConfig, TextConfig


@dataclass
class MistralArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rope_traditional: bool
    rope_scaling: Optional[dict]
    tie_word_embeddings: bool
    layer_types: Optional[List[str]]
    sliding_window: Optional[int]
    ada_rms_norm_t_cond: bool
    ada_rms_norm_t_cond_dim: Optional[int]
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_config(cls, cfg: TextConfig) -> "MistralArgs":
        data = cfg.data
        head_dim = data.get("head_dim")
        if head_dim is None:
            head_dim = cfg.hidden_size // cfg.num_attention_heads
        return cls(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            intermediate_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            rms_norm_eps=cfg.rms_norm_eps,
            vocab_size=cfg.vocab_size,
            head_dim=int(head_dim),
            max_position_embeddings=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
            rope_traditional=bool(data.get("rope_traditional", False)),
            rope_scaling=data.get("rope_scaling"),
            tie_word_embeddings=cfg.tie_word_embeddings,
            layer_types=cfg.layer_types,
            sliding_window=cfg.sliding_window,
            ada_rms_norm_t_cond=cfg.ada_rms_norm_t_cond,
            ada_rms_norm_t_cond_dim=cfg.ada_rms_norm_t_cond_dim,
            attention_bias=bool(data.get("attention_bias", False)),
            mlp_bias=bool(data.get("mlp_bias", False)),
        )


# Keep a local RMSNorm to force fp32 normalization; nn.RMSNorm shows larger
# low-precision drift (especially bf16), which breaks parity-sensitive paths.
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states_fp32 * hidden_states_fp32, axis=-1, keepdims=True)
        normalized = hidden_states_fp32 * mx.rsqrt(variance + self.eps)
        return normalized.astype(input_dtype) * self.weight


class MistralAttention(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        bsz, seq_len, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(bsz, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(bsz, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(bsz, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.o_proj(output)


class MistralMLP(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MistralBlock(nn.Module):
    def __init__(self, args: MistralArgs, use_sliding: bool = False):
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = MistralAttention(args)
        self.mlp = MistralMLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.ada_rms_norm_t_cond:
            dim = args.ada_rms_norm_t_cond_dim or args.hidden_size
            self.ada_rms_norm_t_cond = nn.Sequential(
                nn.Linear(args.hidden_size, dim, bias=False),
                nn.GELU(),
                nn.Linear(dim, args.hidden_size, bias=False),
            )
        else:
            self.ada_rms_norm_t_cond = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        h_norm = self.post_attention_layernorm(h)
        if self.ada_rms_norm_t_cond is not None:
            if t_cond is None:
                raise ValueError("t_cond is required when ada_rms_norm_t_cond is enabled")
            h_norm = h_norm * (1 + self.ada_rms_norm_t_cond(t_cond))
        r2 = self.mlp(h_norm)
        return h + r2


class MistralModel(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args
        if args.layer_types is not None:
            self.layer_types = args.layer_types
        elif args.sliding_window is not None:
            self.layer_types = ["sliding_attention"] * args.num_hidden_layers
        else:
            self.layer_types = ["full_attention"] * args.num_hidden_layers
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.swa_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )
        self.layers = [
            MistralBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = (
            self.layer_types.index("full_attention") if "full_attention" in self.layer_types else 0
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )
        else:
            swa_mask = None

        for layer, layer_cache in zip(self.layers, cache, strict=False):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=layer_cache, t_cond=t_cond)

        return self.norm(h)


class MistralForCausalLM(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args
        self.model = MistralModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def embed_input_ids(self, input_ids: mx.array) -> mx.array:
        return self.model.embed_tokens(input_ids)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings, t_cond=t_cond)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )
        self._padding_total = (kernel_size - 1) - (stride - 1)
        self.in_channels = in_channels

    def __call__(self, x: mx.array, padding_state: Optional["Conv1dCacheLayer"] = None) -> mx.array:
        # x shape: (batch, length, channels)
        if padding_state is None:
            x = mx.pad(x, [(0, 0), (int(self._padding_total), 0), (0, 0)])
        else:
            x = padding_state.update(x)
        return self.conv(x)


class Conv1dCacheLayer:
    def __init__(self, in_channels: int, left_pad: int):
        self.in_channels = in_channels
        self.left_pad = left_pad
        self.cache: Optional[mx.array] = None

    def update(self, hidden_states: mx.array) -> mx.array:
        if self.left_pad <= 0:
            return hidden_states

        batch = hidden_states.shape[0]
        if self.cache is None:
            self.cache = mx.zeros(
                (batch, self.left_pad, self.in_channels), dtype=hidden_states.dtype
            )

        padded = mx.concatenate([self.cache, hidden_states], axis=1)
        self.cache = padded[:, -self.left_pad :, :]
        return padded


class Conv1dPaddingCache:
    def __init__(self, conv1: CausalConv1d, conv2: CausalConv1d):
        self.layers = [
            Conv1dCacheLayer(conv1.in_channels, int(conv1._padding_total)),
            Conv1dCacheLayer(conv2.in_channels, int(conv2._padding_total)),
        ]


class VoxtralRealtimeAudioAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        dim = config.d_model
        self.n_heads = config.encoder_attention_heads
        self.head_dim = config.encoder_head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=True)
        rope_theta = float(getattr(config, "rope_theta", 1e6))
        self.rope = initialize_rope(
            self.head_dim,
            rope_theta,
            False,
            None,
            config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_offset: int = 0,
    ) -> mx.array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            rope_offset = int(position_offset + int(cache.offset))
            q = self.rope(q, offset=rope_offset)
            k = self.rope(k, offset=rope_offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            if position_offset:
                q = self.rope(q, offset=int(position_offset))
                k = self.rope(k, offset=int(position_offset))
            else:
                q = self.rope(q)
                k = self.rope(k)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.out_proj(out)


class VoxtralRealtimeAudioMLP(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        dim = config.d_model
        hidden_dim = config.encoder_ffn_dim
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class VoxtralRealtimeAudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.self_attn = VoxtralRealtimeAudioAttention(config)
        self.self_attn_layer_norm = RMSNorm(config.d_model, eps=1e-5)
        self.mlp = VoxtralRealtimeAudioMLP(config)
        self.final_layer_norm = RMSNorm(config.d_model, eps=1e-5)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
        position_offset: int = 0,
    ) -> mx.array:
        r = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask, cache=cache, position_offset=position_offset)
        x = r + x
        r = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        return r + x


class VoxtralRealtimeAudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.num_mel_bins = config.num_mel_bins
        self.conv1 = CausalConv1d(self.num_mel_bins, config.d_model, kernel_size=3, stride=1)
        self.conv2 = CausalConv1d(config.d_model, config.d_model, kernel_size=3, stride=2)
        self.layers = [VoxtralRealtimeAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = RMSNorm(config.d_model, eps=1e-5)

    def _forward_conv(
        self,
        feat: mx.array,
        padding_cache: Optional[Conv1dPaddingCache] = None,
    ) -> mx.array:
        if feat.ndim == 2:
            feat = feat.T[None, ...]
        c1_cache = padding_cache.layers[0] if padding_cache is not None else None
        c2_cache = padding_cache.layers[1] if padding_cache is not None else None
        x = nn.gelu(self.conv1(feat, padding_state=c1_cache))
        x = nn.gelu(self.conv2(x, padding_state=c2_cache))
        return x

    def forward_conv_features(
        self,
        feat: mx.array,
        padding_cache: Optional[Conv1dPaddingCache] = None,
    ) -> tuple[mx.array, Optional[Conv1dPaddingCache]]:
        x = self._forward_conv(feat, padding_cache=padding_cache)
        return x[0], padding_cache

    def forward_transformer(
        self,
        hidden_states: mx.array,
        cache: Optional[List[Any]] = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, Optional[List[Any]]]:
        x = hidden_states[None, ...]
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache, strict=False):
            mask = create_attention_mask(x, layer_cache, window_size=self.config.sliding_window)
            x = layer(x, mask, cache=layer_cache, position_offset=position_offset)
        x = self.layer_norm(x)
        return x[0], cache

    def __call__(self, input_features: List[mx.array]) -> List[mx.array]:
        outputs: List[mx.array] = []
        for feat in input_features:
            x, _ = self.forward_conv_features(feat)
            x, _ = self.forward_transformer(x)
            outputs.append(x)
        return outputs

    def make_cache(self) -> List[Any]:
        from mlx_lm.models import cache as lm_cache

        max_size = self.config.sliding_window or 750
        return [lm_cache.RotatingKVCache(max_size=max_size, keep=0) for _ in self.layers]

    def make_padding_cache(self) -> Conv1dPaddingCache:
        return Conv1dPaddingCache(self.conv1, self.conv2)


class AudioLanguageAdapter(nn.Module):
    def __init__(self, hidden_size: int, dim: int):
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_out(nn.gelu(self.w_in(x)))


class VoxtralModel(nn.Module):
    def __init__(self, text_cfg: TextConfig, audio_cfg: AudioConfig):
        super().__init__()
        self.language_model = MistralForCausalLM(MistralArgs.from_config(text_cfg))
        self.audio_encoder = VoxtralRealtimeAudioEncoder(audio_cfg)
        self.audio_language_adapter = AudioLanguageAdapter(
            hidden_size=audio_cfg.encoder_ffn_dim,
            dim=text_cfg.hidden_size,
        )

    @property
    def layers(self):
        return self.language_model.layers

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        if t_cond is None and hasattr(self, "_t_cond"):
            t_cond = self._t_cond
        return self.language_model(
            input_ids, cache=cache, input_embeddings=input_embeddings, t_cond=t_cond
        )
