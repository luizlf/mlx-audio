from __future__ import annotations

import mlx.core as mx

from mlx_audio.stt.models.voxtral_realtime.model import CausalConv1d


def test_causal_conv1d_matches_vllm_odd_length_no_cache_shape() -> None:
    conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    x = mx.zeros((1, 7, 1), dtype=mx.float32)

    y = conv(x)

    # vLLM whisper-causal behavior uses ceil length for stride-2 odd inputs.
    assert int(y.shape[1]) == 4
