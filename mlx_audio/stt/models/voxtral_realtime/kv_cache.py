from __future__ import annotations

from typing import Optional

import mlx.core as mx


class VoxtralSlidingWindowKVCache:
    """Sliding-window KV cache optimized for fixed-size chunk appends.

    This avoids repeated concat-heavy growth when appending multi-token chunks.
    """

    def __init__(self, *, window_size: int, chunk_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self.window_size = int(window_size)
        self.chunk_size = int(chunk_size)
        self.capacity = int(self.window_size + self.chunk_size - 1)

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

        self.offset = 0
        self._idx = 0
        self._filled = 0

    def size(self) -> int:
        return int(min(self.offset, self.capacity))

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        B, n_kv_heads, S, k_head_dim = keys.shape
        if S != self.chunk_size:
            raise ValueError(
                f"Expected chunk_size={self.chunk_size}, got {S}. "
                "VoxtralSlidingWindowKVCache requires fixed-size appends."
            )

        if self.keys is None:
            v_head_dim = int(values.shape[3])
            self.keys = mx.zeros((B, n_kv_heads, self.capacity, k_head_dim), dtype=keys.dtype)
            self.values = mx.zeros((B, n_kv_heads, self.capacity, v_head_dim), dtype=values.dtype)
            self._filled = 0
            self._idx = 0
            self.offset = 0

        if self._filled < self.capacity:
            if self._filled + S <= self.capacity:
                start = self._filled
                self.keys[..., start : start + S, :] = keys
                self.values[..., start : start + S, :] = values
                self._filled += S
                self.offset += S
                if self._filled == self.capacity:
                    self._idx = 0
                    return self.keys, self.values
                self._idx = self._filled
                return (
                    self.keys[..., : self._filled, :],
                    self.values[..., : self._filled, :],
                )

            past_keep = min(self.window_size - 1, self._filled)
            if past_keep:
                src_start = self._filled - past_keep
                self.keys[..., :past_keep, :] = self.keys[..., src_start : self._filled, :]
                self.values[..., :past_keep, :] = self.values[..., src_start : self._filled, :]

            self.keys[..., past_keep : past_keep + S, :] = keys
            self.values[..., past_keep : past_keep + S, :] = values

            self._filled = past_keep + S
            self.offset += S
            if self._filled == self.capacity:
                self._idx = 0
                return self.keys, self.values

            self._idx = self._filled
            return (
                self.keys[..., : self._filled, :],
                self.values[..., : self._filled, :],
            )

        start = int(self._idx)
        first = min(S, self.capacity - start)
        if first:
            self.keys[..., start : start + first, :] = keys[..., :first, :]
            self.values[..., start : start + first, :] = values[..., :first, :]
        if first < S:
            remaining = S - first
            self.keys[..., :remaining, :] = keys[..., first:, :]
            self.values[..., :remaining, :] = values[..., first:, :]

        self.offset += S
        self._idx = (start + S) % self.capacity
        return self.keys, self.values

    def make_mask(
        self,
        N: int,
        *,
        window_size: Optional[int] = None,
        return_array: bool = False,
    ):
        if N <= 1:
            return None

        ws = int(window_size or self.window_size)
        offset = min(ws - 1, int(self.offset))
        if not return_array and (offset + N) <= ws:
            return "causal"

        from mlx_lm.models.cache import create_causal_mask

        mask = create_causal_mask(N, offset=offset, window_size=ws)

        if self._filled >= self.capacity:
            start_after = (int(self._idx) + int(N)) % int(self.capacity)
            if start_after:
                mask = mx.roll(mask, shift=int(start_after), axis=1)

        return mask
