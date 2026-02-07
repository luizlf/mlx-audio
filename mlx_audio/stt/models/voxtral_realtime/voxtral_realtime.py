import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput
from .api import VoxtralRealtime, load_tokenizer
from .audio import iter_chunks
from .converter import align_weights, remap_weights
from .model import VoxtralModel

from .model_config import ModelConfig


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


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
        if self._runtime is not None:
            return self._runtime
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
                        yield StreamingResult(text=text, is_final=False, language=language)
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
            generation_tps=out.generation_tokens / total_time if total_time > 0 else 0.0,
        )

    def stream_transcribe(self, audio, **kwargs) -> Generator[StreamingResult, None, None]:
        kwargs = kwargs.copy()
        kwargs["stream"] = True
        return self.generate(audio, **kwargs)
