"""Local chat backend: runs a vision-language checkpoint with transformers.

ParseBench measured these models behind vLLM. To keep outputs comparable the
backend reproduces what that server does with each request:

- the chat messages are rendered with the checkpoint's own chat template and
  the request's ``chat_template_kwargs``,
- sampling follows vLLM: ``temperature`` below 1e-5 or ``top_k: 1`` is greedy,
  unset values fall back to the checkpoint's ``generation_config.json``,
- presence/frequency penalties use vLLM semantics (output tokens only),
- top-level ``extra_args`` are ignored, exactly like stock vLLM ignores them,
- ``max_tokens`` is capped by the context length the benchmark served with,
- output stops at EOS and is decoded with the request's ``skip_special_tokens``.

Concurrent requests with identical sampling settings are micro-batched, which
is where most of the speed on a single GPU comes from.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

# Must be set before torch is imported. Without it ROCm builds fall back to the
# math SDPA kernel, which makes the vision encoder ~8x slower on large crops.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

logger = logging.getLogger(__name__)


class LocalInferenceError(RuntimeError):
    pass


def local_inference_missing() -> str | None:
    """Return why local inference cannot run, or None when the packages exist."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed (run scripts/install-torch)"
    if importlib.util.find_spec("transformers") is None:
        return "transformers is not installed (uv sync --extra local --inexact)"
    return None


_device_info: dict[str, dict[str, Any]] = {}
_device_info_lock = threading.Lock()


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(preference: str, device: str):
    import torch

    if preference != "auto":
        return getattr(torch, preference)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def describe_local_device(preference: str = "auto") -> dict[str, Any]:
    """Name the accelerator local models would run on (cached; imports torch)."""
    with _device_info_lock:
        if preference in _device_info:
            return dict(_device_info[preference])
        missing = local_inference_missing()
        if missing:
            return {"available": False, "detail": missing}
        import torch

        device = resolve_device(preference)
        info: dict[str, Any] = {"available": True, "device": device, "torch": torch.__version__}
        if device == "cuda":
            if not torch.cuda.is_available():
                return {"available": False, "detail": "LOCAL_DEVICE=cuda but no GPU is available."}
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["memory_gb"] = round(props.total_memory / 1e9, 1)
            info["runtime"] = "ROCm" if torch.version.hip else "CUDA"
        elif device == "mps":
            info["device_name"] = "Apple GPU"
            info["runtime"] = "MPS"
        else:
            info["device_name"] = "CPU"
            info["runtime"] = "CPU"
        _device_info[preference] = info
        return dict(info)


def free_accelerator_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class LocalModel(Protocol):
    """Something the pool can load, keep and unload (a checkpoint plus its runtime)."""

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _Sampling:
    max_tokens: int
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    skip_special_tokens: bool
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    require_complete: bool = False

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5 or self.top_k == 1


@dataclass
class _Pending:
    sampling: _Sampling
    prompt: str
    images: list[Image.Image]
    future: asyncio.Future


def _decode_data_uri(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only data: image URLs are supported by the local backend")
    _, b64 = url.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    image.load()
    return image.convert("RGB")


class LocalVisionModel:
    """One loaded checkpoint plus its request batcher. Implements ChatBackend."""

    def __init__(
        self,
        repo: str,
        revision: str | None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_batch: int = 8,
        max_model_len: int = 8192,
    ):
        self.repo = repo
        self.revision = revision
        self._device_pref = device
        self._dtype_pref = dtype
        self.max_batch = max(1, max_batch)
        self.max_model_len = max_model_len
        self.device: str | None = None
        self.dtype_name: str | None = None
        self._model = None
        self._processor = None
        self._eos_ids: set[int] = set()
        self._pad_id: int | None = None
        self._defaults = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
        self._queue: deque[_Pending] = deque()
        self._wakeup: asyncio.Event | None = None
        self._worker: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gpu_lock = threading.Lock()

    # ------------------------------------------------------------------ load
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Blocking: load weights onto the device and run a short warm-up."""
        missing = local_inference_missing()
        if missing:
            raise LocalInferenceError(missing)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = resolve_device(self._device_pref)
        dtype = resolve_dtype(self._dtype_pref, device)
        processor = AutoProcessor.from_pretrained(self.repo, revision=self.revision)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            self.repo, revision=self.revision, dtype=dtype
        )
        patch_embed_convs_as_matmul(model)
        model.to(device)
        model.eval()

        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        if processor.tokenizer.eos_token_id is not None:
            eos_ids.add(processor.tokenizer.eos_token_id)
        pad = model.generation_config.pad_token_id
        if pad is None:
            pad = processor.tokenizer.pad_token_id
        if pad is None:
            pad = next(iter(eos_ids))

        # vLLM uses the checkpoint's generation_config for sampling values a
        # request leaves unset.
        gen = model.generation_config
        for key in ("temperature", "top_p", "top_k"):
            value = getattr(gen, key, None)
            if value is not None:
                self._defaults[key] = value

        self._processor = processor
        self._model = model
        self._eos_ids = {int(i) for i in eos_ids if i is not None}
        self._pad_id = int(pad)
        self.device = device
        self.dtype_name = str(dtype).removeprefix("torch.")

        # Warm-up compiles kernels so the first real page is not penalised.
        warm = Image.new("RGB", (224, 224), "white")
        prompt = self._render_prompt([{"type": "image"}, {"type": "text", "text": "Text:"}], {})
        sampling = _Sampling(4, 0.0, 0.0, 1.0, True)
        self._generate_batch([_Pending(sampling, prompt, [warm], None)])  # type: ignore[arg-type]

    def unload(self) -> None:
        self._model = None
        self._processor = None
        free_accelerator_memory()

    def describe(self) -> dict[str, Any]:
        info = dict(describe_local_device(self._device_pref))
        info.update(
            {
                "backend": "transformers",
                "device": self.device or info.get("device"),
                "dtype": self.dtype_name,
                "max_batch": self.max_batch,
            }
        )
        return info

    # ------------------------------------------------------------ ChatBackend
    async def complete(self, payload: dict) -> str | None:
        if not self.loaded:
            raise LocalInferenceError("model is not loaded")
        try:
            pending = self._prepare(payload)
        except Exception:
            logger.exception("could not prepare local request")
            return None
        self._ensure_worker()
        assert self._wakeup is not None
        self._queue.append(pending)
        self._wakeup.set()
        return await pending.future

    def _prepare(self, payload: dict) -> _Pending:
        messages = payload["messages"]
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("expected a single user message")
        parts: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for part in messages[0]["content"]:
            if part.get("type") == "image_url":
                images.append(_decode_data_uri(part["image_url"]["url"]))
                parts.append({"type": "image"})
            elif part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
        prompt = self._render_prompt(parts, payload.get("chat_template_kwargs") or {})

        def pick(key: str) -> Any:
            value = payload.get(key)
            return self._defaults[key] if value is None else value

        sampling = _Sampling(
            max_tokens=int(payload.get("max_tokens") or 1024),
            presence_penalty=float(payload.get("presence_penalty") or 0.0),
            frequency_penalty=float(payload.get("frequency_penalty") or 0.0),
            repetition_penalty=float(payload.get("repetition_penalty") or 1.0),
            skip_special_tokens=bool(payload.get("skip_special_tokens", True)),
            temperature=float(pick("temperature")),
            top_p=float(pick("top_p")),
            top_k=int(pick("top_k")),
            require_complete=bool(payload.get("arena_require_complete", False)),
        )
        if sampling.greedy:
            sampling = _Sampling(
                sampling.max_tokens,
                sampling.presence_penalty,
                sampling.frequency_penalty,
                sampling.repetition_penalty,
                sampling.skip_special_tokens,
                require_complete=sampling.require_complete,
            )
        future = asyncio.get_running_loop().create_future()
        return _Pending(sampling, prompt, images, future)

    def _render_prompt(self, parts: list[dict[str, Any]], template_kwargs: dict[str, Any]) -> str:
        return self._processor.apply_chat_template(  # type: ignore[union-attr]
            [{"role": "user", "content": parts}],
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )

    def _ensure_worker(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._worker is None or self._worker.done():
            self._loop = loop
            self._wakeup = asyncio.Event()
            self._queue = deque(p for p in self._queue if not p.future.done())
            self._worker = loop.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        assert self._wakeup is not None
        while True:
            while not self._queue:
                self._wakeup.clear()
                await self._wakeup.wait()
            # Requests of one recognition wave are issued together; yield once
            # so the whole wave is queued before a batch is cut.
            await asyncio.sleep(0.005)
            head = self._queue[0].sampling
            batch = [p for p in self._queue if p.sampling == head][: self.max_batch]
            for p in batch:
                self._queue.remove(p)
            results = await asyncio.to_thread(self._generate_safely, batch)
            for p, result in zip(batch, results, strict=True):
                if not p.future.done():
                    p.future.set_result(result)

    # ------------------------------------------------------------ generation
    def _generate_safely(self, batch: list[_Pending]) -> list[str | None]:
        try:
            return self._generate_batch(batch)
        except Exception:
            if len(batch) == 1:
                logger.exception("local generation failed")
                return [None]
            logger.warning("batched generation failed; retrying one by one", exc_info=True)
            return [self._generate_safely([p])[0] for p in batch]

    def _generate_batch(self, batch: list[_Pending]) -> list[str | None]:
        import torch
        from transformers import GenerationConfig, LogitsProcessorList

        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise LocalInferenceError("model is not loaded")
        sampling = batch[0].sampling
        images = [image for p in batch for image in p.images]
        with self._gpu_lock, torch.inference_mode():
            inputs = processor(
                text=[p.prompt for p in batch],
                images=images or None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            longest = int(inputs["attention_mask"].sum(dim=1).max())
            max_new = min(sampling.max_tokens, self.max_model_len - longest)
            if max_new <= 0:
                return [None] * len(batch)

            processors = LogitsProcessorList()
            if sampling.presence_penalty or sampling.frequency_penalty:
                processors.append(
                    _OutputTokenPenalties(
                        prompt_len, sampling.presence_penalty, sampling.frequency_penalty
                    )
                )
            options: dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": False,
                "repetition_penalty": sampling.repetition_penalty,
                "eos_token_id": sorted(self._eos_ids),
                "pad_token_id": self._pad_id,
                "use_cache": True,
            }
            if not sampling.greedy:
                options.update(
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    # vLLM's -1 (disabled) is 0 in transformers.
                    top_k=max(0, sampling.top_k),
                )
            config = GenerationConfig(**options)
            started = time.perf_counter()
            output = model.generate(**inputs, generation_config=config, logits_processor=processors)
            logger.debug(
                "batch=%d max_tokens=%d prompt=%d generated=%d in %.2fs",
                len(batch),
                sampling.max_tokens,
                prompt_len,
                output.shape[1] - prompt_len,
                time.perf_counter() - started,
            )

        tokenizer = processor.tokenizer
        results: list[str | None] = []
        for row in output[:, prompt_len:].tolist():
            cut = next((i for i, tok in enumerate(row) if tok in self._eos_ids), len(row))
            if sampling.require_complete and cut == len(row):
                results.append(None)
                continue
            results.append(
                tokenizer.decode(
                    row[:cut],
                    skip_special_tokens=sampling.skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
            )
        return results


def patch_embed_convs_as_matmul(model) -> int:
    """Swap stride == kernel Conv3d layers (the ViT patch embedding) for a matmul.

    The result is identical, but MIOpen on ROCm autotunes every new input
    shape, which costs tens of seconds per crop because crop sizes vary.
    """
    from torch import nn

    class PatchMatmul(nn.Module):
        def __init__(self, conv: nn.Conv3d):
            super().__init__()
            self.weight = conv.weight
            self.bias = conv.bias
            self.kernel_size = tuple(conv.kernel_size)
            self.stride = tuple(conv.stride)

        def forward(self, x):
            if tuple(x.shape[2:]) != self.kernel_size:
                return nn.functional.conv3d(x, self.weight, self.bias, stride=self.stride)
            n, out = x.shape[0], self.weight.shape[0]
            y = nn.functional.linear(x.reshape(n, -1), self.weight.view(out, -1), self.bias)
            return y.view(n, out, 1, 1, 1)

    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if (
                isinstance(child, nn.Conv3d)
                and tuple(child.kernel_size) == tuple(child.stride)
                and all(p == 0 for p in child.padding)
                and child.groups == 1
                and tuple(child.dilation) == (1, 1, 1)
            ):
                setattr(parent, name, PatchMatmul(child))
                replaced += 1
    return replaced


class _OutputTokenPenalties:
    """vLLM's presence/frequency penalties: computed over generated tokens only."""

    def __init__(self, prompt_len: int, presence: float, frequency: float):
        self.prompt_len = prompt_len
        self.presence = presence
        self.frequency = frequency

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids[:, self.prompt_len :]
        if generated.shape[1] == 0:
            return scores
        counts = torch.zeros_like(scores).scatter_add_(
            1, generated, torch.ones_like(generated, dtype=scores.dtype)
        )
        return scores - self.frequency * counts - self.presence * (counts > 0).to(scores.dtype)


class LocalModelPool:
    """Keeps loaded checkpoints around between experiments, up to ``max_loaded``."""

    def __init__(self, *, keep_loaded: bool, max_loaded: int):
        self._keep_loaded = keep_loaded
        self._max_loaded = max(1, max_loaded)
        self._models: OrderedDict[Hashable, LocalModel] = OrderedDict()
        self._lock = asyncio.Lock()

    def is_loaded(self, key: Hashable) -> bool:
        model = self._models.get(key)
        return bool(model and model.loaded)

    async def acquire(
        self, key: Hashable, build: Callable[[], LocalModel]
    ) -> tuple[LocalModel, float]:
        """Return the loaded model and the seconds spent loading it (0 if cached)."""
        async with self._lock:
            model = self._models.get(key)
            if model is not None and model.loaded:
                self._models.move_to_end(key)
                return model, 0.0
            self._models.pop(key, None)
            keep = self._max_loaded - 1 if self._keep_loaded else 0
            while len(self._models) > keep:
                _, old = self._models.popitem(last=False)
                await asyncio.to_thread(old.unload)
            model = build()
            started = time.perf_counter()
            try:
                await asyncio.to_thread(model.load)
            except BaseException:
                await asyncio.to_thread(model.unload)
                raise
            self._models[key] = model
            return model, time.perf_counter() - started

    async def release(self, key: Hashable) -> None:
        if self._keep_loaded:
            return
        async with self._lock:
            model = self._models.pop(key, None)
            if model is not None:
                await asyncio.to_thread(model.unload)

    async def close(self) -> None:
        """Release model memory and owned helper processes when the API stops."""
        async with self._lock:
            while self._models:
                _, model = self._models.popitem(last=False)
                await asyncio.to_thread(model.unload)
