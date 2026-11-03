"""Measure prefill and decode speed of the local backend for a few image sizes.

uv run python scripts/bench_local.py [--attn sdpa|eager] [--aotriton]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aotriton", action="store_true")
    parser.add_argument("--attn", default="sdpa")
    args = parser.parse_args()
    if args.aotriton:
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

    from app.engines.backends.transformers_local import _patch_embed_convs_as_matmul

    repo = "cloudraker/rakedoc-nano"
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    _patch_embed_convs_as_matmul(model)
    model.to("cuda").eval()

    def run(size: tuple[int, int], new_tokens: int) -> tuple[float, int]:
        image = Image.new("RGB", size, "white")
        content = [{"type": "image"}, {"type": "text", "text": "\nText Recognition:\n"}]
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        config = GenerationConfig(
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=151643,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, generation_config=config)
        torch.cuda.synchronize()
        return time.perf_counter() - started, inputs["input_ids"].shape[1]

    run((448, 448), 2)
    for size in [(448, 448), (1036, 1036), (1400, 1100)]:
        ttft, prompt_len = run(size, 1)
        total, _ = run(size, 65)
        per_token = (total - ttft) / 64
        print(
            f"{size}: prompt={prompt_len} ttft={ttft:.2f}s decode={per_token * 1000:.0f}ms/token",
            flush=True,
        )


if __name__ == "__main__":
    main()
