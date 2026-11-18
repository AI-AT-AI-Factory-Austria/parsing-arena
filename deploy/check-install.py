"""Check packaged dependencies and hardware without downloading model weights."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Also parse one bundled page with pypdf"
    )
    args = parser.parse_args()
    from app.catalog import MODELS
    from app.config import get_settings
    from app.engines.registry import CPU_DISABLED_MODELS, EngineRegistry

    settings = get_settings()
    registry = EngineRegistry(settings)
    failures = []
    for module in ("torch", "transformers", "docling", "doctr", "opendataloader_pdf"):
        importlib.import_module(module)
        print(f"OK: {module}")
    import torch

    device = "cpu" if settings.local_device == "cpu" else "cuda"
    tensor = torch.ones(8, 8, device=device)
    assert (tensor @ tensor).sum().item() == 512
    if device == "cuda":
        import triton
        from torch.nn.attention.flex_attention import flex_attention

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q = torch.randn(1, 1, 128, 64, device=device, dtype=dtype)
        assert torch.isfinite(torch.compile(flex_attention)(q, q, q)).all()
        print(f"OK: CUDA / Triton {triton.__version__}")
    for model in MODELS:
        availability = registry.availability(model)
        if availability["runnable"]:
            print(f"OK: {model.name}")
        elif device == "cpu" and model.id in CPU_DISABLED_MODELS:
            print(f"CPU exclusion: {model.name} (long processing times)")
        else:
            failures.append(f"{model.name}: {availability['reason']}")
    subprocess.run([settings.marker_python, "-c", "import marker, surya, torch"], check=True)
    subprocess.run([settings.llama_cpp_binary, "--version"], check=True)
    subprocess.run(["java", "-version"], check=True)
    subprocess.run(["tesseract", "--version"], check=True)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content("<table><tr><td>Parsing Arena</td></tr></table>")
        assert page.locator("td").inner_text() == "Parsing Arena"
        browser.close()
    if failures:
        raise SystemExit("\n".join(failures))
    if args.smoke:
        subprocess.run(
            [
                sys.executable,
                "scripts/smoke_model.py",
                "samples/sample-report.pdf",
                "--model",
                "pypdf",
                "--pages",
                "1",
            ],
            check=True,
        )
    print("Installation checks passed. No document-processing APIs were called.")


if __name__ == "__main__":
    main()
