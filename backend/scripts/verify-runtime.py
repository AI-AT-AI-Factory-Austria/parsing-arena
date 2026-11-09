"""Check the deployment and optionally parse one synthetic page with each model.

No gallery documents, database changes, LlamaParse, Mistral or remote endpoints.
Run with the backend venv, from any directory. --smoke downloads missing weights.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.catalog import ALL_MODELS
from app.config import BACKEND_DIR, Settings
from app.documents import render_page
from app.engines.base import PageSource
from app.engines.open_engine import settings_prefix
from app.engines.registry import EngineRegistry
from app.runtime import profile_error, runtime_profile

MODELS = [
    "lightonocr-3-0.8b",
    "infinity-parser2-flash",
    "ovisocr3",
]


async def verify(args) -> int:
    settings = Settings(
        local_inference=True,
        keep_models_loaded=False,
        **{f"{settings_prefix(m)}_endpoint_url": "" for m in args.models},
    )
    registry = EngineRegistry(settings)
    print(json.dumps({"profile": runtime_profile(settings), "device": registry.local_device()}))
    if error := profile_error(settings):
        print(error, file=sys.stderr)
        return 1
    failures = []
    output_dir = settings.data_dir / "runtime-checks" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    results = []
    try:
        for model_id in args.models:
            engine = registry.open_engine(model_id)
            reason = engine.unavailable_reason()
            if reason:
                print(f"{model_id}: unavailable: {reason}", flush=True)
                failures.append(model_id)
                continue
            print(
                f"{model_id}: configured ({getattr(engine, 'runtime_label', None) or 'native'})",
                flush=True,
            )
            if not args.smoke:
                continue
            model_dir = output_dir / model_id
            model_dir.mkdir(parents=True)
            source = BACKEND_DIR / "samples" / "sample-report.pdf"
            image_path = model_dir / "page.png"
            render_page(
                source, ".pdf", 1, dpi=engine.render.dpi, max_pixels=engine.render.max_pixels
            ).save(image_path)

            async def report(progress, model_id=model_id):
                print(f"{model_id}: {progress}", flush=True)

            try:
                backend, load_s = await engine.prepare()
                result = await engine.parse(
                    backend,
                    PageSource([1], {1: image_path}, source, ".pdf"),
                    report=report,
                    artifact_dir=model_dir,
                )
                (model_dir / "output.md").write_text(result.markdown, encoding="utf-8")
                if "northwind" not in result.markdown.lower() or not result.pages[0].elements:
                    raise RuntimeError("Test page header or layout elements are missing.")
                info = {
                    "model": model_id,
                    "load_s": load_s,
                    "parse_s": result.pages[0].seconds,
                    "characters": len(result.markdown),
                    "runtime": engine.describe(backend),
                }
                results.append(info)
                print(json.dumps(info), flush=True)
            except Exception as exc:  # noqa: BLE001 -- continue with independent models
                failures.append(model_id)
                results.append({"model": model_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{model_id}: FAILED: {exc}", file=sys.stderr, flush=True)
            finally:
                await engine.release()
    finally:
        await registry.pool.close()
    if args.smoke:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run real local inference")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    options = parser.parse_args()
    assert all(ALL_MODELS[m].implemented for m in options.models)
    raise SystemExit(asyncio.run(verify(options)))
