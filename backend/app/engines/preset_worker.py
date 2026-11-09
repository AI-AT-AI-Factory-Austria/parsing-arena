"""JSON-line worker, also executable from Marker's isolated Python environment."""

import json
import os
import re
import subprocess
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote

if __package__:
    from .presets import HYBRID_ARGS, MARKER_CONFIGS, SURYA_CHECKPOINT, odl_options
else:
    from presets import HYBRID_ARGS, MARKER_CONFIGS, SURYA_CHECKPOINT, odl_options


def artifact_links(markdown, folder, root):
    """Make library-generated image links work with the arena's artifact endpoint."""

    def replace(match):
        url = unquote(match[2] or match[3]).replace("\\", "/")
        path = (folder / url).resolve()
        if path.is_relative_to(root.resolve()) and path.is_file():
            return match[1] + "<" + path.relative_to(root.resolve()).as_posix() + ">" + match[4]
        return match[0]

    return re.sub(r"(!\[[^\]]*\]\()(?:<([^>]+)>|([^\s)]+))(\))", replace, markdown)


def odl_elements(data, page, width, height):
    labels = {
        "heading": "Section-header",
        "paragraph": "Text",
        "table": "Table",
        "image": "Picture",
        "figure": "Picture",
        "list": "List-item",
    }
    elements = []

    def visit(item):
        box = item.get("bounding box")
        if box and item.get("page number") == page:
            left, bottom, right, top = box
            elements.append(
                {
                    "category": labels.get(item.get("type"), "Text"),
                    "bbox": [
                        max(0, min(1, v))
                        for v in (
                            left / width,
                            1 - top / height,
                            right / width,
                            1 - bottom / height,
                        )
                    ],
                    "content": item.get("content", ""),
                    "layout_order": len(elements),
                }
            )
        else:
            for child in item.get("kids", []):
                visit(child)

    visit(data)
    return elements


def stop_windows_process(pid, backend):
    import psutil

    try:
        process = psutil.Process(pid)
        owned = [*reversed(process.children(recursive=True)), process]
    except psutil.NoSuchProcess:
        return
    for child in owned:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(owned, timeout=10)


def configure_surya_cleanup():
    if os.name == "nt":
        # On Windows os.kill(pid, 0) sends CTRL_C_EVENT, unlike POSIX's liveness probe.
        from surya.inference.backends import spawn

        spawn._stop_process = stop_windows_process


class Marker:
    def __init__(self, preset):
        from marker.models import create_model_dict
        from PIL import Image

        configure_surya_cleanup()
        self.preset = preset
        self.config = MARKER_CONFIGS[preset]
        self.models = create_model_dict(inference_backend="llamacpp")
        # Loading belongs to model-load time, not the first page's extraction time.
        if preset == "marker_balanced":
            self.models["inference_manager"].start()
            self.models["ocr_error_model"](["A document with readable embedded text."])
        else:
            self.models["fast_layout_model"]([Image.new("RGB", (640, 800), "white")])

    def describe(self):
        return {
            "backend": "marker-pdf",
            "preset": self.preset,
            "version": version("marker-pdf"),
            "surya_version": version("surya-ocr"),
            "device": "cpu" if self.preset == "marker_fast" else "llama.cpp",
            "surya_checkpoint": SURYA_CHECKPOINT if self.preset == "marker_balanced" else None,
            "inference_backend": "llamacpp" if self.preset == "marker_balanced" else None,
            "gguf_model": "datalab-to/surya-ocr-2.5-gguf/surya-2.5.gguf"
            if self.preset == "marker_balanced"
            else None,
            "output_format": "Markdown + layout boxes",
            **self.config,
        }

    def convert(self, source, page, output):
        from marker.converters.pdf import PdfConverter

        converter = PdfConverter(
            artifact_dict=self.models, config={**self.config, "page_range": [page - 1]}
        )
        document = converter.build_document(source)
        rendered = converter.resolve_dependencies(converter.renderer)(document)
        markdown = rendered.markdown
        images_dir = output / "artifacts"
        images_dir.mkdir(exist_ok=True)
        for name, image in rendered.images.items():
            filename = f"page-{page}-{Path(name).name}"
            image.save(images_dir / filename)
            markdown = markdown.replace(name, f"artifacts/{filename}")
        elements = []
        labels = {
            "SectionHeader": "Section-header",
            "ListItem": "List-item",
            "PageHeader": "Page-header",
            "PageFooter": "Page-footer",
        }
        for group in document.pages:
            width, height = group.polygon.width, group.polygon.height
            for block_id in group.structure or []:
                block = document.get_block(block_id)
                x1, y1, x2, y2 = block.polygon.bbox
                category = block.block_type.name
                elements.append(
                    {
                        "category": labels.get(category, category),
                        "bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
                        "content": block.raw_text(document),
                        "layout_order": len(elements),
                    }
                )
        return {"markdown": markdown, "elements": elements}

    def close(self):
        from marker.models import shutdown_models

        shutdown_models(self.models)


class OpenDataLoader:
    def __init__(self, preset):
        self.preset = preset
        self.options = odl_options(preset)
        self.server = None
        if preset == "opendataloader_docling":
            from opendataloader_pdf.hybrid_launcher import HybridServer

            self.server = HybridServer(args=HYBRID_ARGS, startup_timeout=900).start()

    def describe(self):
        info = {
            "backend": "opendataloader-pdf",
            "preset": self.preset,
            "version": version("opendataloader-pdf"),
            "device": "cpu",
            "output_format": "Markdown + JSON",
            **self.options,
        }
        if self.server:
            info.update(
                ocr_engine="easyocr",
                ocr_languages=["de", "en"],
                force_ocr=False,
                do_ocr=True,
                do_table_structure=True,
                do_formula_enrichment=False,
                do_picture_description=False,
            )
        return info

    def convert(self, source, page, output):
        from opendataloader_pdf import convert

        folder = output / "artifacts" / f"page-{page}"
        folder.mkdir(parents=True, exist_ok=True)
        options = {**self.options}
        if self.server:
            options.update(
                hybrid_url=self.server.url, hybrid_timeout="600000", hybrid_triage_log=True
            )
        convert(
            input_path=source,
            output_dir=str(folder),
            format="markdown,json",
            pages=str(page),
            image_output="external",
            timeout=900,
            **options,
        )
        path = folder / (Path(source).stem + ".md")
        if not path.exists():
            raise RuntimeError("OpenDataLoader did not produce Markdown output.")
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(source) as pdf:
            width, height = pdf[page - 1].get_size()
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return {
            "markdown": artifact_links(path.read_text(encoding="utf-8"), folder, output),
            "elements": odl_elements(data, page, width, height),
        }

    def close(self):
        if self.server:
            self.server.stop()


def main():
    if os.name == "nt":
        # The upstream libraries start their own local servers/JVM. Keep those
        # children hidden as well as this worker, without modifying dependencies.
        original_popen = subprocess.Popen

        class HiddenProcess(original_popen):
            def __init__(self, *args, **kwargs):
                kwargs["creationflags"] = (
                    kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
                )
                super().__init__(*args, **kwargs)

        subprocess.Popen = HiddenProcess
    # Library progress/logging goes to stderr; stdout is reserved for the protocol.
    response_stream, sys.stdout = sys.stdout, sys.stderr

    def respond(value):
        response_stream.write("ARENA_RESULT:" + json.dumps(value, ensure_ascii=True) + "\n")
        response_stream.flush()

    runtime = None
    try:
        preset = sys.argv[1]
        runtime = Marker(preset) if preset in MARKER_CONFIGS else OpenDataLoader(preset)
        respond(runtime.describe())
        for line in sys.stdin:
            try:
                request = json.loads(line)
                respond(
                    runtime.convert(request["source"], request["page"], Path(request["output"]))
                )
            except Exception as exc:  # noqa: BLE001 - return library errors across the RPC boundary
                traceback.print_exc()
                respond({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001 - return startup errors across the RPC boundary
        traceback.print_exc()
        respond({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    main()
