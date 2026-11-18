"""One hardware label for the landing table and picker; format is secondary detail."""

from .backends.transformers_local import describe_local_device


def execution_info(model_id, engine, settings, runnable):
    backend = engine.backend_kind
    if backend == "endpoint":
        return "Remote model server", "OpenAI-compatible API; runs outside this installation."
    detail = getattr(engine, "runtime_label", None)
    cpu = backend == "cpu" or settings.local_device == "cpu"
    if cpu:
        return "Local CPU", detail or {
            "pypdf": "Embedded PDF text; no neural model.",
            "tesseract": "Tesseract OCR on CPU.",
            "docling": "Docling layout and table models on CPU.",
            "marker_fast": "PyTorch auxiliary models on CPU; no OCR.",
            "opendataloader_core": "Java PDF extraction; no neural model.",
            "opendataloader_docling": "Java + Docling / EasyOCR on CPU.",
        }.get(model_id, "PyTorch on CPU.")
    if not runnable:
        return None, detail
    if getattr(engine, "uses_vulkan", False):
        return "Local AMD GPU · Vulkan", "llama.cpp · F16 GGUF weights."
    if model_id == "unlimited-ocr-2":
        from .unlimited import installed_runtime

        installed = installed_runtime(settings)
        runtime = installed[1]["backend"] if installed else "GPU"
        vendor = "AMD" if runtime == "Vulkan" else "NVIDIA"
        return f"Local {vendor} GPU · {runtime}", "Patched llama.cpp · BF16 GGUF; multi-page."
    device = describe_local_device(settings.local_device)
    if not device.get("available"):
        return "Local runtime", detail
    if model_id == "marker_balanced":
        runtime = device.get("runtime")
        label = {
            "ROCm": "Local AMD GPU · Vulkan",
            "CUDA": "Local NVIDIA GPU · CUDA",
        }.get(runtime, "Local CPU")
        return label, "llama.cpp for Surya OCR; auxiliary PyTorch models run on CPU."
    label = {
        "ROCm": "Local AMD GPU · ROCm",
        "CUDA": "Local NVIDIA GPU · CUDA",
        "MPS": "Local Apple GPU · Metal",
        "CPU": "Local CPU",
    }.get(device.get("runtime"), "Local runtime")
    return label, f"PyTorch · {device.get('device_name', device.get('runtime', 'local'))}"
