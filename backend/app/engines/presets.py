"""Explicit library presets. IDs are also part of every runtime's cache key."""

MARKER_CONFIGS = {
    "marker_fast": {"mode": "fast", "disable_ocr": True, "use_llm": False},
    "marker_balanced": {
        "mode": "balanced",
        "disable_ocr": False,
        "force_ocr": False,
        "use_llm": False,
    },
}
SURYA_CHECKPOINT = "datalab-to/surya-ocr-2.5"
DOCTR_CONFIGS = {
    "doctr_compact": ("db_mobilenet_v3_large", "crnn_mobilenet_v3_small"),
    "doctr_default": ("fast_base", "crnn_vgg16_bn"),
    "doctr_large_transformer": ("db_resnet50", "vitstr_base"),
}


def doctr_options(preset: str) -> dict:
    detection, recognition = DOCTR_CONFIGS[preset]
    return dict(
        det_arch=detection,
        reco_arch=recognition,
        pretrained=True,
        detect_layout=False,
        detect_tables=False,
    )


def odl_options(preset: str) -> dict:
    if preset == "opendataloader_core":
        return {"hybrid": "off"}
    if preset != "opendataloader_docling":
        raise ValueError(f"Unknown OpenDataLoader preset: {preset}")
    return {"hybrid": "docling-fast", "hybrid_mode": "auto", "hybrid_fallback": False}


HYBRID_ARGS = [
    "--ocr-engine",
    "easyocr",
    "--ocr-lang",
    "de,en",
    "--device",
    "cpu",
    "--no-enrich-formula",
    "--no-enrich-picture-description",
]
