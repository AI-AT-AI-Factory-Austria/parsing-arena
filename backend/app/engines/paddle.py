"""PP-OCRv6 detector/recognizer pairs using PaddlePaddle's Transformers exports."""

import asyncio
import json
import math
import sys
from functools import wraps
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from .backends.transformers_local import (
    describe_local_device,
    free_accelerator_memory,
    resolve_device,
)
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .doctr import use_native_rocm_layers
from .open_engine import OpenModelEngine, parse_page_by_page

# Both halves are pinned: changing just the recognizer changes the preset too.
PADDLE_PRESETS = {
    "paddleocr_tiny": (
        "tiny",
        "07595f982703daf0d4e120a12a01da8073542f3a",
        "6f2d2d51b4b4226d7a2329a02f416f4994106f3a",
    ),
    "paddleocr_small": (
        "small",
        "eae2ee920a39fb3087637d3dbb58df1896ec1f24",
        "fe049fb103f57443fe8840c54ed06b702f3c1de5",
    ),
    "paddleocr_medium": (
        "medium",
        "4236c2b61741a259c091fd879dcc4edc339e916c",
        "024cad6a831de75c2c3c26e711ba8c4a82ccd24b",
    ),
}
RECOGNITION_BATCH = 8
TEXT_SCORE = 0.5
DETECTION = {"threshold": 0.3, "box_threshold": 0.6, "unclip_ratio": 1.5}


def use_native_large_convs(model):
    """Use PyTorch's GPU convolution for 9x9 kernels that stall Windows MIOpen."""
    import torch
    from torch import nn

    def native_forward(original):
        @wraps(original)
        def forward(*args, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return original(*args, **kwargs)

        return forward

    count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) and layer.kernel_size == (9, 9):
            layer.forward = native_forward(layer.forward)
            count += 1
    return count


def ordered_polygons(boxes):
    """PaddleOCR reading order: top-to-bottom, left-to-right within a text row."""
    ordered = sorted(boxes, key=lambda box: (box[0][1], box[0][0]))
    for i in range(len(ordered) - 1):
        for j in range(i, -1, -1):
            if (
                abs(ordered[j + 1][0][1] - ordered[j][0][1]) < 10
                and ordered[j + 1][0][0] < ordered[j][0][0]
            ):
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break
    return ordered


def text_crop(image, polygon):
    """Rectify a quadrilateral into an upright text line, preserving RGB channels."""
    import cv2
    import numpy as np

    points = np.asarray(polygon, dtype=np.float32)
    width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    width, height = max(1, int(width)), max(1, int(height))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    if height / width >= 1.5:
        crop = np.rot90(crop)
    return Image.fromarray(crop)


def text_output(lines, size):
    """Only accepted text enters Markdown/overlays; raw confidences remain in the artifact."""
    width, height = size
    texts, elements = [], []
    for line in lines:
        text, score = line["text"], line["score"]
        if not text.strip() or not math.isfinite(score) or score < TEXT_SCORE:
            continue
        xs, ys = zip(*line["polygon"], strict=True)
        box = [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height]
        texts.append(text)
        item = element("Text", box, text, len(elements))
        elements.append(item)
    return "\n\n".join(texts), elements


class PaddleRuntime:
    def __init__(self, preset: str, device: str = "auto"):
        self.preset = preset
        self.device_preference = device
        self.device = None
        self.detector = self.recognizer = None
        self.det_processor = self.rec_processor = None
        self.native_rocm_layers = {}
        tier, self.det_revision, self.rec_revision = PADDLE_PRESETS[preset]
        self.det_repo = f"PaddlePaddle/PP-OCRv6_{tier}_det_safetensors"
        self.rec_repo = f"PaddlePaddle/PP-OCRv6_{tier}_rec_safetensors"

    @property
    def loaded(self):
        return self.detector is not None and self.recognizer is not None

    def load(self):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            AutoModelForTextRecognition,
        )

        torch.set_num_threads(4)
        self.device = resolve_device(self.device_preference)
        self.det_processor = AutoImageProcessor.from_pretrained(
            self.det_repo, revision=self.det_revision
        )
        self.rec_processor = AutoImageProcessor.from_pretrained(
            self.rec_repo, revision=self.rec_revision
        )
        self.detector = (
            AutoModelForObjectDetection.from_pretrained(
                self.det_repo, revision=self.det_revision, dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        self.recognizer = (
            AutoModelForTextRecognition.from_pretrained(
                self.rec_repo, revision=self.rec_revision, dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        if sys.platform == "win32" and self.device.startswith("cuda") and torch.version.hip:
            # Same selective GPU BN workaround as docTR; convolutions remain on the GPU.
            self.native_rocm_layers = {
                "detector": use_native_rocm_layers(self.detector),
                "recognizer": use_native_rocm_layers(self.recognizer),
                "large_convs": use_native_large_convs(self.detector),
            }
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def unload(self):
        self.detector = self.recognizer = None
        self.det_processor = self.rec_processor = None
        self.device = None
        free_accelerator_memory()

    def describe(self):
        return {
            **describe_local_device(self.device_preference),
            "backend": "transformers",
            "transformers_version": version("transformers"),
            "device": self.device,
            "dtype": "float32",
            "preset": self.preset,
            "detector": self.det_repo,
            "detector_revision": self.det_revision,
            "recognizer": self.rec_repo,
            "recognizer_revision": self.rec_revision,
            "render_dpi": 200,
            "recognition_batch_size": RECOGNITION_BATCH,
            "text_score_threshold": TEXT_SCORE,
            "detection": DETECTION,
            "orientation_classification": False,
            "unwarping": False,
            "table_recognition": False,
            "output_format": "text lines + boxes",
            "native_rocm_layers": self.native_rocm_layers,
        }

    def extract(self, image):
        import numpy as np
        import torch

        with torch.inference_mode():
            inputs = self.det_processor(images=image, return_tensors="pt").to(self.device)
            prediction = self.detector(pixel_values=inputs["pixel_values"])
            detected = self.det_processor.post_process_object_detection(
                prediction, target_sizes=inputs["target_sizes"], **DETECTION
            )[0]
            polygons = ordered_polygons(detected["boxes"].cpu().tolist())
            if not polygons:
                return []
            pixels = np.asarray(image.convert("RGB"))
            crops = [text_crop(pixels, polygon) for polygon in polygons]
            # Similar widths share a batch, avoiding huge amounts of recognizer padding.
            order = sorted(range(len(crops)), key=lambda i: crops[i].width / crops[i].height)
            lines = [None] * len(crops)
            for start in range(0, len(order), RECOGNITION_BATCH):
                indices = order[start : start + RECOGNITION_BATCH]
                inputs = self.rec_processor(
                    images=[crops[i] for i in indices], return_tensors="pt"
                ).to(self.device)
                predictions = self.recognizer(**inputs)
                results = self.rec_processor.post_process_text_recognition(predictions)
                for i, result in zip(indices, results, strict=True):
                    score = float(result["score"])
                    lines[i] = {
                        "text": result["text"],
                        "score": score if math.isfinite(score) else 0.0,
                        "polygon": polygons[i],
                    }
            return lines


class PaddleOcrEngine(OpenModelEngine):
    render = RenderSpec(dpi=200)
    supports_endpoint = False

    @property
    def pool_key(self):
        return type(self).__name__, PADDLE_PRESETS[self.model.id], self.settings.local_device

    def build_local(self):
        return PaddleRuntime(self.model.id, self.settings.local_device)

    async def parse(
        self, backend: PaddleRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        async def parse_one(number, image):
            lines = await asyncio.to_thread(backend.extract, image)
            (artifact_dir / f"paddle-page-{number}.json").write_text(
                json.dumps({"size": image.size, "lines": lines}, ensure_ascii=False),
                encoding="utf8",
            )
            return text_output(lines, image.size)

        return await parse_page_by_page(pages, report, parse_one)
