from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    SegformerImageProcessor,
)

from models.image_analysis import DetectedRegion


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
CANDIDATE_MODEL_NAME = "Pranilllllll/segformer-satellite-segementation"

_processor = None
_model = None
_candidate_processor = None
_candidate_model = None
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
SUPPORTED_OVERLAY_CLASSES = {
    "building": (255, 60, 40),
    "road": (255, 210, 40),
    "tree": (40, 200, 90),
    "vegetation": (40, 200, 90),
    "water": (40, 130, 255),
    "river": (40, 130, 255),
    "lake": (40, 130, 255),
    "sea": (40, 130, 255),
}


def _load_model() -> tuple[Any, Any]:
    global _processor, _model

    if _processor is None or _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
        _model.to("cpu")
        _model.eval()

    return _processor, _model


def analyze_image(image_bytes: bytes) -> list[DetectedRegion]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        mask = probabilities.argmax(dim=0)

    total_pixels = width * height
    regions = []
    labels = model.config.id2label

    for class_id in torch.unique(mask).tolist():
        class_mask = mask == class_id
        pixel_area = int(class_mask.sum().item())
        if pixel_area == 0:
            continue

        coordinates = torch.nonzero(class_mask, as_tuple=False)
        y_min, x_min = coordinates.min(dim=0).values.tolist()
        y_max, x_max = coordinates.max(dim=0).values.tolist()
        class_confidence = float(probabilities[class_id][class_mask].mean().item())

        regions.append(
            DetectedRegion(
                feature=labels.get(class_id, f"class_{class_id}").lower(),
                pixel_area=pixel_area,
                area_percent=round(pixel_area / total_pixels * 100, 2),
                confidence=round(class_confidence * 100, 2),
                bounding_box={
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                },
            )
        )

    return sorted(regions, key=lambda region: region.pixel_area, reverse=True)


def generate_segmentation_overlay(image_bytes: bytes) -> str:
    """Save a pixel-level ADE20K overlay and return its backend-relative path."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        mask = probabilities.argmax(dim=0)

    labels = model.config.id2label
    base_image = image.convert("RGBA")
    dimmed_image = Image.blend(base_image, Image.new("RGBA", image.size, (0, 0, 0, 255)), 0.35)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    highlighted_percentages = {}
    total_pixels = width * height

    for class_id in torch.unique(mask).tolist():
        label = labels.get(class_id, f"class_{class_id}").lower()
        color = next(
            (color for name, color in SUPPORTED_OVERLAY_CLASSES.items() if name in label),
            None,
        )
        if color is None:
            continue

        class_mask = mask == class_id
        pixel_count = int(class_mask.sum().item())
        highlighted_percentages[label] = round(pixel_count / total_pixels * 100, 2)
        coordinates = torch.nonzero(class_mask, as_tuple=False).tolist()
        for y, x in coordinates:
            overlay_pixels[x, y] = (*color, 135)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"segmentation_overlay_{uuid4().hex}.png"
    output_path = RESULTS_DIR / filename
    Image.alpha_composite(dimmed_image, overlay).save(output_path, format="PNG")

    print(f"SEGMENTATION OVERLAY: {output_path}")
    print(f"IMAGE DIMENSIONS: {width}x{height}")
    print(f"HIGHLIGHTED CLASSES: {sorted(highlighted_percentages)}")
    for label, percentage in highlighted_percentages.items():
        print(f"- {label}: {percentage}% of pixels highlighted")

    return f"/results/{filename}"


def _load_candidate_model() -> tuple[Any, Any]:
    global _candidate_processor, _candidate_model

    if _candidate_processor is None or _candidate_model is None:
        try:
            _candidate_processor = AutoImageProcessor.from_pretrained(CANDIDATE_MODEL_NAME)
        except ValueError:
            _candidate_processor = SegformerImageProcessor.from_pretrained(CANDIDATE_MODEL_NAME)
        _candidate_model = AutoModelForSemanticSegmentation.from_pretrained(CANDIDATE_MODEL_NAME)
        _candidate_model.to("cpu")
        _candidate_model.eval()

    return _candidate_processor, _candidate_model


def compare_candidate_model(image_bytes: bytes) -> list[DetectedRegion]:
    """Run the candidate model for comparison without changing the default model."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    processor, model = _load_candidate_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        mask = probabilities.argmax(dim=0)

    total_pixels = width * height
    labels = model.config.id2label
    regions = []

    for class_id in torch.unique(mask).tolist():
        class_mask = mask == class_id
        pixel_area = int(class_mask.sum().item())
        if pixel_area == 0:
            continue

        coordinates = torch.nonzero(class_mask, as_tuple=False)
        y_min, x_min = coordinates.min(dim=0).values.tolist()
        y_max, x_max = coordinates.max(dim=0).values.tolist()
        confidence = float(probabilities[class_id][class_mask].mean().item())
        feature = labels.get(class_id, f"class_{class_id}").lower()

        regions.append(
            DetectedRegion(
                feature=feature,
                pixel_area=pixel_area,
                area_percent=round(pixel_area / total_pixels * 100, 2),
                confidence=round(confidence * 100, 2),
                bounding_box={
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                },
            )
        )

    regions = sorted(regions, key=lambda region: region.pixel_area, reverse=True)
    print(f"CANDIDATE MODEL: {CANDIDATE_MODEL_NAME}")
    for region in regions:
        print(
            f"- {region.feature}: {region.area_percent}% of image, "
            f"confidence {region.confidence}%, bounding box {region.bounding_box}"
        )
    return regions
