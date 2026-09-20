from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Large satellite images can be extremely expensive for a VLM.
# We resize only the VLM input; original files remain unchanged.
MAX_IMAGE_SIZE = 768


@dataclass
class VQAResult:
    answer: str
    confidence: float
    model: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "model": self.model,
            "evidence": self.evidence,
        }


_processor = None
_model = None


def _load_model():
    global _processor, _model

    if _processor is None or _model is None:
        print("Loading SmolVLM...")
        print("Model:", MODEL_NAME)

        _processor = AutoProcessor.from_pretrained(MODEL_NAME)

        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
        )

        _model.eval()

        print("SmolVLM loaded successfully")

    return _processor, _model


def _prepare_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("No image was provided.")

    from io import BytesIO

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Keep aspect ratio while limiting the largest dimension.
    image.thumbnail(
        (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    return image


def run_vqa(
    image_bytes: bytes,
    question: str,
) -> VQAResult:

    if not question or not question.strip():
        raise ValueError("No question was provided.")

    processor, model = _load_model()

    image = _prepare_image(image_bytes)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": question.strip(),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=40,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # Extract only the assistant response.
    if "Assistant:" in generated_text:
        answer = generated_text.split(
            "Assistant:",
            1,
        )[1].strip()
    else:
        answer = generated_text.strip()

    if not answer:
        answer = "The model could not generate an answer."

    return VQAResult(
        answer=answer,
        confidence=0.0,
        model=MODEL_NAME,
        evidence={
            "image_size_used": [
                image.width,
                image.height,
            ],
            "status": "vqa_inference_completed",
            "note": (
                "SmolVLM is used as a lightweight VQA baseline. "
                "It is not itself a remote-sensing-specific fine-tuned model."
            ),
        },
    )