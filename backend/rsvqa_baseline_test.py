import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


DATASET_DIR = Path(r"..\dataset\rsvqa")
IMAGE_DIR = DATASET_DIR / "Images_LR"

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def load_js_json(filename):
    path = DATASET_DIR / filename
    text = path.read_text(encoding="utf-8")

    match = re.search(
        r"JSON.parse\('(.+)'\)",
        text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not parse {filename}")

    return json.loads(
        match.group(1).replace("\\'", "'")
    )


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[.,!?]", "", text)
    return text


def extract_numbers(text):
    text = normalize_text(text)

    numbers = re.findall(r"\b\d+\b", text)

    words = re.findall(r"\b[a-z]+\b", text)

    for word in words:
        if word in NUMBER_WORDS:
            numbers.append(NUMBER_WORDS[word])

    return numbers


def evaluate_answer(question_type, model_answer, ground_truth):
    model_answer = normalize_text(model_answer)
    ground_truth = normalize_text(ground_truth)

    # -----------------------------------------------
    # Count questions
    # -----------------------------------------------

    if question_type == "count":
        model_numbers = extract_numbers(model_answer)

        return ground_truth in model_numbers

    # -----------------------------------------------
    # Other question types
    # -----------------------------------------------

    if ground_truth == model_answer:
        return True

    if ground_truth in model_answer.split():
        return True

    return False


# --------------------------------------------------
# Load RSVQA metadata
# --------------------------------------------------

print("Loading RSVQA metadata...")

questions_data = load_js_json(
    "questionsS2_reduced.js"
)

answers_data = load_js_json(
    "answersS2_reduced.js"
)

images_data = load_js_json(
    "imagesS2_reduced.js"
)

questions = questions_data["questions"]
answers = answers_data["answers"]
images = images_data["images"]


answers_by_id = {
    answer["id"]: answer["answer"]
    for answer in answers
}


images_by_id = {
    image["id"]: image
    for image in images
}


print("Questions loaded :", len(questions))
print("Images in metadata:", len(images))
print()


# --------------------------------------------------
# Select 20 QUESTIONS from 20 DIFFERENT IMAGES
# --------------------------------------------------

selected_questions = []
used_images = set()

for question in questions:

    image_id = question["img_id"]

    if image_id in used_images:
        continue

    selected_questions.append(question)
    used_images.add(image_id)

    if len(selected_questions) == 20:
        break


print(
    "Selected questions:",
    len(selected_questions),
)

print(
    "Different images:",
    len(used_images),
)

print()


# --------------------------------------------------
# Load SmolVLM
# --------------------------------------------------

print("Loading SmolVLM...")
print("Model:", MODEL_NAME)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.eval()

print("SmolVLM loaded successfully")
print()


# --------------------------------------------------
# Statistics
# --------------------------------------------------

stats = {}

total_correct = 0
total_tested = 0


# --------------------------------------------------
# Run 20-question diverse-image baseline
# --------------------------------------------------

for index, question in enumerate(
    selected_questions,
    start=1,
):

    question_type = question["type"]

    image_info = images_by_id[
        question["img_id"]
    ]

    image_id = Path(
        image_info["original_name"]
    ).stem

    image_path = (
        IMAGE_DIR / f"{image_id}.tif"
    )

    answer_id = question[
        "answers_ids"
    ][0]

    ground_truth = str(
        answers_by_id[answer_id]
    ).strip()


    if not image_path.exists():

        print(
            f"Image missing: {image_path}"
        )

        continue


    # -----------------------------------------------
    # Load image
    # -----------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # -----------------------------------------------
    # Prepare prompt
    # -----------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": question[
                        "question"
                    ],
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


    # -----------------------------------------------
    # Generate answer
    # -----------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
        )


    generated_text = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
    )


    # -----------------------------------------------
    # Extract Assistant answer
    # -----------------------------------------------

    if "Assistant:" in generated_text:

        model_answer = (
            generated_text
            .split(
                "Assistant:",
                1,
            )[1]
            .strip()
        )

    else:

        model_answer = (
            generated_text.strip()
        )


    # -----------------------------------------------
    # Evaluate
    # -----------------------------------------------

    is_correct = evaluate_answer(
        question_type,
        model_answer,
        ground_truth,
    )


    # -----------------------------------------------
    # Update statistics
    # -----------------------------------------------

    if question_type not in stats:

        stats[question_type] = {
            "correct": 0,
            "total": 0,
        }

    stats[question_type]["total"] += 1
    total_tested += 1


    if is_correct:

        stats[question_type]["correct"] += 1
        total_correct += 1


    # -----------------------------------------------
    # Print result
    # -----------------------------------------------

    print("=" * 70)

    print(
        f"Test {index}/20"
    )

    print(
        "Question ID :",
        question["id"],
    )

    print(
        "Image       :",
        image_id,
    )

    print(
        "Type        :",
        question_type,
    )

    print(
        "Question    :",
        question["question"],
    )

    print(
        "Model       :",
        model_answer,
    )

    print(
        "Ground Truth:",
        ground_truth,
    )

    print(
        "Result      :",
        "CORRECT"
        if is_correct
        else "WRONG",
    )


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print()
print("=" * 70)
print("DIVERSE-IMAGE BASELINE SUMMARY")
print("=" * 70)

print()

for question_type, result in stats.items():

    accuracy = (
        result["correct"]
        / result["total"]
        * 100
    )

    print(
        f"{question_type:15}"
        f" : {result['correct']}/"
        f"{result['total']}"
        f" ({accuracy:.2f}%)"
    )


print()
print("-" * 70)

if total_tested > 0:

    overall_accuracy = (
        total_correct
        / total_tested
        * 100
    )

    print(
        "Total tested    :",
        total_tested,
    )

    print(
        "Total correct   :",
        total_correct,
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.2f}%"
    )

print("=" * 70)