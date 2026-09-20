import json
import re
from pathlib import Path


DATASET_DIR = Path(r"..\dataset\rsvqa")


def load_js_json(filename):
    path = DATASET_DIR / filename
    text = path.read_text(encoding="utf-8")

    match = re.search(r"JSON.parse\('(.+)'\)", text, re.DOTALL)

    if not match:
        raise ValueError(f"Could not parse {filename}")

    return json.loads(match.group(1).replace("\\'", "'"))


questions_data = load_js_json("questionsS2_reduced.js")
answers_data = load_js_json("answersS2_reduced.js")
images_data = load_js_json("imagesS2_reduced.js")

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


print("Dataset loaded successfully")
print("=" * 50)
print("Questions:", len(questions))
print("Answers:", len(answers))
print("Metadata images:", len(images))

image_folder = DATASET_DIR / "Images_LR"

missing = []

for image in images:
    image_id = Path(image["original_name"]).stem
    tif_file = image_folder / f"{image_id}.tif"

    if not tif_file.exists():
        missing.append(tif_file.name)

print("Actual TIFF files:", len(list(image_folder.glob("*.tif"))))
print("Missing metadata images:", len(missing))
print("Missing:", missing)