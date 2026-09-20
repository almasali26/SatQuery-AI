import re

from models.image_analysis import AnalysisResult, DetectedRegion


SUPPORTED_FEATURES = {
    "building": ("building", "buildings"),
    "road": ("road", "roads", "street", "streets", "highway", "highways"),
    "water": ("water", "river", "rivers", "lake", "lakes", "sea"),
    "vegetation": ("vegetation", "tree", "trees", "grass", "forest", "forests"),
}

ANSWER_EXPLANATION = (
    "The answer is based on pretrained SegFormer semantic-segmentation results "
    "and image coordinates. It is not a satellite-specific visual question-answering model."
)


def _tokens(question: str) -> set[str]:
    return set(re.findall(r"[a-z]+", question.lower()))


def _requested_feature(question: str) -> str | None:
    question_tokens = _tokens(question)
    for feature, aliases in SUPPORTED_FEATURES.items():
        if question_tokens.intersection(aliases):
            return feature
    return None


def _region_matches(feature: str, label: str) -> bool:
    label_tokens = _tokens(label)
    aliases = set(SUPPORTED_FEATURES[feature])
    if feature == "vegetation":
        aliases.update({"plant", "palm", "field", "earth"})
    return bool(label_tokens.intersection(aliases))


def _matching_regions(feature: str, regions: list[DetectedRegion]) -> list[DetectedRegion]:
    return [region for region in regions if _region_matches(feature, region.feature)]


def _intent(question: str) -> str | None:
    question_tokens = _tokens(question)
    question_text = question.lower()
    if question_tokens.intersection({"dominant", "main"}) or "most area" in question_text:
        return "dominant"
    if {"how", "many"}.issubset(question_tokens) or "count" in question_tokens:
        return "count"
    if question_tokens.intersection({"percentage", "percent", "area", "much"}):
        return "area"
    if question_tokens.intersection({"where", "location", "show"}):
        return "location"
    if question_tokens.intersection({"objects", "features", "identify"}):
        return "list"
    if "visible" in question_tokens and "what" in question_tokens:
        return "list"
    if question_tokens.intersection({"are", "is", "any", "find", "visible", "detect", "detected"}):
        return "presence"
    return None


def _confidence(regions: list[DetectedRegion]) -> float:
    if not regions:
        return 0
    return round(sum(region.confidence for region in regions) / len(regions), 2)


def _unsupported(regions: list[DetectedRegion]) -> AnalysisResult:
    return AnalysisResult(
        answer="Unsupported question.",
        confidence=0,
        explanation=(
            "This prototype supports presence, location, area, count, list, and dominant-feature "
            "questions about buildings, roads, water, and vegetation. "
            f"{ANSWER_EXPLANATION}"
        ),
        detected_regions=regions,
        supported_question=False,
    )


def answer_question(question: str, regions: list[DetectedRegion]) -> AnalysisResult:
    intent = _intent(question)
    requested_feature = _requested_feature(question)

    if intent == "list":
        detected = [
            feature
            for feature in SUPPORTED_FEATURES
            if _matching_regions(feature, regions)
        ]
        answer = ", ".join(detected) if detected else "No supported features were detected."
        return AnalysisResult(
            answer=f"Detected feature categories: {answer}",
            confidence=_confidence(regions),
            explanation=ANSWER_EXPLANATION,
            detected_regions=regions,
            supported_question=True,
        )

    if intent == "dominant":
        feature_areas = {
            feature: sum(region.area_percent for region in _matching_regions(feature, regions))
            for feature in SUPPORTED_FEATURES
        }
        detected_areas = {feature: area for feature, area in feature_areas.items() if area > 0}
        if not detected_areas:
            answer = "No supported feature regions were detected."
            confidence = 0
        else:
            dominant_feature = max(detected_areas, key=detected_areas.get)
            answer = (
                f"The dominant detected feature is {dominant_feature}, "
                f"covering approximately {detected_areas[dominant_feature]:.2f}% of the image."
            )
            confidence = _confidence(_matching_regions(dominant_feature, regions))
        return AnalysisResult(
            answer=answer,
            confidence=confidence,
            explanation=ANSWER_EXPLANATION,
            detected_regions=regions,
            supported_question=True,
        )

    if intent is None or requested_feature is None:
        return _unsupported(regions)

    matching = _matching_regions(requested_feature, regions)
    if not matching:
        return AnalysisResult(
            answer=f"No {requested_feature} regions were detected in the image.",
            confidence=0,
            explanation=ANSWER_EXPLANATION,
            detected_regions=regions,
            supported_question=True,
        )

    if intent == "presence":
        answer = f"Yes, {len(matching)} {requested_feature} region(s) were detected."
    elif intent == "count":
        answer = f"{len(matching)} {requested_feature} region(s) were detected."
    elif intent == "area":
        total_area = sum(region.area_percent for region in matching)
        answer = f"{requested_feature.capitalize()} covers approximately {total_area:.2f}% of the image."
    elif intent == "location":
        boxes = []
        for region in matching:
            box = region.bounding_box
            boxes.append(
                f"x={box['x_min']}, y={box['y_min']} to "
                f"x={box['x_max']}, y={box['y_max']}"
            )
        answer = f"{requested_feature.capitalize()} regions were detected around " + "; ".join(boxes) + "."
    else:
        return _unsupported(regions)

    return AnalysisResult(
        answer=answer,
        confidence=_confidence(matching),
        explanation=ANSWER_EXPLANATION,
        detected_regions=regions,
        supported_question=True,
    )
