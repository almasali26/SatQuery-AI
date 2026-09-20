from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DetectedRegion:
    feature: str
    pixel_area: int
    area_percent: float
    confidence: float
    bounding_box: dict[str, int]


@dataclass
class AnalysisResult:
    answer: str
    confidence: float
    explanation: str
    detected_regions: list[DetectedRegion]
    supported_question: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["detected_regions"] = [asdict(region) for region in self.detected_regions]
        return result
