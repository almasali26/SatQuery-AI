from dataclasses import dataclass


@dataclass
class AgentDecision:
    task: str
    reason: str
    tools: list[str]

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "reason": self.reason,
            "tools": self.tools,
        }


def route_query(question: str, image_count: int = 1) -> AgentDecision:
    """
    Lightweight agentic controller for SatQuery AI.

    It selects the analysis task from the user's query
    and the number of input images.
    """

    if not question or not question.strip():
        raise ValueError("No question was provided.")

    query = question.lower().strip()

    # Two-image analysis
    if image_count >= 2:
        change_words = [
            "change",
            "changed",
            "difference",
            "before",
            "after",
            "increase",
            "decrease",
            "compare",
        ]

        optical_sar_words = [
            "sar",
            "radar",
            "optical",
            "multispectral",
        ]

        if any(word in query for word in optical_sar_words):
            return AgentDecision(
                task="optical_sar_analysis",
                reason="The query indicates complementary analysis of optical and SAR imagery.",
                tools=["optical_sar_analysis"],
            )

        if any(word in query for word in change_words):
            return AgentDecision(
                task="change_analysis",
                reason="Two images were provided and the query asks about temporal or visual change.",
                tools=["change_detection"],
            )

        return AgentDecision(
            task="image_comparison",
            reason="Two images were provided, so the controller selected paired-image analysis.",
            tools=["image_comparison"],
        )

    # Single-image analysis
    grounding_words = [
        "where",
        "location",
        "locate",
        "region",
        "highlight",
    ]

    caption_words = [
        "describe",
        "description",
        "scene",
        "caption",
        "what is visible",
    ]

    if any(word in query for word in grounding_words):
        return AgentDecision(
            task="grounding",
            reason="The query asks about a location or region in the image.",
            tools=["segmentation"],
        )

    if any(word in query for word in caption_words):
        return AgentDecision(
            task="captioning",
            reason="The query asks for a description of the satellite scene.",
            tools=["vqa", "segmentation"],
        )

    return AgentDecision(
        task="vqa",
        reason="The query is a general question about the satellite image.",
        tools=["vqa"],
    )