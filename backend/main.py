from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.change_detector import detect_changes
from services.query_engine import answer_question
from services.satellite_analyzer import (
    analyze_image as run_image_analysis,
    generate_segmentation_overlay,
)
from services.vqa_service import run_vqa
from services.agent_controller import route_query
from services.optical_sar_service import analyze_optical_sar


RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="SatQuery AI",
    description="AI-powered satellite image analysis system",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://sat-query-ai-sepia.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to SatQuery AI!",
        "status": "Backend is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(""),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await image.read()

    try:
        agent_decision = route_query(
            question,
            image_count=1,
        )

        detected_regions = run_image_analysis(image_bytes)

        vqa_result = run_vqa(
            image_bytes,
            question,
        )

        response = vqa_result.to_dict()
        response["agent_decision"] = agent_decision.to_dict()

        response["segmentation_overlay"] = (
            generate_segmentation_overlay(image_bytes)
        )

        response["detected_regions"] = detected_regions

        return response

    except Exception as error:
        print("IMAGE ANALYSIS ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {repr(error)}",
        ) from error


@app.post("/compare")
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    question: str = Form(
        "Compare these two images and identify the changes."
    ),
):
    if (
        not image1.content_type
        or not image1.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="image1 must be an image file.",
        )

    if (
        not image2.content_type
        or not image2.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="image2 must be an image file.",
        )

    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    print("COMPARE QUESTION:", repr(question))

    try:
        agent_decision = route_query(
            question,
            image_count=2,
        )

        print(
            "AGENT DECISION:",
            agent_decision.to_dict(),
        )

    except Exception as error:
        print(
            "AGENT ROUTING ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Agent routing failed: {repr(error)}",
        ) from error

    if agent_decision.task == "optical_sar_analysis":
        try:
            print("RUNNING OPTICAL-SAR ANALYSIS")

            optical_sar_result = analyze_optical_sar(
                image1_bytes
            )

            optical_sar_result["agent_decision"] = (
                agent_decision.to_dict()
            )

            return optical_sar_result

        except FileNotFoundError as error:
            print(
                "OPTICAL-SAR FILE ERROR:",
                repr(error),
            )

            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error

        except ValueError as error:
            print(
                "OPTICAL-SAR VALUE ERROR:",
                repr(error),
            )

            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        except Exception as error:
            print(
                "OPTICAL-SAR ERROR:",
                repr(error),
            )

            raise HTTPException(
                status_code=500,
                detail=f"Optical-SAR analysis failed: {repr(error)}",
            ) from error

    try:
        result = detect_changes(
            image1_bytes,
            image2_bytes,
            image1.filename or "image1.tif",
            image2.filename or "image2.tif",
        )

        result["agent_decision"] = agent_decision.to_dict()

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(
            "CHANGE DETECTION ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Change detection failed: {repr(error)}",
        ) from error