from pathlib import Path
import tempfile
import uuid

import rasterio

from services.spectral_change_detector import (
    process_full_scene,
    create_full_scene_preview,
)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

RESULTS_DIR = (
    BACKEND_DIR
    / "data"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATASET_DIR = PROJECT_ROOT / "dataset"


# ------------------------------------------------------------
# KNOWN SENTINEL-2 SCL FILES
# ------------------------------------------------------------

SCL_2020 = (
    DATASET_DIR
    / "sentinel2_2020"
    / "T43QCA_20200115T053151_SCL_20m.jp2"
)

SCL_2025 = (
    DATASET_DIR
    / "sentinel2_2025"
    / "T43QCA_20250118T053141_SCL_20m.jp2"
)


# ------------------------------------------------------------
# DETECT YEAR FROM FILE NAME
# ------------------------------------------------------------

def detect_year(filename):
    name = Path(filename).name.lower()

    if "2020" in name:
        return "2020"

    if "2025" in name:
        return "2025"

    return None


# ------------------------------------------------------------
# CHECK SENTINEL-2 INPUT
# ------------------------------------------------------------

def validate_sentinel_input(
    image_path,
    filename,
):
    year = detect_year(filename)

    if year is None:
        raise ValueError(
            "For Sentinel-2 spectral change analysis, "
            "the image filename must contain 2020 or 2025."
        )

    try:
        with rasterio.open(image_path) as src:
            if src.count < 4:
                raise ValueError(
                    "The Sentinel-2 image must contain at least "
                    "4 bands: B02, B03, B04 and B08."
                )

            if src.crs is None:
                raise ValueError(
                    "The Sentinel-2 image has no CRS information."
                )

            if src.width != src.height:
                raise ValueError(
                    "The Sentinel-2 image dimensions are not supported "
                    "for this demo."
                )

    except rasterio.errors.RasterioIOError as error:
        raise ValueError(
            "The uploaded file is not a readable GeoTIFF."
        ) from error

    return year


# ------------------------------------------------------------
# GET SCL FILE FOR YEAR
# ------------------------------------------------------------

def get_scl_path(year):
    if year == "2020":
        scl_path = SCL_2020

    elif year == "2025":
        scl_path = SCL_2025

    else:
        raise ValueError(
            f"Unsupported Sentinel-2 year: {year}"
        )

    if not scl_path.exists():
        raise FileNotFoundError(
            f"SCL file not found: {scl_path}"
        )

    return scl_path


# ------------------------------------------------------------
# MAIN CHANGE DETECTOR
# ------------------------------------------------------------

def detect_changes(
    image1_bytes,
    image2_bytes,
    image1_name="image1.tif",
    image2_name="image2.tif",
):
    """
    Run Sentinel-2 spectral change detection.

    The uploaded images must be the 4-band Sentinel-2 GeoTIFF
    files used by the SatQuery AI demonstration dataset.

    Expected file names should contain:
        - 2020
        - 2025
    """

    image1_year = detect_year(image1_name)
    image2_year = detect_year(image2_name)

    if image1_year is None or image2_year is None:
        raise ValueError(
            "Please upload the Sentinel-2 2020 and 2025 "
            "GeoTIFF files for spectral change analysis."
        )

    if image1_year == image2_year:
        raise ValueError(
            "Please select one 2020 image and one 2025 image."
        )

    temp_files = []

    try:
        # ----------------------------------------------------
        # Create temporary uploaded GeoTIFF files
        # ----------------------------------------------------

        temp1 = tempfile.NamedTemporaryFile(
            suffix=".tif",
            delete=False,
        )

        temp1.write(image1_bytes)
        temp1.close()

        temp2 = tempfile.NamedTemporaryFile(
            suffix=".tif",
            delete=False,
        )

        temp2.write(image2_bytes)
        temp2.close()

        image1_path = Path(temp1.name)
        image2_path = Path(temp2.name)

        temp_files.extend(
            [
                image1_path,
                image2_path,
            ]
        )

        # ----------------------------------------------------
        # Validate uploaded images
        # ----------------------------------------------------

        validate_sentinel_input(
            image1_path,
            image1_name,
        )

        validate_sentinel_input(
            image2_path,
            image2_name,
        )

        # ----------------------------------------------------
        # Get matching SCL files
        # ----------------------------------------------------

        scl1_path = get_scl_path(
            image1_year
        )

        scl2_path = get_scl_path(
            image2_year
        )

        # ----------------------------------------------------
        # Output file names
        # ----------------------------------------------------

        run_id = uuid.uuid4().hex

        output_mask_path = (
            RESULTS_DIR
            / f"spectral_change_{run_id}.tif"
        )

        output_preview_path = (
            RESULTS_DIR
            / f"spectral_change_{run_id}.png"
        )

        # ----------------------------------------------------
        # Run spectral change detection
        # ----------------------------------------------------

        result = process_full_scene(
            image1_path,
            image2_path,
            scl1_path,
            scl2_path,
            output_mask_path,
        )

        # ----------------------------------------------------
        # Create visual preview
        # ----------------------------------------------------

        create_full_scene_preview(
            image1_path,
            output_mask_path,
            output_preview_path,
        )

        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        change_percentage = float(
            result["change_percent"]
        )

        return {
            "status": "success",
            "analysis_type": "sentinel2_spectral_change",
            "image1_year": image1_year,
            "image2_year": image2_year,
            "change_percentage": round(
                change_percentage,
                2,
            ),
            "threshold": float(
                result["threshold"]
            ),
            "valid_pixels": int(
                result["valid_pixels"]
            ),
            "changed_pixels": int(
                result["changed_pixels"]
            ),
            "change_map": (
                f"/results/{output_preview_path.name}"
            ),
            "change_mask": (
                f"/results/{output_mask_path.name}"
            ),
            "explanation": (
                f"The system detected "
                f"{change_percentage:.2f}% spectral change "
                f"between the {image1_year} and {image2_year} "
                f"Sentinel-2 observations after masking "
                f"cloud, shadow, snow and other invalid pixels."
            ),
        }

    finally:
        # ----------------------------------------------------
        # Remove temporary uploaded files
        # ----------------------------------------------------

        for temp_file in temp_files:
            try:
                temp_file.unlink(
                    missing_ok=True
                )
            except OSError:
                pass