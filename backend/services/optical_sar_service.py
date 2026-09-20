from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio


DATASET_DIR = (
    Path(__file__).resolve().parents[1] / "dataset"
)

SAR_VV_PATH = DATASET_DIR / "sar_vv_pune.tif"
SAR_VH_PATH = DATASET_DIR / "sar_vh_pune.tif"


def _read_sar_band(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"SAR file not found: {path.name}"
        )

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        profile = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs) if src.crs else None,
            "bounds": [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
        }

    return data, profile


def _safe_stats(data: np.ndarray):
    valid = np.isfinite(data)

    if not np.any(valid):
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    values = data[valid]

    return {
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
    }


def analyze_optical_sar(optical_image_bytes: bytes):
    """
    Analyze the supplied optical image together with the
    locally stored Sentinel-1 VV and VH SAR bands.

    The optical image is currently used as the visual reference.
    VV and VH statistics provide complementary SAR information.
    """

    if not optical_image_bytes:
        raise ValueError("Optical image is empty.")

    vv, vv_info = _read_sar_band(SAR_VV_PATH)
    vh, vh_info = _read_sar_band(SAR_VH_PATH)

    if vv.shape != vh.shape:
        raise ValueError(
            "VV and VH SAR images have different dimensions."
        )

    vv_valid = np.isfinite(vv)
    vh_valid = np.isfinite(vh)

    valid_pixels = vv_valid & vh_valid

    if not np.any(valid_pixels):
        raise ValueError(
            "No valid pixels were found in the SAR images."
        )

    vv_values = vv[valid_pixels]
    vh_values = vh[valid_pixels]

    vv_mean = float(np.mean(vv_values))
    vh_mean = float(np.mean(vh_values))

    # VV/VH difference is useful as a simple SAR backscatter
    # relationship indicator.
    ratio_difference = vv_values - vh_values
    ratio_mean = float(np.mean(ratio_difference))

    if ratio_mean > 5:
        sar_observation = (
            "VV backscatter is noticeably stronger than VH "
            "backscatter in the analyzed scene."
        )
    elif ratio_mean < 1:
        sar_observation = (
            "VV and VH backscatter levels are relatively close "
            "in the analyzed scene."
        )
    else:
        sar_observation = (
            "VV shows moderately stronger backscatter than VH "
            "in the analyzed scene."
        )

    return {
        "status": "success",
        "analysis_type": "optical_sar_complementary_analysis",
        "message": (
            "Optical imagery was paired with Sentinel-1 VV/VH "
            "SAR information."
        ),
        "sar_data": {
            "vv_file": SAR_VV_PATH.name,
            "vh_file": SAR_VH_PATH.name,
            "dimensions": {
                "width": vv_info["width"],
                "height": vv_info["height"],
            },
            "crs": vv_info["crs"],
            "vv_statistics": _safe_stats(vv),
            "vh_statistics": _safe_stats(vh),
            "vv_minus_vh_mean": round(ratio_mean, 2),
        },
        "observation": sar_observation,
        "evidence": {
            "optical_input_received": True,
            "sar_vv_loaded": True,
            "sar_vh_loaded": True,
            "valid_sar_pixels": int(np.sum(valid_pixels)),
        },
        "confidence": {
            "value": None,
            "status": "not_calibrated",
            "note": (
                "This is an analytical observation, not a "
                "calibrated probability."
            ),
        },
    }