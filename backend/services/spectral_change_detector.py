from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image


# ============================================================
# SATQUERY AI
# FULL SCENE SPECTRAL CHANGE DETECTOR
# ============================================================

EPS = 1e-6

DEFAULT_WINDOW_SIZE = 1024

# Top 2% spectral differences are considered potential change
CHANGE_PERCENTILE = 98.0

# Remove very small noisy regions
MINIMUM_REGION_PIXELS = 1000

# Maximum preview dimension
PREVIEW_MAX_SIZE = 1600


# ============================================================
# SCL CLOUD / SHADOW MASKING
# ============================================================

SCL_INVALID_CLASSES = {
    0,   # No Data
    3,   # Cloud Shadow
    8,   # Cloud Medium Probability
    9,   # Cloud High Probability
    10,  # Cirrus
    11   # Snow / Ice
}


# ============================================================
# READ SCL CLOUD / SHADOW MASK
# ============================================================

def read_scl_mask(
    scl_src,
    window
):
    """
    Read Sentinel-2 Scene Classification Layer (SCL).

    SCL is 20m resolution, while the spectral bands
    are 10m resolution.

    Nearest-neighbour resampling keeps the original
    SCL class values unchanged.
    """

    x = int(window.col_off)
    y = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    # Convert 10m window coordinates to 20m coordinates
    scl_window = Window(
        x / 2,
        y / 2,
        width / 2,
        height / 2
    )

    scl = scl_src.read(
        1,
        window=scl_window,
        out_shape=(height, width),
        resampling=Resampling.nearest
    )

    return scl


# ============================================================
# CREATE VALID PIXEL MASK
# ============================================================

def create_valid_mask(
    scl1,
    scl2
):
    """
    Return pixels that are valid in BOTH images.

    Cloud, cloud-shadow, cirrus, snow/ice and no-data
    pixels are excluded.
    """

    valid1 = ~np.isin(
        scl1,
        list(SCL_INVALID_CLASSES)
    )

    valid2 = ~np.isin(
        scl2,
        list(SCL_INVALID_CLASSES)
    )

    return valid1 & valid2


# ============================================================
# READ FOUR SENTINEL-2 BANDS
# ============================================================

def read_four_bands(
    src,
    window
):
    """
    Read four Sentinel-2 bands.

    Band mapping:
        Band 1 = B02 Blue
        Band 2 = B03 Green
        Band 3 = B04 Red
        Band 4 = B08 NIR
    """

    blue = src.read(
        1,
        window=window
    ).astype(
        np.float32
    )

    green = src.read(
        2,
        window=window
    ).astype(
        np.float32
    )

    red = src.read(
        3,
        window=window
    ).astype(
        np.float32
    )

    nir = src.read(
        4,
        window=window
    ).astype(
        np.float32
    )

    return (
        blue,
        green,
        red,
        nir
    )


# ============================================================
# NDVI
# ============================================================

def calculate_ndvi(
    red,
    nir
):
    """
    NDVI = (NIR - Red) / (NIR + Red)
    """

    denominator = nir + red

    ndvi = np.zeros_like(
        nir,
        dtype=np.float32
    )

    valid = np.abs(
        denominator
    ) > EPS

    ndvi[valid] = (
        (nir[valid] - red[valid]) /
        (denominator[valid] + EPS)
    )

    return np.clip(
        ndvi,
        -1.0,
        1.0
    )


# ============================================================
# NDWI
# ============================================================

def calculate_ndwi(
    green,
    nir
):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    """

    denominator = green + nir

    ndwi = np.zeros_like(
        green,
        dtype=np.float32
    )

    valid = np.abs(
        denominator
    ) > EPS

    ndwi[valid] = (
        (green[valid] - nir[valid]) /
        (denominator[valid] + EPS)
    )

    return np.clip(
        ndwi,
        -1.0,
        1.0
    )


# ============================================================
# NORMALIZED DIFFERENCE
# ============================================================

def normalized_difference(
    image1,
    image2
):
    """
    Normalized absolute difference.
    """

    return np.abs(
        image1 - image2
    ) / (
        np.abs(image1) +
        np.abs(image2) +
        EPS
    )


# ============================================================
# SPECTRAL CHANGE SCORE
# ============================================================

def calculate_change_score(
    blue1,
    green1,
    red1,
    nir1,
    blue2,
    green2,
    red2,
    nir2
):
    """
    Calculate combined spectral change score.

    Weights:

        B02  = 15%
        B03  = 15%
        B04  = 15%
        B08  = 15%
        NDVI = 25%
        NDWI = 15%
    """

    # --------------------------------------------------------
    # Raw spectral bands
    # --------------------------------------------------------

    b02_change = normalized_difference(
        blue1,
        blue2
    )

    b03_change = normalized_difference(
        green1,
        green2
    )

    b04_change = normalized_difference(
        red1,
        red2
    )

    b08_change = normalized_difference(
        nir1,
        nir2
    )

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    ndvi1 = calculate_ndvi(
        red1,
        nir1
    )

    ndvi2 = calculate_ndvi(
        red2,
        nir2
    )

    ndvi_change = (
        np.abs(
            ndvi1 - ndvi2
        ) / 2.0
    )

    # --------------------------------------------------------
    # NDWI
    # --------------------------------------------------------

    ndwi1 = calculate_ndwi(
        green1,
        nir1
    )

    ndwi2 = calculate_ndwi(
        green2,
        nir2
    )

    ndwi_change = (
        np.abs(
            ndwi1 - ndwi2
        ) / 2.0
    )

    # --------------------------------------------------------
    # Combined weighted score
    # --------------------------------------------------------

    score = (
        0.15 * b02_change +
        0.15 * b03_change +
        0.15 * b04_change +
        0.15 * b08_change +
        0.25 * ndvi_change +
        0.15 * ndwi_change
    )

    return score.astype(
        np.float32
    )


# ============================================================
# REMOVE SMALL REGIONS
# ============================================================

def remove_small_regions(
    mask,
    minimum_pixels=MINIMUM_REGION_PIXELS
):
    """
    Remove tiny connected regions.
    """

    mask_uint8 = (
        mask > 0
    ).astype(
        np.uint8
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        mask_uint8
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_pixels:

            cleaned[
                labels == label
            ] = 1

    return cleaned


# ============================================================
# CREATE CHANGE MASK
# ============================================================

def create_change_mask(
    score,
    threshold
):
    """
    Convert spectral score to binary change mask.
    """

    valid = np.isfinite(
        score
    )

    mask = np.zeros(
        score.shape,
        dtype=np.uint8
    )

    mask[
        valid &
        (score >= threshold)
    ] = 1

    mask = remove_small_regions(
        mask
    )

    return mask


# ============================================================
# STRETCH BAND
# ============================================================

def stretch_band(
    band
):
    """
    Convert Sentinel-2 values to 8-bit display values.
    """

    band = band.astype(
        np.float32
    )

    valid = np.isfinite(
        band
    )

    if not np.any(valid):

        return np.zeros_like(
            band,
            dtype=np.uint8
        )

    values = band[
        valid
    ]

    low = np.percentile(
        values,
        2
    )

    high = np.percentile(
        values,
        98
    )

    if high <= low:

        return np.zeros_like(
            band,
            dtype=np.uint8
        )

    stretched = (
        (band - low) /
        (high - low)
    )

    stretched = np.clip(
        stretched,
        0,
        1
    )

    return (
        stretched * 255
    ).astype(
        np.uint8
    )


# ============================================================
# CREATE RGB PREVIEW
# ============================================================

def create_rgb_preview(
    blue,
    green,
    red
):
    """
    Create RGB image using:

        Red   = B04
        Green = B03
        Blue  = B02
    """

    red8 = stretch_band(
        red
    )

    green8 = stretch_band(
        green
    )

    blue8 = stretch_band(
        blue
    )

    rgb = np.stack(
        [
            red8,
            green8,
            blue8
        ],
        axis=-1
    )

    return rgb


# ============================================================
# CREATE CLEAR CHANGE OVERLAY
# ============================================================

def create_change_overlay(
    rgb,
    change_mask
):
    """
    Create clear visual change map.

    Background:
        Original satellite image, darkened.

    Changed areas:
        Bright RED.

    Boundaries:
        YELLOW.
    """

    # --------------------------------------------------------
    # Make sure RGB is uint8
    # --------------------------------------------------------

    rgb = np.asarray(
        rgb,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Boolean change mask
    # --------------------------------------------------------

    changed = (
        change_mask > 0
    )

    # --------------------------------------------------------
    # Darken original image
    # --------------------------------------------------------

    overlay = (
        rgb.astype(
            np.float32
        ) * 0.65
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Bright RED change regions
    # --------------------------------------------------------

    overlay[
        changed
    ] = np.array(
        [
            255,
            0,
            0
        ],
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Create contour mask
    # --------------------------------------------------------

    mask_uint8 = (
        changed * 255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Find connected region boundaries
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Draw YELLOW boundaries
    #
    # OpenCV draws BGR.
    # Since our overlay is RGB, use:
    # R=255, G=255, B=0
    # --------------------------------------------------------

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 255, 0),
        2
    )

    return overlay


# ============================================================
# COMPATIBILITY CHECK
# ============================================================

def check_compatibility(
    src1,
    src2
):
    print(
        "Compatibility check:",
        end=" "
    )

    # --------------------------------------------------------
    # Width
    # --------------------------------------------------------

    if src1.width != src2.width:

        print("FAILED")

        raise ValueError(
            "Image widths are different."
        )

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    if src1.height != src2.height:

        print("FAILED")

        raise ValueError(
            "Image heights are different."
        )

    # --------------------------------------------------------
    # Bands
    # --------------------------------------------------------

    if src1.count < 4:

        print("FAILED")

        raise ValueError(
            "First image must contain at least 4 bands."
        )

    if src2.count < 4:

        print("FAILED")

        raise ValueError(
            "Second image must contain at least 4 bands."
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if src1.crs != src2.crs:

        print("FAILED")

        raise ValueError(
            "CRS mismatch between images."
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    if not np.allclose(
        src1.transform,
        src2.transform
    ):

        print("FAILED")

        raise ValueError(
            "Geospatial transforms do not match."
        )

    print("PASSED")


# ============================================================
# GENERATE WINDOWS
# ============================================================

def generate_windows(
    width,
    height,
    window_size=DEFAULT_WINDOW_SIZE
):
    windows = []

    for y in range(
        0,
        height,
        window_size
    ):

        for x in range(
            0,
            width,
            window_size
        ):

            w = min(
                window_size,
                width - x
            )

            h = min(
                window_size,
                height - y
            )

            windows.append(
                Window(
                    x,
                    y,
                    w,
                    h
                )
            )

    return windows


# ============================================================
# COLLECT GLOBAL SCORE SAMPLES
# ============================================================

def collect_score_samples(
    src1,
    src2,
    scl1,
    scl2,
    windows
):
    samples = []

    total_windows = len(
        windows
    )

    for index, window in enumerate(
        windows,
        start=1
    ):

        # ----------------------------------------------------
        # Read first image
        # ----------------------------------------------------

        (
            blue1,
            green1,
            red1,
            nir1
        ) = read_four_bands(
            src1,
            window
        )

        # ----------------------------------------------------
        # Read second image
        # ----------------------------------------------------

        (
            blue2,
            green2,
            red2,
            nir2
        ) = read_four_bands(
            src2,
            window
        )

        # ----------------------------------------------------
        # Calculate spectral score
        # ----------------------------------------------------

        score = calculate_change_score(
            blue1,
            green1,
            red1,
            nir1,
            blue2,
            green2,
            red2,
            nir2
        )

        # ----------------------------------------------------
        # READ SCL MASKS
        # ----------------------------------------------------

        scl_mask1 = read_scl_mask(
            scl1,
            window
        )

        scl_mask2 = read_scl_mask(
            scl2,
            window
        )

        # ----------------------------------------------------
        # Keep only pixels valid in BOTH dates
        # ----------------------------------------------------

        valid_scl = create_valid_mask(
            scl_mask1,
            scl_mask2
        )

        # ----------------------------------------------------
        # Exclude cloud / shadow / snow pixels
        # ----------------------------------------------------

        score[
            ~valid_scl
        ] = np.nan

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # Calculate valid AFTER applying SCL mask.
        # ----------------------------------------------------

        valid = np.isfinite(
            score
        )

        values = score[
            valid
        ]

        if values.size > 0:

            # Sample every 16th pixel
            sampled = values[
                ::16
            ]

            samples.append(
                sampled
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 10 == 0
            or index == total_windows
        ):

            print(
                f"  Windows processed: "
                f"{index}/{total_windows}"
            )

    if not samples:

        raise RuntimeError(
            "No valid spectral pixels found."
        )

    all_samples = np.concatenate(
        samples
    )

    return all_samples


# ============================================================
# PROCESS FULL SCENE
# ============================================================

def process_full_scene(
    image1_path,
    image2_path,
    scl1_path,
    scl2_path,
    output_mask_path
):

    with rasterio.open(
        image1_path
    ) as src1, rasterio.open(
        image2_path
    ) as src2, rasterio.open(
        scl1_path
    ) as scl1, rasterio.open(
        scl2_path
    ) as scl2:

        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        print(
            f"Image size: "
            f"{src1.width} x {src1.height}"
        )

        print(
            f"CRS: {src1.crs}"
        )

        # ----------------------------------------------------
        # Compatibility
        # ----------------------------------------------------

        check_compatibility(
            src1,
            src2
        )

        # ----------------------------------------------------
        # Generate windows
        # ----------------------------------------------------

        windows = generate_windows(
            src1.width,
            src1.height
        )

        # ====================================================
        # PASS 1
        # ====================================================

        print()

        print(
            "PASS 1/2: "
            "Calculating global spectral threshold..."
        )

        samples = collect_score_samples(
            src1,
            src2,
            scl1,
            scl2,
            windows
        )

        # ----------------------------------------------------
        # Calculate threshold
        # ----------------------------------------------------

        threshold = float(
            np.percentile(
                samples,
                CHANGE_PERCENTILE
            )
        )

        print(
            f"Global threshold: "
            f"{threshold:.6f}"
        )

        # ====================================================
        # PASS 2
        # ====================================================

        print()

        print(
            "PASS 2/2: "
            "Creating complete change mask..."
        )

        # ----------------------------------------------------
        # Output GeoTIFF profile
        # ----------------------------------------------------

        profile = src1.profile.copy()

        profile.update(
            count=1,
            dtype=rasterio.uint8,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER"
        )

        total_valid_pixels = 0

        total_changed_pixels = 0

        # ----------------------------------------------------
        # Create output GeoTIFF
        # ----------------------------------------------------

        with rasterio.open(
            output_mask_path,
            "w",
            **profile
        ) as dst:

            total_windows = len(
                windows
            )

            for index, window in enumerate(
                windows,
                start=1
            ):

                # ------------------------------------------------
                # Read image 1
                # ------------------------------------------------

                (
                    blue1,
                    green1,
                    red1,
                    nir1
                ) = read_four_bands(
                    src1,
                    window
                )

                # ------------------------------------------------
                # Read image 2
                # ------------------------------------------------

                (
                    blue2,
                    green2,
                    red2,
                    nir2
                ) = read_four_bands(
                    src2,
                    window
                )

                # ------------------------------------------------
                # Calculate score
                # ------------------------------------------------

                score = calculate_change_score(
                    blue1,
                    green1,
                    red1,
                    nir1,
                    blue2,
                    green2,
                    red2,
                    nir2
                )

                # ------------------------------------------------
                # READ SCL MASKS
                # ------------------------------------------------

                scl_mask1 = read_scl_mask(
                    scl1,
                    window
                )

                scl_mask2 = read_scl_mask(
                    scl2,
                    window
                )

                # ------------------------------------------------
                # Keep only pixels valid in BOTH dates
                # ------------------------------------------------

                valid_scl = create_valid_mask(
                    scl_mask1,
                    scl_mask2
                )

                # ------------------------------------------------
                # Exclude cloud / shadow / snow pixels
                # ------------------------------------------------

                score[
                    ~valid_scl
                ] = np.nan

                # ------------------------------------------------
                # Valid pixels
                # ------------------------------------------------

                valid = np.isfinite(
                    score
                )

                # ------------------------------------------------
                # Create mask
                # ------------------------------------------------

                mask = create_change_mask(
                    score,
                    threshold
                )

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                valid_count = int(
                    np.count_nonzero(
                        valid
                    )
                )

                changed_count = int(
                    np.count_nonzero(
                        mask & valid
                    )
                )

                total_valid_pixels += (
                    valid_count
                )

                total_changed_pixels += (
                    changed_count
                )

                # ------------------------------------------------
                # Write mask
                # ------------------------------------------------

                dst.write(
                    mask.astype(
                        np.uint8
                    ),
                    1,
                    window=window
                )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    index == 1
                    or index % 10 == 0
                    or index == total_windows
                ):

                    print(
                        f"  Windows processed: "
                        f"{index}/{total_windows}"
                    )

        # ====================================================
        # FINAL STATISTICS
        # ====================================================

        print()

        print(
            f"Valid pixels: "
            f"{total_valid_pixels}"
        )

        print(
            f"Changed pixels: "
            f"{total_changed_pixels}"
        )

        if total_valid_pixels > 0:

            change_percent = (
                total_changed_pixels /
                total_valid_pixels
            ) * 100.0

        else:

            change_percent = 0.0

        print(
            f"Full-scene spectral change: "
            f"{change_percent:.2f}%"
        )

        print(
            f"Full change mask saved to: "
            f"{output_mask_path}"
        )

        return {
            "threshold": threshold,
            "valid_pixels": total_valid_pixels,
            "changed_pixels": total_changed_pixels,
            "change_percent": change_percent
        }


# ============================================================
# CREATE FULL SCENE PREVIEW
# ============================================================

def create_full_scene_preview(
    image_path,
    mask_path,
    output_preview_path
):

    print(
        "Reading satellite image for preview..."
    )

    with rasterio.open(
        image_path
    ) as src_image, rasterio.open(
        mask_path
    ) as src_mask:

        width = src_image.width

        height = src_image.height

        # ----------------------------------------------------
        # Calculate preview size
        # ----------------------------------------------------

        scale = min(
            1.0,
            PREVIEW_MAX_SIZE /
            max(
                width,
                height
            )
        )

        preview_width = max(
            1,
            int(
                width * scale
            )
        )

        preview_height = max(
            1,
            int(
                height * scale
            )
        )

        print(
            f"Preview size: "
            f"{preview_width} x "
            f"{preview_height}"
        )

        # ====================================================
        # READ RGB
        # ====================================================

        red = src_image.read(
            3,
            out_shape=(
                preview_height,
                preview_width
            ),
            resampling=Resampling.bilinear
        )

        green = src_image.read(
            2,
            out_shape=(
                preview_height,
                preview_width
            ),
            resampling=Resampling.bilinear
        )

        blue = src_image.read(
            1,
            out_shape=(
                preview_height,
                preview_width
            ),
            resampling=Resampling.bilinear
        )

        # ====================================================
        # CREATE RGB
        # ====================================================

        rgb = create_rgb_preview(
            blue,
            green,
            red
        )

        # ====================================================
        # READ CHANGE MASK
        #
        # IMPORTANT:
        # nearest is used.
        # ====================================================

        change_mask = src_mask.read(
            1,
            out_shape=(
                preview_height,
                preview_width
            ),
            resampling=Resampling.nearest
        )

        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        overlay = create_change_overlay(
            rgb,
            change_mask
        )

        # ====================================================
        # SAVE PNG USING PILLOW
        # ====================================================

        output_preview_path = Path(
            output_preview_path
        )

        output_preview_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "Saving PNG preview..."
        )

        preview_image = Image.fromarray(
            overlay,
            "RGB"
        )

        preview_image.save(
            str(output_preview_path),
            format="PNG",
            optimize=True
        )

        # ====================================================
        # VERIFY
        # ====================================================

        if not output_preview_path.exists():

            raise RuntimeError(
                "Preview file was not created."
            )

        file_size = (
            output_preview_path.stat().st_size
        )

        if file_size <= 0:

            raise RuntimeError(
                "Preview file is empty."
            )

        print(
            f"Preview saved to: "
            f"{output_preview_path}"
        )

        print(
            f"Preview size: "
            f"{file_size:,} bytes"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "SATQUERY AI - FULL SCENE "
        "SPECTRAL CHANGE DETECTOR"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # --------------------------------------------------------
    # Sentinel-2 2020
    # --------------------------------------------------------

    image_2020 = (
        project_root /
        "dataset" /
        "sentinel2_2020" /
        "sentinel2_2020_B02_B03_B04_B08.tif"
    )

    # --------------------------------------------------------
    # Sentinel-2 2025
    # --------------------------------------------------------

    image_2025 = (
        project_root /
        "dataset" /
        "sentinel2_2025" /
        "sentinel2_2025_B02_B03_B04_B08.tif"
    )

    # --------------------------------------------------------
    # Sentinel-2 SCL 2020
    # --------------------------------------------------------

    scl_2020 = (
        project_root /
        "dataset" /
        "sentinel2_2020" /
        "T43QCA_20200115T053151_SCL_20m.jp2"
    )

    # --------------------------------------------------------
    # Sentinel-2 SCL 2025
    # --------------------------------------------------------

    scl_2025 = (
        project_root /
        "dataset" /
        "sentinel2_2025" /
        "T43QCA_20250118T053141_SCL_20m.jp2"
    )

    # --------------------------------------------------------
    # Results folder
    # --------------------------------------------------------

    results_dir = (
        project_root /
        "backend" /
        "data" /
        "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Output GeoTIFF
    # --------------------------------------------------------

    output_mask = (
        results_dir /
        "spectral_change_full_scene.tif"
    )

    # --------------------------------------------------------
    # Output PNG
    # --------------------------------------------------------

    output_preview = (
        results_dir /
        "spectral_change_full_scene.png"
    )

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if not image_2020.exists():

        raise FileNotFoundError(
            "2020 Sentinel-2 image not found:\n"
            f"{image_2020}"
        )

    if not image_2025.exists():

        raise FileNotFoundError(
            "2025 Sentinel-2 image not found:\n"
            f"{image_2025}"
        )

    if not scl_2020.exists():

        raise FileNotFoundError(
            "2020 SCL file not found:\n"
            f"{scl_2020}"
        )

    if not scl_2025.exists():

        raise FileNotFoundError(
            "2025 SCL file not found:\n"
            f"{scl_2025}"
        )

    # ========================================================
    # RUN FULL SCENE DETECTOR
    # ========================================================

    result = process_full_scene(
        image_2020,
        image_2025,
        scl_2020,
        scl_2025,
        output_mask
    )

    # ========================================================
    # CREATE PREVIEW
    # ========================================================

    print()

    print(
        "Creating full-scene preview..."
    )

    create_full_scene_preview(
        image_2025,
        output_mask,
        output_preview
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 60)

    print(
        "PROCESS COMPLETED"
    )

    print("=" * 60)

    print(
        f"Spectral change: "
        f"{result['change_percent']:.2f}%"
    )

    print()

    print(
        "Change mask:"
    )

    print(
        output_mask
    )

    print()

    print(
        "Preview:"
    )

    print(
        output_preview
    )

    print()

    print("=" * 60)