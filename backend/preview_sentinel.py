import rasterio
import numpy as np
from PIL import Image
from pathlib import Path


def create_rgb_preview(tif_path, output_path):
    with rasterio.open(tif_path) as src:
        # Sentinel-2 bands:
        # Band 1 = B02 Blue
        # Band 2 = B03 Green
        # Band 3 = B04 Red
        # Band 4 = B08 NIR

        red = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(1).astype(np.float32)

    # Percentile stretch for visualization
    def stretch(channel):
        low = np.percentile(channel, 2)
        high = np.percentile(channel, 98)

        channel = (channel - low) / (high - low + 1e-6)
        channel = np.clip(channel, 0, 1)

        return (channel * 255).astype(np.uint8)

    rgb = np.dstack([
        stretch(red),
        stretch(green),
        stretch(blue)
    ])

    Image.fromarray(rgb).save(output_path)

    print(f"Created: {output_path}")


base = Path(r"C:\Projects\SatQuery-AI\dataset")

image_2020 = (
    base
    / "sentinel2_2020"
    / "sentinel2_2020_B02_B03_B04_B08.tif"
)

image_2025 = (
    base
    / "sentinel2_2025"
    / "sentinel2_2025_B02_B03_B04_B08.tif"
)

output_2020 = base / "sentinel2_2020_preview.jpg"
output_2025 = base / "sentinel2_2025_preview.jpg"


print("Creating 2020 RGB preview...")
create_rgb_preview(image_2020, output_2020)

print("Creating 2025 RGB preview...")
create_rgb_preview(image_2025, output_2025)

print()
print("========================================")
print("PREVIEWS CREATED SUCCESSFULLY")
print("========================================")
print(output_2020)
print(output_2025)