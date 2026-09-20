from pathlib import Path

from services.spectral_change_detector import analyze_change


# ------------------------------------------------------------
# SATQUERY AI - SAME IMAGE SANITY TEST
# ------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

image_2020 = (
    project_root
    / "dataset"
    / "sentinel2_2020"
    / "sentinel2_2020_B02_B03_B04_B08.tif"
)

print("=" * 60)
print("SATQUERY AI - SAME IMAGE SANITY TEST")
print("=" * 60)

print("\nComparing the exact same 2020 image with itself...")

result = analyze_change(
    image_2020,
    image_2020
)

print("\n" + "=" * 60)
print("SAME IMAGE TEST RESULT")
print("=" * 60)

print(f"Change percentage: {result['change_percent']}%")
print(f"Changed pixels: {result['changed_pixels']}")
print(f"Valid pixels: {result['valid_pixels']}")

if result["change_percent"] <= 0.01:
    print("\nPASS: Same image produces approximately 0% change.")
else:
    print(
        "\nWARNING: Same image produced "
        f"{result['change_percent']}% change."
    )