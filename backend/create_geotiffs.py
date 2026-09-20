import rasterio
import os

datasets = [
    {
        "name": "sentinel2_2020",
        "folder": r"C:\Projects\SatQuery-AI\dataset\sentinel2_2020",
        "prefix": "T43QCA_20200115T053151"
    },
    {
        "name": "sentinel2_2025",
        "folder": r"C:\Projects\SatQuery-AI\dataset\sentinel2_2025",
        "prefix": "T43QCA_20250118T053141"
    }
]

for item in datasets:
    folder = item["folder"]
    prefix = item["prefix"]

    band_files = [
        os.path.join(folder, f"{prefix}_B02_10m.jp2"),
        os.path.join(folder, f"{prefix}_B03_10m.jp2"),
        os.path.join(folder, f"{prefix}_B04_10m.jp2"),
        os.path.join(folder, f"{prefix}_B08_10m.jp2"),
    ]

    output = os.path.join(
        folder,
        f"{item['name']}_B02_B03_B04_B08.tif"
    )

    print(f"\nCreating: {output}")

    with rasterio.open(band_files[0]) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=4,
            compress="deflate",
            tiled=True
        )

        with rasterio.open(output, "w", **profile) as dst:
            for band_number, band_file in enumerate(band_files, start=1):
                print(f"  Writing Band {band_number}: {os.path.basename(band_file)}")

                with rasterio.open(band_file) as src_band:
                    dst.write(src_band.read(1), band_number)

    print("Created successfully.")

print("\nDONE - Both GeoTIFF files created.")
