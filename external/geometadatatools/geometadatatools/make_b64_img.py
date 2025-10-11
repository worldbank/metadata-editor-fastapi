import base64
from io import BytesIO

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal


def to_base64_image(
    gdf: gpd.GeoDataFrame | None = None, raster_data: np.ndarray | None = None, cmap: str = "viridis"
) -> str:
    """
    Converts a GeoDataFrame or raster data to a Base64 encoded image string.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to be plotted.
        raster_data (np.ndarray): The raster data to be plotted.
        cmap (str): The colormap to be used for the raster data. Default is 'viridis'.

    Returns:
        str: Base64 encoded string of the plot.

    Example:
        img_string = to_base64_image(gdf)
        html_img_tag = f'<img src="data:image/png;base64,{img_string}" />'

    """
    fig, ax = plt.subplots()
    try:
        if raster_data is None and gdf is None:
            raise ValueError("Either gdf or raster_data must be provided.")

        if raster_data is not None:
            # If raster data is provided, plot it
            ax.imshow(raster_data, cmap=cmap)
        else:
            # Plot the GeoDataFrame
            if "geometry" not in gdf.columns:
                # gdf.plot(ax=ax, cmap=cmap)
                return ""
            gdf.plot(ax=ax, cmap=cmap, edgecolor="black")

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Convert the BytesIO object to a Base64 encoded string
        return base64.b64encode(buf.read()).decode("utf-8")
    finally:
        # Ensure the plot is closed even if an error occurs
        plt.close(fig)


def choose_colormap(band: gdal.Band, default_cmap="viridis") -> str:
    """
    Choose a colormap based on raster metadata.

    Args:
        band (gdal.Band): GDAL band object.
        default_cmap (str): Default colormap to use if no match is found.

    Returns:
        str: The name of the chosen colormap.
    """
    metadata = band.GetMetadata_Dict()
    # Check for specific metadata keys
    if metadata:
        if (
            "elevation" in metadata.get("description", "").lower()
            or "elevation" in metadata.get("long_name", "").lower()
        ):
            return "terrain"
        if (
            "temperature" in metadata.get("description", "").lower()
            or "temperature" in metadata.get("long_name", "").lower()
        ):
            return "coolwarm"
        if (
            "categorical" in metadata.get("description", "").lower()
            or "categories" in metadata.get("long_name", "").lower()
        ):
            return "tab10"
        if "probability" in metadata.get("description", "").lower():
            return "plasma"

    # Default colormap
    return default_cmap


def raster_to_base64_images(dataset: gdal.Dataset, layer_name_or_band_index: None | str | int = None) -> list[str]:
    """
    Convert raster bands to base64 images.

    Args:
        dataset (gdal.Dataset): GDAL dataset object.
        layer_name_or_band_index (None|str|int): Layer name or band index to read from the dataset.
        If None, all bands will be processed.

    Returns:
        list: List of base64 encoded images.
    """
    bands = dataset.RasterCount
    imgs = []

    if layer_name_or_band_index is not None:
        band = dataset.GetRasterBand(layer_name_or_band_index)
        # band_data = dataset.GetRasterBand(band)
        # band_array = band_data.ReadAsArray()
        raster_data = band.ReadAsArray()
        mask_band = band.GetMaskBand()
        mask_data = mask_band.ReadAsArray()
        chosen_cmap = choose_colormap(band)
        raster_data = np.where(mask_data == 0, np.nan, raster_data)
        img = to_base64_image(gdf=None, raster_data=raster_data, cmap=chosen_cmap)
        imgs.append(img)
    else:
        for i in range(1, bands + 1):
            band = dataset.GetRasterBand(i)  # Band i
            # band_data = dataset.GetRasterBand(band)
            # band_array = band_data.ReadAsArray()
            raster_data = band.ReadAsArray()
            mask_band = band.GetMaskBand()
            mask_data = mask_band.ReadAsArray()
            chosen_cmap = choose_colormap(band)
            raster_data = np.where(mask_data == 0, np.nan, raster_data)
            img = to_base64_image(gdf=None, raster_data=raster_data, cmap=chosen_cmap)
            imgs.append(img)
    return imgs
