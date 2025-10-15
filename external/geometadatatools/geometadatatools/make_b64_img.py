import base64
from io import BytesIO

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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


# def raster_to_base64_images(
#     dataset: gdal.Dataset,
#     layer_name_or_band_index: None | str | int = None,
#     max_pixels: int = 100_000,
# ) -> list[str]:
#     """
#     Convert raster bands to base64 images. Downsamples automatically so previews
#     won't load the full-resolution raster into memory.

#     Args:
#         dataset (gdal.Dataset): GDAL dataset object.
#         layer_name_or_band_index (None|str|int): Layer name or band index to read from the dataset.
#             If None, all bands will be processed.
#         max_pixels (int): maximum number of pixels to decode per preview (default 100_000).

#     Returns:
#         list[str]: List of base64 encoded images (downsampled).
#     """
#     xsize = dataset.RasterXSize
#     ysize = dataset.RasterYSize
#     bands = dataset.RasterCount
#     imgs: list[str] = []

#     def compute_buf(x: int, y: int, max_pix: int) -> tuple[int, int]:
#         total = x * y
#         if total <= max_pix:
#             return x, y
#         scale = (max_pix / total)**0.5
#         return max(1, int(x * scale)), max(1, int(y * scale))

#     def read_band_preview(band: gdal.Band) -> np.ndarray:
#         buf_x, buf_y = compute_buf(xsize, ysize, max_pixels)
#         # Read full window but request a downsampled buffer
#         arr = band.ReadAsArray(0, 0, xsize, ysize, buf_x, buf_y)
#         # Try mask band if available (downsample same way); some bands may not have a mask
#         try:
#             mask = band.GetMaskBand()
#             if mask is not None:
#                 m = mask.ReadAsArray(0, 0, xsize, ysize, buf_x, buf_y)
#                 arr = np.where(m == 0, np.nan, arr)
#         except Exception:
#             pass
#         return arr

#     def normalize_for_display(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         # map to 0-1 ignoring NaNs then to 0-255 uint8
#         if arr is None:
#             return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=bool)
#         a = np.array(arr, dtype=np.float64)
#         mask = np.isnan(a)
#         valid = ~mask
#         if not np.any(valid):
#             return np.zeros(a.shape, dtype=np.uint8), mask
#         vmin = float(np.nanmin(a))
#         vmax = float(np.nanmax(a))
#         if vmin == vmax:
#             norm = np.clip(a - vmin, 0, 1)
#         else:
#             norm = (a - vmin) / (vmax - vmin)
#         # keep mask separate (don't map NaN -> 0)
#         norm = np.nan_to_num(norm, nan=0.0)
#         return (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8), mask

#     # Helper to create a base64 PNG using existing utility (smaller arrays now)
#     def make_img_from_array(arr: np.ndarray, cmap: str | None = "viridis") -> str:
#         # Convert single-band to RGB using a colormap
#         if arr.ndim == 2:
#             norm8, mask = normalize_for_display(arr)
#             cmap_fn = cm.get_cmap(cmap or "viridis")
#             rgba = cmap_fn(norm8 / 255.0)  # returns float RGBA
#             rgba8 = (rgba * 255).astype(np.uint8)
#             # set masked (NaN) pixels to fully transparent, then composite onto white
#             rgba8[mask, 3] = 0
#             # composite onto white so null pixels appear white (instead of low-colour)
#             alpha = rgba8[:, :, 3:4].astype(np.float32) / 255.0
#             rgb = rgba8[:, :, :3].astype(np.float32)
#             white = np.ones_like(rgb) * 255.0
#             comp = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
#             img_arr = comp
#         else:
#             # if multi-band small array, try to use first three bands as RGB
#             bands = min(arr.shape[0], 3)
#             stacked = []
#             masks = []
#             for i in range(bands):
#                 n8, m = normalize_for_display(arr[i])
#                 stacked.append(n8)
#                 masks.append(m)
#             img_arr = np.stack(stacked, axis=-1)
#             # any pixel that is NaN in all used bands -> white
#             combined_mask = np.logical_and.reduce(masks)
#             img_arr[combined_mask] = 255
#         # Use matplotlib / PIL via existing to_base64_image by passing the small array
#         return to_base64_image(gdf=None, raster_data=img_arr, cmap=None)

#     if layer_name_or_band_index is not None:
#         band = dataset.GetRasterBand(layer_name_or_band_index)
#         arr = read_band_preview(band)
#         chosen_cmap = choose_colormap(band)
#         imgs.append(make_img_from_array(arr, chosen_cmap))
#     else:
#         for i in range(1, bands + 1):
#             band = dataset.GetRasterBand(i)
#             arr = read_band_preview(band)
#             chosen_cmap = choose_colormap(band)
#             imgs.append(make_img_from_array(arr, chosen_cmap))

#     return imgs


def raster_to_base64_images(
    dataset: gdal.Dataset,
    layer_name_or_band_index: None | str | int = None,
    max_pixels: int = 100_000,
) -> list[str]:
    """
    Convert raster bands to base64 images. Downsamples automatically so previews
    won't load the full-resolution raster into memory.
    """
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    bands = dataset.RasterCount
    imgs: list[str] = []

    def compute_buf(x: int, y: int, max_pix: int) -> tuple[int, int]:
        total = x * y
        if total <= max_pix:
            return x, y
        scale = (max_pix / total) ** 0.5
        return max(1, int(x * scale)), max(1, int(y * scale))

    def read_band_preview(band: gdal.Band) -> np.ndarray:
        buf_x, buf_y = compute_buf(xsize, ysize, max_pixels)
        try:
            arr = band.ReadAsArray(0, 0, xsize, ysize, buf_x, buf_y)
        except Exception as e:
            print(f"⚠️ GDAL read error: {e}. Returning empty array.")
            return np.zeros((1, 1), dtype=np.uint8)
        try:
            mask = band.GetMaskBand()
            if mask is not None:
                m = mask.ReadAsArray(0, 0, xsize, ysize, buf_x, buf_y)
                arr = np.where(m == 0, np.nan, arr)
        except Exception:
            pass
        return arr

    def normalize_for_display(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if arr is None:
            return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=bool)
        a = np.array(arr, dtype=np.float64)
        mask = np.isnan(a)
        valid = ~mask
        if not np.any(valid):
            return np.zeros(a.shape, dtype=np.uint8), mask
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))
        norm = np.zeros_like(a, dtype=np.float64)
        if vmin != vmax:
            norm = (a - vmin) / (vmax - vmin)
        norm = np.nan_to_num(norm, nan=0.0)
        return (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8), mask

    def make_img_from_array(arr: np.ndarray, cmap: str | None = "viridis", textless: bool = False) -> str:
        """
        Renders an array to base64. If textless=True, all matplotlib text elements
        (axes, ticks, labels, titles) are disabled.
        """
        import matplotlib.pyplot as plt

        norm8, mask = normalize_for_display(arr if arr.ndim == 2 else arr[0])
        cmap_fn = cm.get_cmap(cmap or "viridis")
        rgba = cmap_fn(norm8 / 255.0)
        rgba8 = (rgba * 255).astype(np.uint8)
        rgba8[mask, 3] = 0

        # Create figure explicitly so we can strip text safely
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(rgba8)
        if textless:
            # Strip all text elements and decorations
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Convert to base64
        import base64
        from io import BytesIO

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def safe_make_img(arr: np.ndarray, cmap: str | None):
        """Try to render image normally; fallback with text removed."""
        try:
            return make_img_from_array(arr, cmap, textless=False)
        except Exception as e:
            print(f"⚠️ Rendering failed ({e}). Retrying without text...")
            try:
                return make_img_from_array(arr, cmap, textless=True)
            except Exception as e2:
                print(f"⚠️ Fallback also failed: {e2}")
                return ""

    if layer_name_or_band_index is not None:
        band = dataset.GetRasterBand(layer_name_or_band_index)
        arr = read_band_preview(band)
        chosen_cmap = choose_colormap(band)
        imgs.append(safe_make_img(arr, chosen_cmap))
    else:
        for i in range(1, bands + 1):
            band = dataset.GetRasterBand(i)
            arr = read_band_preview(band)
            chosen_cmap = choose_colormap(band)
            imgs.append(safe_make_img(arr, chosen_cmap))

    return imgs