from pathlib import Path
from typing import Optional

import fiona
import geopandas as gpd
from fiona.errors import DriverError
from osgeo import gdal
from pyogrio.errors import DataSourceError

from .analytics import (
    file_level_raster_stats,
    get_file_analytics,
    get_gdf_analytics,
    get_raster_analytics,
)
from .geospatial_schema_elements import file_level_bounding_box_in_wgs84
from .make_b64_img import raster_to_base64_images, to_base64_image
from .read_datafile import file_to_gdal, file_to_geodf
from .utils import catch_exceptions_as_warnings

gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
gdal.UseExceptions()

__all__ = [
    "read_and_enrich",
    "get_file_info",
    "total_bounding_box_in_wgs84",
]


# @catch_exceptions_as_warnings
# def gdf_to_dict(gdf: gpd.GeoDataFrame) -> dict:
#     """
#     Converts a GeoDataFrame to a dictionary representation.
#     Args:
#         gdf (gpd.GeoDataFrame): The GeoDataFrame to be converted.
#     Returns:
#         dict: A dictionary representation of the GeoDataFrame.
#     """
#     copying = gdf.copy(deep=True)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", UserWarning)
#         if 'geometry' in copying.columns:
#             copying["geometry"] = gdf.geometry.apply(lambda geom: geom.__geo_interface__ if geom else None)
#         return copying.to_json()


@catch_exceptions_as_warnings
def _get_crs_from_gdf(gdf: gpd.GeoDataFrame) -> dict:
    """
    Returns the coordinate reference system (CRS) of a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame whose CRS is to be retrieved.

    Returns:
        dict: A dictionary containing the CRS information.
    """
    if not hasattr(gdf, "crs") or gdf.crs is None:
        # warnings.warn("No CRS information found. The GeoDataFrame may not have a valid CRS.", UserWarning, stacklevel=2)
        return {}

    try:
        return gdf.crs.to_json_dict()
    except AttributeError:
        # warnings.warn("No CRS information found. The GeoDataFrame may not have a valid CRS.", UserWarning, stacklevel=2)
        return {}


# write a function to get the projection of the dataset
def _get_projection_from_raster(dataset: gdal.Dataset) -> str:
    """
    Get the projection of a GDAL dataset.

    Args:
        dataset (gdal.Dataset): GDAL dataset object.

    Returns:
        str: Projection in WKT format.

    Example:
    ```python
    dataset = gdal.Open("path/to/your/file.tif")
    projection = get_projection(dataset)
    print(projection)  # GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
    ```
    """
    # the ouput here is meaningless to me. It's annoying it's a string not a dict, I did try to find a dict version...
    projection = dataset.GetProjection()

    if not projection:
        return ""

    return projection


def read_and_enrich(
    filepath: str | Path,
    layer_name_or_band_index: str | int,
    categorical_allow_list: Optional[list[str]] = None,
    categorical_deny_list: Optional[list[str]] = None,
    return_object: bool = False,
) -> gpd.GeoDataFrame | dict:
    """
    Reads a file and enriches it with additional metadata.

    Args:
        filepath (str or Path): The path to the file to be read and enriched.
        layer_name_or_band_index (str|int, optional): The name of the layer or band index to read from the file.
            If None, the first layer or band will be used.
        return_object (bool): If True, returns the GeoDataFrame or GDAL dataset object along with the metadata.
            If False, returns only the metadata.

    Returns:
        dict or tuple: A dictionary containing the enriched metadata, and optionally the GeoDataFrame or GDAL dataset object.
    """
    try:
        # Attempt to read as vector data
        gdf = file_to_geodf(filepath, layer_name_or_band_index)
        crs = _get_crs_from_gdf(gdf)
        analytics = get_gdf_analytics(
            gdf,
            categorical_allow_list=categorical_allow_list or [],
            categorical_deny_list=categorical_deny_list or [],
        )
        img_string = to_base64_image(gdf) if not gdf.empty else None

        return_dict = {
            "crs": crs,
            "analytics": analytics,
            "img_strings": [img_string],
        }

        return (gdf, return_dict) if return_object else return_dict

    except DataSourceError:
        # Fallback to raster data
        dataset = file_to_gdal(filepath)
        projection = _get_projection_from_raster(dataset)
        analytics = get_raster_analytics(dataset, layer_name_or_band_index)
        img_strings = raster_to_base64_images(dataset, layer_name_or_band_index) or []

        return_dict = {
            "projection": projection,
            "analytics": analytics,
            "img_strings": img_strings,
        }

        return (dataset, return_dict) if return_object else return_dict


def get_file_info(filename) -> list[dict | str]:
    return_dict = {"file": get_file_analytics(filename)}

    try:
        layers = fiona.listlayers(filename)
        file_type = "vector"
    except DriverError:
        dataset = gdal.Open(filename)
        if not dataset:
            raise ValueError(f"Unable to open file: {filename}") from None
        file_type = "raster"
        layers = list(range(1, dataset.RasterCount + 1))
        return_dict["raster_stats"] = file_level_raster_stats(dataset)

    return_dict["layers"] = layers
    return_dict["type"] = file_type
    return return_dict


def _get_vector_or_raster(filename: str | Path) -> str:
    """
    Determine whether a file is a vector or raster dataset.

    Args:
        filename (str or Path): The path to the file.

    Returns:
        str: "vector" if the file is a vector dataset, "raster" if it is a raster dataset.

    Raises:
        ValueError: If the file type cannot be determined or the file cannot be opened.
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise ValueError(f"File does not exist: {filename}")

    try:
        if fiona.listlayers(filename):
            return "vector"
    except DriverError:
        pass

    dataset = gdal.Open(str(filename))
    if dataset:
        return "raster"

    raise ValueError(f"Unable to determine file type: {filename}")


def total_bounding_box_in_wgs84(filename: str | Path) -> dict:
    """
    Get the total bounding box for a file, ensuring it encompasses all layers.

    Args:
        filename (str or Path): The path to the file.

    Returns:
        dict: A dictionary containing the bounding box in WGS 84 with keys 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    file_type = _get_vector_or_raster(filename)
    if file_type == "vector":
        return file_level_bounding_box_in_wgs84(filename, "vector")
    return file_level_bounding_box_in_wgs84(filename, "raster")
