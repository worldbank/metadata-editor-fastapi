import os
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from osgeo import gdal

from .geospatial_schema_elements import layer_level_geographic_element_in_wgs84_from_gdf

# from .utils import catch_exceptions_as_warnings


def _is_categorical(gdf, col) -> bool:
    """
    Determines if a column is categorical based on the proportion of unique values.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the column.
        col (str): The name of the column to check.
        threshold (float): The maximum proportion of unique values to consider the column categorical.

    Returns:
        bool: True if the column is categorical, False otherwise.
    """
    unique_count = gdf[col].nunique()
    non_null_count = gdf[col].notna().sum()
    return bool(unique_count <= 100 and unique_count < non_null_count)

    # # Avoid division by zero
    # if unique_count == 0:
    #     return False
    # elif non_null_count <= 400:
    #     return unique_count <= 20 and unique_count < non_null_count
    # else:

    #     # Calculate the proportion of unique values
    #     proportion_unique = unique_count / non_null_count

    #     # Check if the proportion is below the threshold
    #     return proportion_unique <= 0.05


# @catch_exceptions_as_warnings
def _summarize_column(gdf, col, categorical_allow_list: list[str], categorical_deny_list: list[str]) -> dict:
    """
    Summarizes a column in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the column to summarize.
        col (str): The name of the column to summarize.
        categorical_allow_list (list): List of columns to treat as categorical.
        categorical_deny_list (list): List of columns to ignore as categorical.

    Returns:
        dict: A dictionary containing the summary of the column.
    """
    sub_analysis = {}
    sub_analysis["valid"] = int(gdf[col].notna().sum())  # convert to into from np.int64 so it can be serialized
    if sub_analysis["valid"] == 0:
        return sub_analysis
    if (
        (pd.api.types.is_object_dtype(gdf[col]) or pd.api.types.is_integer_dtype(gdf[col]))
        and (_is_categorical(gdf, col) or col in categorical_allow_list)
        and col not in categorical_deny_list
    ):
        frequencies = gdf[col].value_counts().to_frame().sort_index()
        frequencies["percentage"] = frequencies / frequencies.sum() * 100
        sub_analysis["frequencies"] = frequencies.to_dict("index")
        sub_analysis["data_dictionary"] = gdf[col].dropna().unique().tolist()
        sub_analysis["number_of_categories"] = len(sub_analysis["data_dictionary"])
    elif col in categorical_deny_list and col in categorical_allow_list:
        warnings.warn(
            f"Column {col} is in the deny list but is also in the allow list. Consider removing it from one or the other.",
            stacklevel=2,
        )
    elif col in categorical_allow_list:
        warnings.warn(
            f"Column {col} is not categorical, it's {gdf[col].dtype} but is in the allow list. Consider removing it from the allow list.",
            stacklevel=2,
        )
    elif pd.api.types.is_numeric_dtype(gdf[col]):
        # calculate basic statistics
        sub_analysis["summary_statistics"] = {
            "mean": float(gdf[col].mean()),
            "median": float(gdf[col].median()),
            "std_dev": float(gdf[col].std()),
            "min": float(gdf[col].min()),
            "max": float(gdf[col].max()),
        }

    return sub_analysis


# @catch_exceptions_as_warnings
def band_stats(dataset: gdal.Dataset, band_index: int) -> dict:
    """
    Summarize a specific band of a GDAL dataset.

    Args:
        dataset (gdal.Dataset): GDAL dataset object.
        band_index (int): Index of the band to summarize.

    Returns:
        dict: Dictionary containing band statistics and metadata.
    """
    band = dataset.GetRasterBand(band_index)
    if not band:
        return {}

    try:
        band_stats = band.GetStatistics(0, 1)
    except RuntimeError as e:
        warnings.warn(f"Failed to get statistics for band {band_index}: {e}", UserWarning, stacklevel=2)
        band_stats = [None, None, None, None]
    if band_stats is None:
        warnings.warn(f"Band {band_index} has no statistics.", UserWarning, stacklevel=2)
        band_stats = [None, None, None, None]
    band_dict = dict(zip(["min", "max", "mean", "std_dev"], band.GetStatistics(0, 1)))

    return {
        "band_index": band_index,
        "summary_statistics": band_dict,
        "metadata": band.GetMetadata_Dict(),
        "nodata_value": band.GetNoDataValue(),
        "unit_type": band.GetUnitType(),
        # what is a RasterColorTable?
    }


# @catch_exceptions_as_warnings
def get_file_size(filepath: str | Path) -> dict:
    """
    Function to get the file size in a human readable format.

    Parameters:
    filepath (str): Path to the file.

    Returns:
    dict: Dictionary with the file size and the unit.
    """
    file_size = os.path.getsize(filepath)
    if file_size < 1024:
        return {"size": file_size, "unit": "bytes"}
    if file_size < 1024 * 1024:
        return {"size": file_size / 1024, "unit": "KB"}
    if file_size < 1024 * 1024 * 1024:
        return {"size": file_size / (1024 * 1024), "unit": "MB"}
    return {"size": file_size / (1024 * 1024 * 1024), "unit": "GB"}


# @catch_exceptions_as_warnings
def get_file_analytics(filepath: str | Path) -> dict:
    """
    Function to get the file analytics of a file.
    Args:
        filepath (str): Path to the file.
        file_type (str): Type of the file. Can be 'raster' or 'vector'.
    Returns:
        dict: Dictionary with the file analytics.
    """

    # assert file_type in ["raster", "vector"], "file_type must be either 'raster' or 'vector'"
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return_dict = {}
    file_size = get_file_size(filepath)
    return_dict["file_size"] = file_size
    return_dict["file_name"] = filepath.name
    # extension
    return_dict["file_extension"] = filepath.suffix
    # creation date
    return_dict["creation_date"] = datetime.fromtimestamp(filepath.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
    # modification date
    return_dict["modification_date"] = datetime.fromtimestamp(filepath.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    # last access date
    return_dict["last_access_date"] = datetime.fromtimestamp(filepath.stat().st_atime).strftime("%Y-%m-%d %H:%M:%S")
    return return_dict


# @catch_exceptions_as_warnings
def layer_stats(d: gpd.GeoDataFrame | gdal.Band, file_type: str) -> dict:
    """
    Function to get the top level statistics of a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to analyze.
        type (str): The type of the file. Can be 'vector' or 'raster'.

    Returns:
        dict: A dictionary containing the top level statistics.
    """
    assert file_type in ["vector", "raster"], "type must be either 'vector' or 'raster'"
    # bounding_box = gdf.total_bounds
    # # Create a dictionary with the bounding box coordinates
    # bounding_box_dict = {
    #     "minx": float(bounding_box[0]),
    #     "miny": float(bounding_box[1]),
    #     "maxx": float(bounding_box[2]),
    #     "maxy": float(bounding_box[3]),
    # }
    # check if gdf has an attribute named "crs"
    if file_type == "vector" and hasattr(d, "crs"):
        # get name of layer
        geographic_element = layer_level_geographic_element_in_wgs84_from_gdf(d)
        return {"rows": len(d), "columns": len(d.columns), **geographic_element}
    if file_type == "vector":
        return {"rows": len(d), "columns": len(d.columns)}
    return {"rows": d.RasterYSize, "columns": d.RasterXSize}


# @catch_exceptions_as_warnings
def file_level_raster_stats(dataset: gdal.Dataset) -> dict:
    """
    Get top-level statistics of a GDAL dataset.

    Args:
        dataset (gdal.Dataset): GDAL dataset object.

    Returns:
        dict: Dictionary containing the number of bands, rows, columns, and projection.
    """
    # geotransform = dataset.GetGeoTransform()
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # # bounding box
    # min_x = geotransform[0]
    # max_x = geotransform[0] + geotransform[1] * width
    # min_y = geotransform[3] + geotransform[5] * height
    # max_y = geotransform[3]
    # # bounding_box_dict = {"minx": min_x, "miny": min_y, "maxx": max_x, "maxy": max_y}

    return {
        "bands": dataset.RasterCount,
        "rows": height,
        "cols": width,
        # "bounding_box": bounding_box_dict,
        **dataset.GetMetadata(),
    }


def get_gdf_analytics(
    gdf: gpd.GeoDataFrame, categorical_allow_list: list[str], categorical_deny_list: list[str]
) -> dict:
    """
    Analyzes the variables in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to analyze.
        categorical_allow_list (list): List of columns to treat as categorical.
        categorical_deny_list (list): List of columns to ignore as categorical.

    Returns:
        dict: A dictionary containing the analysis of the variables.
    """
    analysis = {}
    # analysis["file"] = get_file_analytics(filepath, "vector")

    analysis["layer"] = layer_stats(gdf, "vector")

    analysis["feature_types"] = {col: str(dtype) for col, dtype in gdf.dtypes.items()}

    features = {}
    for col in gdf.columns:
        if col != "geometry":
            features[col] = _summarize_column(gdf, col, categorical_allow_list, categorical_deny_list)
    analysis["feature_statistics"] = features

    return analysis


def get_raster_analytics(dataset: gdal.Dataset, layer_name_or_band_index: str | int) -> dict:
    """
    Analyzes the variables in a raster dataset.

    Args:
        dataset (gdal.Dataset): The raster dataset to analyze.
        layer_name_or_band_index (None|str|int, optional): The name of the layer or band index to read from the file.
            If None, the first layer or band will be used.

    Returns:
        dict: A dictionary containing the analysis of the variables.
    """
    analysis = {}

    # features = {}
    # if layer_name_or_band_index is None:
    #     for band_index in range(1, dataset.RasterCount + 1):
    #         features[f"{band_index}"] = _summarize_band(dataset, band_index)
    # else:
    #     features[f"{layer_name_or_band_index}"] = _summarize_band(dataset, layer_name_or_band_index)
    analysis["feature_statistics"] = band_stats(dataset, layer_name_or_band_index)

    return analysis
