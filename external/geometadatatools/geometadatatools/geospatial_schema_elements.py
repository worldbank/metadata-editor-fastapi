import warnings

import fiona
import geopandas as gpd
import pygeohash
from osgeo import gdal, osr
from shapely.geometry import box

from .utils import catch_exceptions_as_warnings


def geohash_calculator(bounding_box: dict[str, float]) -> str:
    """ "
    Generate a geohash for a given bounding box.

    Args:
        bounding_box (dict): A dictionary containing the bounding box coordinates with keys 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        str: The geohash for the bounding box.

    """
    centroid_lat = (bounding_box["ymin"] + bounding_box["ymax"]) / 2
    centroid_lon = (bounding_box["xmin"] + bounding_box["xmax"]) / 2
    return pygeohash.encode(centroid_lat, centroid_lon)


def file_level_vector_bounding_box_in_wgs84(file_path: str) -> dict:
    """
    Get the bounding box for a whole vector file, ensuring it encompasses all layers.

    Args:
        file_path (str): Path to the vector file.

    Returns:
        dict: A dictionary containing the bounding box in WGS 84 with keys 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    # Initialize variables to store the overall bounding box
    overall_bounds = None

    layers = fiona.listlayers(file_path)
    if not layers:
        return {}

    for layer in layers:
        # Read the current layer as a GeoDataFrame
        gdf = gpd.read_file(file_path, layer=layer)

        # Get the bounding box for the current layer in WGS 84
        layer_bounds = layer_level_vector_bounding_box_in_wgs84(gdf)
        if len(layer_bounds) == 0:
            continue

        # Update the overall bounding box
        if overall_bounds is None:
            overall_bounds = layer_bounds
        else:
            overall_bounds = {
                "xmin": min(overall_bounds["xmin"], layer_bounds["xmin"]),
                "ymin": min(overall_bounds["ymin"], layer_bounds["ymin"]),
                "xmax": max(overall_bounds["xmax"], layer_bounds["xmax"]),
                "ymax": max(overall_bounds["ymax"], layer_bounds["ymax"]),
            }

    if overall_bounds is None:
        return {}
    return overall_bounds


def file_level_raster_bounding_box_in_wgs84(raster_file: str) -> dict:
    """
    Get the bounding box of a raster file projected into WGS 84 (EPSG:4326).

    Args:
        raster_file (str): Path to the raster file.

    Returns:
        dict: A dictionary containing the bounding box in WGS 84 with keys 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    # Open the raster file
    dataset = gdal.Open(raster_file)
    if not dataset:
        raise ValueError(f"Unable to open raster file: {raster_file}")

    try:
        # Get the geotransform and projection
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # Get the spatial reference of the raster
        source_srs = osr.SpatialReference()
        source_srs.ImportFromWkt(projection)

        # Define the target spatial reference (WGS 84)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

        # Create a coordinate transformation
        transform = osr.CoordinateTransformation(source_srs, target_srs)
    except RuntimeError:
        return {}

    # Get the raster bounds in the source CRS
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = x_min + geotransform[1] * dataset.RasterXSize
    y_min = y_max + geotransform[5] * dataset.RasterYSize

    # Transform the corners to WGS 84
    lower_left = transform.TransformPoint(x_min, y_min)
    upper_left = transform.TransformPoint(x_min, y_max)
    lower_right = transform.TransformPoint(x_max, y_min)
    upper_right = transform.TransformPoint(x_max, y_max)

    # Calculate the bounding box in WGS 84
    xmin = min(lower_left[0], upper_left[0], lower_right[0], upper_right[0])
    ymin = min(lower_left[1], upper_left[1], lower_right[1], upper_right[1])
    xmax = max(lower_left[0], upper_left[0], lower_right[0], upper_right[0])
    ymax = max(lower_left[1], upper_left[1], lower_right[1], upper_right[1])

    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def layer_level_vector_bounding_box_in_wgs84(gdf: gpd.GeoDataFrame) -> dict:
    """
    Project the bounding box of a GeoDataFrame to WGS 84 (EPSG:4326).

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to calculate and project the bounding box for.

    Returns:
        dict: A dictionary containing the projected bounding box coordinates.
    """
    if not hasattr(gdf, "total_bounds") or gdf.total_bounds is None:
        return {}
    # Get the bounding box
    xmin, ymin, xmax, ymax = gdf.total_bounds

    if not gdf.crs:
        warnings.warn(
            "GeoDataFrame has an invalid or null CRS. Please set a valid CRS before proceeding.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    bbox = gpd.GeoDataFrame({"geometry": [box(xmin, ymin, xmax, ymax)]}, crs=gdf.crs)
    # Project to WGS 84
    bbox_wgs84 = bbox.to_crs(epsg=4326)
    projected_bounds = bbox_wgs84.total_bounds
    return {
        "xmin": float(projected_bounds[0]),
        "ymin": float(projected_bounds[1]),
        "xmax": float(projected_bounds[2]),
        "ymax": float(projected_bounds[3]),
    }


def file_level_bounding_box_in_wgs84(file_path: str, file_type: str) -> dict:
    """
    Get the bounding box for a whole file, ensuring it encompasses all layers.

    Args:
        file_path (str): Path to the vector file.
        type (str): Type of the file ('vector' or 'raster').

    Returns:
        dict: A dictionary containing the bounding box in WGS 84 with keys 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    if file_type == "vector":
        bounds = file_level_vector_bounding_box_in_wgs84(file_path)
    elif file_type == "raster":
        bounds = file_level_raster_bounding_box_in_wgs84(file_path)
    else:
        raise ValueError("Unsupported file type. Supported types are 'vector' and 'raster'.")
    if len(bounds) == 0:
        return {}
    return bounds_to_geoelement(bounds)


def bounds_to_geoelement(bounds: dict[str, float]) -> dict:
    """
    Convert bounding box coordinates to a GeographicElementItem.

    Args:
        bounds (dict): A dictionary containing the bounding box coordinates with keys 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        dict: A dictionary containing the GeographicElementItem.
    """
    return {
        "geographicBoundingBox": {
            "westBoundLongitude": bounds["xmin"],
            "eastBoundLongitude": bounds["xmax"],
            "southBoundLatitude": bounds["ymin"],
            "northBoundLatitude": bounds["ymax"],
        },
        "geohash": geohash_calculator(bounds),
    }


@catch_exceptions_as_warnings
def layer_level_geographic_element_in_wgs84_from_gdf(gdf: gpd.GeoDataFrame) -> dict:
    """
    Creates a GeographicElementItem (geographicBoundingBox and geohash) over the whole GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to create the GeographicElementItem for.

    Returns:
        dict: A dictionary containing the GeographicElementItem.

    Example:
        ```python
        gdf = gpd.read_file("path/to/your/file.geojson")
        geographic_element = make_geographic_element_gdf(gdf.geometry)
        print(geographic_element)  # {'geographicBoundingBox': {'westBoundLongitude': -180.0, 'eastBoundLongitude': 180.0, 'southBoundLatitude': -90.0, 'northBoundLatitude': 90.0}, 'geohash': 'gcpuv'}
        ```
    """
    projected_bounds = layer_level_vector_bounding_box_in_wgs84(gdf)
    if len(projected_bounds) == 0:
        return {}
    return bounds_to_geoelement(projected_bounds)


# def make_geographic_elements_for_gdf(gdf: gpd.GeoDataFrame) -> list[dict]:
#     """
#     Creates a GeographicElementItem for each geometry in the GeoDataFrame.

#     Args:
#         gdf (GeoDataFrame): The GeoDataFrame containing the geometries.

#     Returns:
#         dict: A dictionary containing the GeographicElementItems for each geometry.
#     """
#     geographic_elements = []
#     for _, row in gdf.iterrows():
#         # Create a GeographicElementItem for each geometry
#         if 'geometry' not in row:
#             geographic_elements.append({})
#         else:
#             geographic_element = make_geographic_element_item(row.geometry)
#             # Add the GeographicElementItem to the dictionary
#             geographic_elements.append(geographic_element)

#     return geographic_elements


# def make_geographic_elements_for_raster(dataset: gdal.Dataset) -> list[dict]:
#     """
#     Creates a GeographicElementItem for a raster dataset.

#     Args:
#         dataset (gdal.Dataset): The GDAL dataset.

#     Returns:
#         list: A list of dictionaries containing the GeographicElementItems.
#     """
#     geotransform = dataset.GetGeoTransform()
#     width = dataset.RasterXSize
#     height = dataset.RasterYSize
#     min_x = geotransform[0]
#     max_x = geotransform[0] + geotransform[1] * width
#     min_y = geotransform[3] + geotransform[5] * height
#     max_y = geotransform[3]

#     # Create a polygon from the bounding box
#     band_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)])
#     return [make_geographic_element_item(band_polygon)]
