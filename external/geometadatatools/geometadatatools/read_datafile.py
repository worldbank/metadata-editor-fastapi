from contextlib import suppress
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
import pyogrio
from osgeo import gdal


def mif_to_gpd(filepath: Path, layer_name_or_band_index: str | int = None) -> gpd.GeoDataFrame:
    """
    Converts a MapInfo MIF file to a GeoDataFrame.

    Args:
        filepath (str): The path to the MIF file.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the geometries and attributes
        extracted from the MIF file.
    """
    with fiona.open(filepath, driver="MapInfo File", layer=layer_name_or_band_index) as src:
        # Convert Fiona collection to GeoDataFrame
        return gpd.GeoDataFrame.from_features(src, crs=src.crs)


# def unzip_file(filepath: Path, temp_dir: Path) -> Path:
#     """
#     Unzips a file into a temporary directory and returns the path to the first supported file.

#     Args:
#         filepath (Path): The path to the zip file.
#         temp_dir (Path): The temporary directory to extract files into.

#     Returns:
#         Path: The path to the first supported file.
#     """
#     import zipfile

#     # Unzip the file into the temporary directory
#     with zipfile.ZipFile(filepath, "r") as zip_ref:
#         zip_ref.extractall(temp_dir)

#     # Find the first file with a supported extension
#     list_of_files = [temp_dir / file for file in temp_dir.iterdir() if file.suffix in SUPPORTED_EXTENSIONS]
#     if not list_of_files:
#         raise ValueError(
#             f"No supported files found in the zip file: {filepath}. Supported extensions are {', '.join(SUPPORTED_EXTENSIONS)}"
#         )
#     return list_of_files[0]


def file_to_geodf(filepath: Path | str, layer_name_or_band_index: None | str | int = None) -> gpd.GeoDataFrame:
    """
    Converts a file to a GeoDataFrame.

    It reads in a vector file (GeoJSON, Shapefile, KML, GPKG, GML, DXF, MIF) or a zip file of a shapefile.


    Args:
        filepath (str or Path): The path to the file to be converted.
        layer_name_or_band_index (str|int, optional): The name of the layer or band index to read from the file.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the geometries and attributes
        extracted from the input file.

    Example:
    ```python
    filepath = "path/to/your/file.geojson"
    gdf = file_to_geodf(filepath)
    ```
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    if filepath.suffix == ".zip":
        try:
            gdf: gpd.GeoDataFrame = gpd.read_file(filepath, layer=layer_name_or_band_index)
        except pyogrio.errors.DataSourceError as err:
            raise NotImplementedError(
                "Will not read zip files except for shp files. Please extract the contents of the zip file manually."
            ) from err
    elif filepath.suffix == ".mif":
        gdf: gpd.GeoDataFrame = mif_to_gpd(filepath)
    else:
        gdf: gpd.GeoDataFrame = gpd.read_file(filepath, layer=layer_name_or_band_index)

    for col in gdf.columns:
        if gdf[col].dtype == "object":  # Check if the column is of type object (string)
            with suppress(ValueError, TypeError):
                gdf[col] = pd.to_numeric(gdf[col].str.replace(",", "").str.replace("%", ""), errors="raise")
    return gdf


def file_to_gdal(filepath: Path | str) -> gdal.Dataset:
    """
    Convert a file to GDAL format.

    Args:
        filepath (str): Path to the input file.

    Returns:
        gdal.Dataset: GDAL dataset object.

    Raises:
        ValueError: If the file cannot be opened.


    """
    # Open the file using GDAL
    dataset = gdal.Open(filepath)

    if dataset is None:
        raise ValueError(f"Could not open file: {filepath}")

    return dataset


def vector_to_dataframe(
    filepath: Path | str, 
    layer_name: None | str = None,
    include_geometry: bool = True,
    geometry_format: str = "wkt"
) -> pd.DataFrame:
    """
    Convert vector geospatial data to a pandas DataFrame.
    
    This function wraps file_to_geodf() and provides options for geometry handling
    to make the data suitable for CSV export or other DataFrame operations.

    Args:
        filepath (str or Path): The path to the vector geospatial file.
        layer_name (str, optional): The name of the layer to extract.
            If None, uses the first layer.
        include_geometry (bool): Whether to include geometry column.
        geometry_format (str): Format for geometry column - "wkt", "geojson", or "exclude".

    Returns:
        pd.DataFrame: A DataFrame containing the vector data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as vector data.
        
    Example:
        ```python
        # Extract with geometry as WKT
        df = vector_to_dataframe("path/to/file.geojson", include_geometry=True, geometry_format="wkt")
        
        # Extract without geometry
        df = vector_to_dataframe("path/to/file.shp", include_geometry=False)
        
        # Extract specific layer
        df = vector_to_dataframe("path/to/file.gpkg", layer_name="buildings")
        ```
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    # Use existing file_to_geodf function
    gdf = file_to_geodf(filepath, layer_name)
    
    # Convert GeoDataFrame to DataFrame based on geometry requirements
    if include_geometry and 'geometry' in gdf.columns:
        gdf_copy = gdf.copy()
        
        if geometry_format == "wkt":
            gdf_copy['geometry'] = gdf_copy['geometry'].apply(
                lambda x: x.wkt if x is not None else None
            )
        elif geometry_format == "geojson":
            gdf_copy['geometry'] = gdf_copy['geometry'].apply(
                lambda x: x.__geo_interface__ if x is not None else None
            )
        # If geometry_format is "exclude", we'll drop it below
        
        df = pd.DataFrame(gdf_copy)
        
        if geometry_format == "exclude":
            df = df.drop(columns=['geometry'])
    else:
        # Drop geometry column
        df = pd.DataFrame(gdf.drop(columns=['geometry'] if 'geometry' in gdf.columns else []))
        
    return df


def raster_to_dataframe(
    filepath: Path | str, 
    band_index: int = 1,
    include_coordinates: bool = True,
    exclude_nodata: bool = True
) -> pd.DataFrame:
    """
    Convert raster data to a pandas DataFrame.
    
    Extracts pixel values with optional coordinates for CSV export or analysis.

    Args:
        filepath (str or Path): The path to the raster file.
        band_index (int): The band index to extract (1-based).
        include_coordinates (bool): Whether to include x, y coordinate columns.
        exclude_nodata (bool): Whether to exclude NoData values from the output.

    Returns:
        pd.DataFrame: A DataFrame containing the raster data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as raster data.
        
    Example:
        ```python
        # Extract band 1 with coordinates
        df = raster_to_dataframe("path/to/file.tif", band_index=1)
        
        # Extract band 2 without coordinates
        df = raster_to_dataframe("path/to/file.nc", band_index=2, include_coordinates=False)
        ```
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    # Use existing file_to_gdal function
    dataset = file_to_gdal(filepath)
    
    if band_index < 1 or band_index > dataset.RasterCount:
        raise ValueError(f"Band index {band_index} out of range. Available bands: 1-{dataset.RasterCount}")
    
    # Get raster properties
    band = dataset.GetRasterBand(band_index)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    # Read band data as numpy array
    data_array = band.ReadAsArray()
    
    # Get NoData value
    nodata_value = band.GetNoDataValue()
    
    # Create data lists
    values = []
    x_coords = []
    y_coords = []
    
    if include_coordinates:
        # Get geotransform for coordinate calculation
        geotransform = dataset.GetGeoTransform()
    
    for row in range(height):
        for col in range(width):
            value = data_array[row, col]
            
            # Skip NoData values if requested
            if exclude_nodata and nodata_value is not None and value == nodata_value:
                continue
            
            values.append(value)
            
            if include_coordinates:
                # Convert pixel coordinates to geographic coordinates
                x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
                y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
                x_coords.append(x)
                y_coords.append(y)
    
    # Create DataFrame
    if include_coordinates:
        df = pd.DataFrame({
            'x': x_coords,
            'y': y_coords,
            f'band_{band_index}_value': values
        })
    else:
        df = pd.DataFrame({
            f'band_{band_index}_value': values
        })
    
    return df
