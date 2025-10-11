# GeoMetadataTools

GeoMetadataTools is a Python library designed to extract, analyze, and visualize metadata from geospatial datasets. It supports both raster and vector formats, providing enriched metadata, bounding box calculations, and top-level visualizations.

## Installation

To install GeoMetadataTools, use pip:

```bash
pip install geometadatatools
```

## Supported Formats

- **Vector**: GeoJSON, Shapefile (SHP), KML, GPKG, GDB
- **Raster**: NetCDF (NC), TIF, BIL, ASCII (XYZ), GeoPDF (PDF), JPG, GIF, ADF, OVR

---

## Usage

### Importing the Library

```python
from geometadatatools import get_file_info, read_and_enrich, total_bounding_box_in_wgs84
```

---

## Examples

### Vector Data

#### Example: GeoJSON File

```python
# Inspect the file
file_info = get_file_info("tests/sample_files/Local_Nature_Reserves_England_3570640822803777921.geojson")
print(file_info["file"])  # Basic file metadata
print(file_info["layers"])  # List of layers
print(file_info["type"])  # File type

# Calculate the total bounding box
bbox = total_bounding_box_in_wgs84("tests/sample_files/Local_Nature_Reserves_England_3570640822803777921.geojson")
print(bbox)

# Enrich the file
gdf, info = read_and_enrich(
    "tests/sample_files/Local_Nature_Reserves_England_3570640822803777921.geojson",
    return_object=True,
    layer_name_or_band_index="Local_Nature_Reserves_England_3570640822803777921"
)
print(info["analytics"]["feature_statistics"])
```

#### Example: GPKG File

```python
# Inspect the file
file_info = get_file_info("tests/sample_files/example.gpkg")
print(file_info["layers"])  # List of layers

# Calculate the total bounding box
bbox = total_bounding_box_in_wgs84("tests/sample_files/example.gpkg")
print(bbox)

# Enrich each layer
for layer in file_info["layers"]:
    gdf, info = read_and_enrich("tests/sample_files/example.gpkg", layer_name_or_band_index=layer, return_object=True)
    print(info["analytics"]["feature_statistics"])
```

---

### Raster Data

#### Example: NetCDF File

```python
# Inspect the file
file_info = get_file_info("tests/sample_files/aep-tccat1-annual-mean_chaz-x0.5_chaz-ensemble-all-ssp245_climatology_p10_2035-2064.nc")
print(file_info["file"])  # Basic file metadata
print(file_info["raster_stats"])  # Raster statistics
print(file_info["layers"])  # List of bands

# Calculate the total bounding box
bbox = total_bounding_box_in_wgs84("tests/sample_files/aep-tccat1-annual-mean_chaz-x0.5_chaz-ensemble-all-ssp245_climatology_p10_2035-2064.nc")
print(bbox)

# Enrich the file
dataset, info = read_and_enrich(
    "tests/sample_files/aep-tccat1-annual-mean_chaz-x0.5_chaz-ensemble-all-ssp245_climatology_p10_2035-2064.nc",
    return_object=True,
    layer_name_or_band_index=1
)
print(info["analytics"]["feature_statistics"])
```

#### Example: TIF File

```python
# Inspect the file
file_info = get_file_info("tests/sample_files/swz_dmsp_100m_2000.tif")
print(file_info["file"])  # Basic file metadata
print(file_info["raster_stats"])  # Raster statistics
print(file_info["layers"])  # List of bands

# Calculate the total bounding box
bbox = total_bounding_box_in_wgs84("tests/sample_files/swz_dmsp_100m_2000.tif")
print(bbox)

# Enrich the file
dataset, info = read_and_enrich(
    "tests/sample_files/swz_dmsp_100m_2000.tif",
    return_object=True,
    layer_name_or_band_index=1
)
print(info["analytics"]["feature_statistics"])
```

---

## Key Functions

### `get_file_info`

Extracts metadata from a file, including file size, type, layers, and raster statistics (if applicable).

### `read_and_enrich`

Reads a specific layer or band from the file and enriches it with detailed metadata, analytics, and visualizations.

### `total_bounding_box_in_wgs84`

Calculates the total bounding box of the file in WGS84 projection.

---

## Acknowledgments

Parts of this repository were written with assistance from Large Language Models (LLMs).