from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class GeospatialFileType(str, Enum):
    """Supported geospatial file types"""
    SHAPEFILE = "shapefile"
    GEOJSON = "geojson"
    KML = "kml"
    KMZ = "kmz"
    GPKG = "gpkg"
    GDB = "gdb"
    NETCDF = "netcdf"
    TIF = "tif"
    BIL = "bil"
    ASCII = "ascii"
    PDF = "pdf"
    JPG = "jpg"
    GIF = "gif"
    ADF = "adf"
    OVR = "ovr"


class GeospatialImportRequest(BaseModel):
    """Request model for importing geospatial data"""
    file_path: str = Field(..., description="Path to the geospatial file")
    layer_name: Optional[str] = Field(None, description="Layer name for multi-layer formats")
    band_index: Optional[int] = Field(None, description="Band index for raster data")
    return_object: bool = Field(False, description="Whether to return the actual data object")


class GeospatialMetadataRequest(BaseModel):
    """Request model for geospatial metadata extraction"""
    file_path: str = Field(..., description="Path to the geospatial file")
    layer_name_or_band_index: Optional[Union[str, int]] = Field(None, description="Layer name or band index")
    categorical_allow_list: Optional[List[str]] = Field(None, description="List of columns to force-treat as categorical")
    categorical_deny_list: Optional[List[str]] = Field(None, description="List of columns to exclude from categorical analysis")
    return_object: bool = Field(False, description="Whether to return the actual data object along with metadata")


class GeospatialBatchMetadataRequest(BaseModel):
    """Request model for batch geospatial metadata extraction"""
    file_path: str = Field(..., description="Path to the geospatial file")
    layer_names_or_band_indices: Optional[List[Union[str, int]]] = Field(None, description="List of layer names or band indices to process. If None, processes all layers.")
    categorical_allow_list: Optional[List[str]] = Field(None, description="List of columns to force-treat as categorical")
    categorical_deny_list: Optional[List[str]] = Field(None, description="List of columns to exclude from categorical analysis")
    return_object: bool = Field(False, description="Whether to return the actual data objects along with metadata")


class GeospatialMetadataResponse(BaseModel):
    """Response model for geospatial metadata"""
    file_info: Dict[str, Any] = Field(..., description="Basic file information")
    layers: Optional[List[str]] = Field(None, description="Available layers in the file")
    type: str = Field(..., description="File type")
    bounding_box: Optional[Dict[str, float]] = Field(None, description="Bounding box in WGS84")
    raster_stats: Optional[Dict[str, Any]] = Field(None, description="Raster statistics if applicable")
    analytics: Optional[Dict[str, Any]] = Field(None, description="Analytical information")
    status: str = Field("success", description="Operation status")


class GeospatialTransformRequest(BaseModel):
    """Request model for coordinate system transformations"""
    file_path: str = Field(..., description="Path to the geospatial file")
    source_crs: str = Field(..., description="Source coordinate reference system")
    target_crs: str = Field(..., description="Target coordinate reference system")
    layer_name: Optional[str] = Field(None, description="Layer name for multi-layer formats")


class GeospatialClipRequest(BaseModel):
    """Request model for spatial clipping operations"""
    file_path: str = Field(..., description="Path to the geospatial file")
    clip_geometry: Dict[str, Any] = Field(..., description="Clipping geometry in GeoJSON format")
    layer_name: Optional[str] = Field(None, description="Layer name for multi-layer formats")


class GeospatialPreviewRequest(BaseModel):
    """Request model for generating preview images"""
    file_path: str = Field(..., description="Path to the geospatial file")
    layer_name: Optional[str] = Field(None, description="Layer name for multi-layer formats")
    band_index: Optional[int] = Field(None, description="Band index for raster data")
    output_format: str = Field("png", description="Output image format")
    width: int = Field(800, description="Output image width")
    height: int = Field(600, description="Output image height")


class GeospatialJobResponse(BaseModel):
    """Response model for queued geospatial operations"""
    message: str = Field(..., description="Status message")
    job_id: str = Field(..., description="Unique job identifier")
    operation_type: str = Field(..., description="Type of geospatial operation")


class GeospatialMetadataJobResponse(BaseModel):
    """Response model for queued geospatial metadata extraction"""
    message: str = Field(..., description="Status message")
    job_id: str = Field(..., description="Unique job identifier")
    operation_type: str = Field("geospatial-metadata-extraction", description="Type of geospatial operation")
    file_path: str = Field(..., description="Path to the file being processed")
    parameters: Dict[str, Any] = Field(..., description="Processing parameters")


class GeospatialBatchJobResponse(BaseModel):
    """Response model for queued batch geospatial metadata extraction"""
    message: str = Field(..., description="Status message")
    job_id: str = Field(..., description="Unique job identifier")
    operation_type: str = Field("geospatial-batch-metadata-extraction", description="Type of geospatial operation")
    file_path: str = Field(..., description="Path to the file being processed")
    total_layers: int = Field(..., description="Total number of layers being processed")
    parameters: Dict[str, Any] = Field(..., description="Processing parameters")


class GeospatialErrorResponse(BaseModel):
    """Error response model for geospatial operations"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class GeospatialFileInfoResponse(BaseModel):
    """Response model for geospatial file information"""
    file_path: str = Field(..., description="Path to the geospatial file")
    file_info: Dict[str, Any] = Field(..., description="Basic file information")
    type: str = Field(..., description="File type (vector or raster)")
    layers: List[Union[str, int]] = Field(..., description="Available layers or bands")
    bounding_box: Optional[Dict[str, float]] = Field(None, description="Overall bounding box in WGS84")
    raster_stats: Optional[Dict[str, Any]] = Field(None, description="Raster statistics if applicable")
    processing_recommendations: Dict[str, Any] = Field(..., description="Recommendations for processing")
    status: str = Field("success", description="Operation status")


class GeospatialDataExtractionRequest(BaseModel):
    """Request model for geospatial data extraction to CSV"""
    file_path: str = Field(..., description="Path to the geospatial file")
    layer_name_or_band_index: Optional[str] = Field(None, description="Layer name (vector) or band index (raster)")
    csv_output_path: str = Field(..., description="Path where the CSV file will be written")


class GeospatialDataExtractionJobResponse(BaseModel):
    """Response model for queued geospatial data extraction"""
    message: str = Field(..., description="Status message")
    job_id: str = Field(..., description="Unique job identifier")
    operation_type: str = Field("geospatial-data-extraction", description="Type of geospatial operation")
    file_path: str = Field(..., description="Path to the file being processed")
    parameters: Dict[str, Any] = Field(..., description="Processing parameters")
