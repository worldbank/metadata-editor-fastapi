import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
from fastapi import HTTPException

# Add the external geometadatatools library to Python path
external_path = os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'geometadatatools')
if external_path not in sys.path:
    sys.path.insert(0, external_path)

# Conditional imports for geospatial dependencies
try:
    from geometadatatools import get_file_info, read_and_enrich, total_bounding_box_in_wgs84
    GEOSPATIAL_PACKAGES_AVAILABLE = True
except ImportError as e:
    GEOSPATIAL_PACKAGES_AVAILABLE = False
    get_file_info = None
    read_and_enrich = None
    total_bounding_box_in_wgs84 = None
    logger = logging.getLogger(__name__)
    logger.warning(f"Geospatial packages not available: {e}")

logger = logging.getLogger(__name__)


class GeospatialService:
    """Service class for handling geospatial operations using geometadatatools"""
    
    def __init__(self):
        """Initialize the geospatial service"""
        self.available = GEOSPATIAL_PACKAGES_AVAILABLE
        self.supported_formats = {
            'vector': ['geojson', 'shp', 'kml', 'kmz', 'gpkg', 'gdb'],
            'raster': ['nc', 'tif', 'bil', 'ascii', 'pdf', 'jpg', 'gif', 'adf', 'ovr']
        }
        
        # Set matplotlib backend to non-interactive to prevent threading issues
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib to work in background threads"""
        try:
            import matplotlib
            # Set backend to non-interactive to prevent GUI issues in background threads
            matplotlib.use('Agg')  # Non-interactive backend
            logger.info("Matplotlib configured with 'Agg' backend for background processing")
        except ImportError:
            logger.warning("Matplotlib not available, image generation will be disabled")
        except Exception as e:
            logger.warning(f"Could not configure matplotlib backend: {e}")
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic metadata for a geospatial file
        
        Args:
            file_path: Path to the geospatial file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            logger.info(f"Getting metadata for file: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file info using geometadatatools
            file_info = get_file_info(file_path)
            
            logger.info(f"Successfully retrieved metadata for {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            raise
    
    def get_bounding_box(self, file_path: str) -> Dict[str, float]:
        """
        Get the bounding box of a geospatial file in WGS84 coordinates
        
        Args:
            file_path: Path to the geospatial file
            
        Returns:
            Dictionary containing bounding box coordinates
        """
        try:
            logger.info(f"Getting bounding box for file: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get bounding box using geometadatatools
            bbox = total_bounding_box_in_wgs84(file_path)
            
            logger.info(f"Successfully retrieved bounding box for {file_path}")
            return bbox
            
        except Exception as e:
            logger.error(f"Error getting bounding box for {file_path}: {str(e)}")
            raise
    
    def enrich_file(self, file_path: str, layer_name: Optional[str] = None, 
                   band_index: Optional[int] = None, return_object: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Enrich a geospatial file with additional metadata and analytics
        
        Args:
            file_path: Path to the geospatial file
            layer_name: Layer name for multi-layer formats
            band_index: Band index for raster data
            return_object: Whether to return the actual data object
            
        Returns:
            Tuple of (data_object, enriched_info)
        """
        try:
            logger.info(f"Enriching file: {file_path}, layer: {layer_name}, band: {band_index}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine layer/band parameter
            layer_or_band = layer_name if layer_name else (band_index if band_index is not None else 0)
            
            # Enrich file using geometadatatools
            if return_object:
                data_object, enriched_info = read_and_enrich(
                    file_path, 
                    layer_or_band, 
                    return_object=True
                )
            else:
                enriched_info = read_and_enrich(
                    file_path, 
                    layer_or_band, 
                    return_object=False
                )
                data_object = None
            
            logger.info(f"Successfully enriched file: {file_path}")
            
            if return_object:
                return data_object, enriched_info
            else:
                return enriched_info
                
        except Exception as e:
            logger.error(f"Error enriching file {file_path}: {str(e)}")
            raise

    def extract_comprehensive_metadata(self, file_path: str, 
                                    layer_name_or_band_index: Optional[Union[str, int]] = None,
                                    categorical_allow_list: Optional[List[str]] = None,
                                    categorical_deny_list: Optional[List[str]] = None,
                                    return_object: bool = False,
                                    generate_images: bool = False) -> Union[Dict[str, Any], Tuple[Any, Dict[str, Any]]]:
        """
        Extract comprehensive metadata from a geospatial file using read_and_enrich
        
        Args:
            file_path: Path to the geospatial file
            layer_name_or_band_index: Layer name or band index for processing
            categorical_allow_list: List of columns to force-treat as categorical
            categorical_deny_list: List of columns to exclude from categorical analysis
            return_object: Whether to return the actual data object along with metadata
            generate_images: Whether to generate base64 images (may cause threading issues)
            
        Returns:
            Either enriched metadata dict or tuple of (data_object, metadata_dict)
        """
        try:
            logger.info(f"Extracting comprehensive metadata for file: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Set default layer/band if not specified
            if layer_name_or_band_index is None:
                # Get file info to determine default layer/band
                file_info = get_file_info(file_path)
                if file_info.get("type") == "vector" and file_info.get("layers"):
                    layer_name_or_band_index = file_info["layers"][0]  # Use first layer
                elif file_info.get("type") == "raster":
                    layer_name_or_band_index = 1  # Use first band
            
            # Extract metadata using read_and_enrich
            if return_object:
                data_object, metadata = read_and_enrich(
                    file_path,
                    layer_name_or_band_index,
                    categorical_allow_list=categorical_allow_list or [],
                    categorical_deny_list=categorical_deny_list or [],
                    return_object=True
                )
                logger.info(f"Successfully extracted metadata with data object for: {file_path}")
                return data_object, metadata
            else:
                metadata = read_and_enrich(
                    file_path,
                    layer_name_or_band_index,
                    categorical_allow_list=categorical_allow_list or [],
                    categorical_deny_list=categorical_deny_list or [],
                    return_object=False
                )
                
                # Remove image strings if image generation is disabled to prevent threading issues
                if not generate_images and "img_strings" in metadata:
                    metadata["img_strings"] = []
                    metadata["images_disabled"] = True
                    metadata["images_disabled_reason"] = "Image generation disabled to prevent threading issues"
                
                logger.info(f"Successfully extracted metadata for: {file_path}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting comprehensive metadata for {file_path}: {str(e)}")
            raise
    
    def get_comprehensive_metadata(self, file_path: str, layer_name: Optional[str] = None,
                                 band_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a geospatial file
        
        Args:
            file_path: Path to the geospatial file
            layer_name: Layer name for multi-layer formats
            band_index: Band index for raster data
            
        Returns:
            Dictionary containing comprehensive metadata
        """
        try:
            logger.info(f"Getting comprehensive metadata for file: {file_path}")
            
            # Get basic file info
            file_info = self.get_file_metadata(file_path)
            
            # Get bounding box
            bbox = self.get_bounding_box(file_path)
            
            # Get enriched metadata for specific layer/band
            layer_or_band = layer_name if layer_name else (band_index if band_index is not None else 0)
            enriched_metadata = self.extract_comprehensive_metadata(file_path, layer_or_band)
            
            # Combine all metadata
            comprehensive_metadata = {
                "file": file_info.get("file", {}),
                "type": file_info.get("type", "unknown"),
                "layers": file_info.get("layers", []),
                "bounding_box": bbox,
                "raster_stats": file_info.get("raster_stats"),
                "enriched_metadata": enriched_metadata
            }
            
            logger.info(f"Successfully retrieved comprehensive metadata for {file_path}")
            return comprehensive_metadata
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metadata for {file_path}: {str(e)}")
            raise

    def extract_data_to_csv(self, file_path: str, 
                           layer_name_or_band_index: Optional[str] = None,
                           csv_output_path: str = None) -> dict:
        """
        Extract data from geospatial files and save as CSV
        
        Args:
            file_path: Path to the geospatial file
            layer_name_or_band_index: Layer name (vector) or band index (raster)
            csv_output_path: Path where the CSV file will be written
            
        Returns:
            dict: Extraction result and metadata
        """
        try:
            logger.info(f"Extracting data from file: {file_path} to CSV: {csv_output_path}")
            
            # Import the new functions
            from geometadatatools import vector_to_dataframe, raster_to_dataframe, get_file_info
            
            # Get file info to determine type
            file_info = get_file_info(file_path)
            file_type = file_info.get("type", "unknown")
            
            if file_type == "vector":
                # Extract vector data with geometry as WKT for CSV compatibility
                df = vector_to_dataframe(
                    file_path,
                    layer_name=layer_name_or_band_index,
                    include_geometry=True,
                    geometry_format="wkt"
                )
            elif file_type == "raster":
                # Extract raster data with coordinates
                band_index = int(layer_name_or_band_index) if layer_name_or_band_index else 1
                df = raster_to_dataframe(
                    file_path,
                    band_index=band_index,
                    include_coordinates=True,
                    exclude_nodata=True
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(csv_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Write DataFrame to CSV
            df.to_csv(csv_output_path, index=False)
            
            result = {
                "file_path": file_path,
                "csv_output_path": csv_output_path,
                "file_type": file_type,
                "layer_name_or_band_index": layer_name_or_band_index,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "status": "success"
            }
            
            logger.info(f"Successfully extracted {len(df)} rows from {file_path} to CSV: {csv_output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting data from {file_path} to CSV: {str(e)}")
            raise

    def get_processing_recommendations(self, file_info: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Generate processing recommendations based on file characteristics
        
        Args:
            file_info: Basic file information from get_file_metadata
            file_path: Path to the geospatial file
            
        Returns:
            Dictionary containing processing recommendations
        """
        try:
            recommendations = {
                "suggested_approach": "single_layer",
                "estimated_processing_time": "unknown",
                "memory_requirements": "unknown",
                "layer_processing_order": [],
                "categorical_analysis": {},
                "warnings": [],
                "tips": []
            }
            
            file_type = file_info.get("type", "unknown")
            layers = file_info.get("layers", [])
            file_size = file_info.get("file", {}).get("file_size", {})
            
            # Determine processing approach
            if len(layers) > 1:
                recommendations["suggested_approach"] = "multi_layer"
                recommendations["tips"].append("Consider using batch processing for multiple layers")
                
                if file_type == "vector":
                    recommendations["tips"].append("Vector layers can be processed independently")
                elif file_type == "raster":
                    recommendations["tips"].append("Raster bands may have interdependencies")
            
            # Estimate processing time based on file size and type
            if file_size and "size" in file_size:
                size_mb = file_size["size"]
                if file_type == "vector":
                    if size_mb < 10:
                        recommendations["estimated_processing_time"] = "fast (< 30 seconds)"
                        recommendations["memory_requirements"] = "low (< 100MB)"
                    elif size_mb < 100:
                        recommendations["estimated_processing_time"] = "medium (1-5 minutes)"
                        recommendations["memory_requirements"] = "medium (100MB-1GB)"
                    else:
                        recommendations["estimated_processing_time"] = "slow (5+ minutes)"
                        recommendations["memory_requirements"] = "high (> 1GB)"
                        recommendations["warnings"].append("Large file - consider using queue endpoint")
                elif file_type == "raster":
                    if size_mb < 50:
                        recommendations["estimated_processing_time"] = "fast (< 1 minute)"
                        recommendations["memory_requirements"] = "low (< 200MB)"
                    elif size_mb < 500:
                        recommendations["estimated_processing_time"] = "medium (2-10 minutes)"
                        recommendations["memory_requirements"] = "medium (200MB-2GB)"
                    else:
                        recommendations["estimated_processing_time"] = "slow (10+ minutes)"
                        recommendations["memory_requirements"] = "high (> 2GB)"
                        recommendations["warnings"].append("Large raster file - consider using queue endpoint")
            
            # Suggest layer processing order
            if file_type == "vector" and layers:
                # For vector files, suggest processing order based on typical importance
                priority_layers = []
                other_layers = []
                
                for layer in layers:
                    layer_lower = layer.lower()
                    if any(keyword in layer_lower for keyword in ["main", "primary", "core", "data", "features"]):
                        priority_layers.append(layer)
                    elif any(keyword in layer_lower for keyword in ["index", "spatial", "boundary", "outline"]):
                        priority_layers.append(layer)
                    else:
                        other_layers.append(layer)
                
                recommendations["layer_processing_order"] = priority_layers + other_layers
                
            elif file_type == "raster" and layers:
                # For raster files, suggest processing order (usually band 1 first)
                recommendations["layer_processing_order"] = list(range(1, len(layers) + 1))
                recommendations["tips"].append("Start with band 1 for initial analysis")
            
            # Categorical analysis recommendations
            if file_type == "vector":
                recommendations["categorical_analysis"] = {
                    "recommended_columns": ["type", "category", "class", "status"],
                    "avoid_columns": ["id", "uuid", "geometry", "coordinates"],
                    "threshold": "100 unique values for automatic detection"
                }
            
            # Add file-specific tips
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".gpkg":
                recommendations["tips"].append("GeoPackage files may contain multiple vector layers")
                recommendations["tips"].append("Use layer names for specific layer processing")
            elif file_ext == ".zip":
                recommendations["tips"].append("ZIP files may contain shapefile components")
                recommendations["warnings"].append("Ensure ZIP contains valid geospatial files")
            elif file_ext in [".nc", ".netcdf"]:
                recommendations["tips"].append("NetCDF files may have multiple variables and time dimensions")
                recommendations["tips"].append("Consider temporal analysis for time-series data")
            
            # Add general recommendations
            if len(layers) > 5:
                recommendations["warnings"].append("Many layers detected - consider batch processing")
                recommendations["tips"].append("Use batch endpoint for processing all layers")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating processing recommendations for {file_path}: {str(e)}")
            return {
                "suggested_approach": "unknown",
                "estimated_processing_time": "unknown",
                "memory_requirements": "unknown",
                "layer_processing_order": [],
                "categorical_analysis": {},
                "warnings": ["Could not generate recommendations"],
                "tips": ["Use basic metadata extraction"]
            }
    
    def validate_file_format(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if a file is a supported geospatial format
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            # Check if it's a supported format
            is_vector = file_ext in self.supported_formats['vector']
            is_raster = file_ext in self.supported_formats['raster']
            
            validation_result = {
                "is_supported": is_vector or is_raster,
                "file_extension": file_ext,
                "data_type": "vector" if is_vector else "raster" if is_raster else "unknown",
                "supported_formats": self.supported_formats
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file format for {file_path}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> Dict[str, list]:
        """
        Get list of supported geospatial file formats
        
        Returns:
            Dictionary containing supported formats by type
        """
        return self.supported_formats.copy()
