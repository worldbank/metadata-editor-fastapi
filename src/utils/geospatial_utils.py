import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import mimetypes

logger = logging.getLogger(__name__)


def is_geospatial_file(file_path: str) -> bool:
    """
    Check if a file is a supported geospatial format based on extension
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is a supported geospatial format
    """
    supported_extensions = {
        # Vector formats
        '.shp', '.geojson', '.json', '.kml', '.kmz', '.gpkg', '.gdb',
        # Raster formats
        '.nc', '.tif', '.tiff', '.bil', '.ascii', '.txt', '.pdf', '.jpg', '.jpeg', '.gif', '.adf', '.ovr'
    }
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_extensions


def get_file_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'


def validate_file_path(file_path: str, storage_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a file path for security and accessibility
    
    Args:
        file_path: Path to validate
        storage_path: Optional storage path restriction
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "is_valid": False,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result["errors"].append("File does not exist")
            return validation_result
        
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            validation_result["errors"].append("Path is not a file")
            return validation_result
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            validation_result["warnings"].append("File is empty")
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            validation_result["errors"].append("File is not readable")
            return validation_result
        
        # Check storage path restriction if specified
        if storage_path:
            abs_file_path = os.path.abspath(file_path)
            abs_storage_path = os.path.abspath(storage_path)
            
            if not abs_file_path.startswith(abs_storage_path):
                validation_result["errors"].append("File path is outside allowed storage directory")
                return validation_result
        
        # Check if it's a geospatial file
        if not is_geospatial_file(file_path):
            validation_result["warnings"].append("File extension suggests this may not be a geospatial file")
        
        # If we get here, the file is valid
        validation_result["is_valid"] = True
        validation_result["file_size"] = file_size
        validation_result["mime_type"] = get_file_mime_type(file_path)
        
    except Exception as e:
        validation_result["errors"].append(f"Validation error: {str(e)}")
    
    return validation_result


def get_file_info_summary(file_path: str) -> Dict[str, Any]:
    """
    Get a summary of file information
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information summary
    """
    try:
        stat = os.stat(file_path)
        path_obj = Path(file_path)
        
        return {
            "filename": path_obj.name,
            "file_path": file_path,
            "file_size": stat.st_size,
            "file_size_formatted": format_file_size(stat.st_size),
            "file_extension": path_obj.suffix.lower(),
            "is_geospatial": is_geospatial_file(file_path),
            "mime_type": get_file_mime_type(file_path),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
            "accessed_time": stat.st_atime
        }
        
    except Exception as e:
        logger.error(f"Error getting file info summary for {file_path}: {str(e)}")
        return {
            "filename": Path(file_path).name,
            "file_path": file_path,
            "error": str(e)
        }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    sanitized = filename
    
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized


def create_output_path(input_path: str, output_dir: str, suffix: str = "", 
                      extension: str = "") -> str:
    """
    Create an output file path based on input file
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Optional suffix to add to filename
        extension: Optional new extension (without dot)
        
    Returns:
        Output file path
    """
    input_path_obj = Path(input_path)
    filename = input_path_obj.stem
    
    if suffix:
        filename = f"{filename}_{suffix}"
    
    if extension:
        if not extension.startswith('.'):
            extension = f".{extension}"
    else:
        extension = input_path_obj.suffix
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, f"{filename}{extension}")


def get_supported_formats_by_category() -> Dict[str, List[str]]:
    """
    Get supported geospatial formats organized by category
    
    Returns:
        Dictionary with formats organized by category
    """
    return {
        "vector": [
            "ESRI Shapefile (.shp)",
            "GeoJSON (.geojson, .json)",
            "KML (.kml)",
            "KMZ (.kmz)",
            "GeoPackage (.gpkg)",
            "File Geodatabase (.gdb)"
        ],
        "raster": [
            "NetCDF (.nc)",
            "GeoTIFF (.tif, .tiff)",
            "BIL (.bil)",
            "ASCII Grid (.ascii, .txt)",
            "GeoPDF (.pdf)",
            "Image formats (.jpg, .jpeg, .gif)",
            "ESRI ASCII (.adf)",
            "Overview files (.ovr)"
        ]
    }
