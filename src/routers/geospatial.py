from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import os
import time
import datetime
import json
import functools
import asyncio

from ..models.geospatial_models import (
    GeospatialImportRequest,
    GeospatialMetadataRequest,
    GeospatialMetadataResponse,
    GeospatialTransformRequest,
    GeospatialClipRequest,
    GeospatialPreviewRequest,
    GeospatialJobResponse,
    GeospatialMetadataJobResponse,
    GeospatialErrorResponse,
    GeospatialBatchMetadataRequest,
    GeospatialBatchJobResponse,
    GeospatialDataExtractionRequest,
    GeospatialDataExtractionJobResponse
)
from ..services.geospatial_service import GeospatialService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geospatial", tags=["geospatial"])

# Initialize the geospatial service
geospatial_service = GeospatialService()


def get_geospatial_service() -> GeospatialService:
    """Dependency to get geospatial service instance"""
    service = geospatial_service
    if not service.available:
        raise HTTPException(
            status_code=503,
            detail="Geospatial features require additional packages. Install with: pip install -r requirements-geospatial.txt"
        )
    return service


@router.get("/")
async def geospatial_root():
    """Root endpoint for geospatial operations"""
    return {
        "message": "Geospatial API - Use /docs for API documentation",
        "supported_formats": geospatial_service.get_supported_formats(),
        "endpoints": [
            "/geospatial/layers",
            "/geospatial/layers-queue",
            "/geospatial/data-queue",
            "/geospatial/metadata",
            "/geospatial/metadata-queue",
            "/geospatial/metadata-queue-batch",
            "/geospatial/metadata-queue-with-images",
            "/geospatial/bounding-box",
            "/geospatial/enrich",
            "/geospatial/enrich-queue",
            "/geospatial/comprehensive",
            "/geospatial/validate",
            "/geospatial/formats"
        ],
        "workflow": {
            "step1": "Use /geospatial/layers or /geospatial/layers-queue to get available layers",
            "step2": "Choose processing approach: layers, data extraction, or metadata analysis",
            "step3": "Use appropriate endpoint for your needs",
            "step4": "Monitor job progress and retrieve results"
        },
        "notes": {
            "layers_endpoint": "/geospatial/layers provides only layer information without analysis",
            "data_extraction": "/geospatial/data-queue extracts raw data as DataFrame for CSV export or analysis",
            "image_generation": "Default endpoints disable image generation to prevent threading issues",
            "images_available": "Use /geospatial/metadata-queue-with-images if you need base64 images (may cause threading issues)"
        }
    }


@router.post("/layers")
async def get_geospatial_layers(
    file_path: str,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Get layer information from a geospatial file
    This endpoint extracts only the available layers/bands without other analysis
    
    Args:
        file_path: Path to the geospatial file
    """
    try:
        # Validate file path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Get basic file information to extract layers only
        file_info = service.get_file_metadata(file_path)
        
        return {
            "file_path": file_path,
            "type": file_info.get("type", "unknown"),
            "layers": file_info.get("layers", []),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting geospatial layers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get layers: {str(e)}")


@router.post("/layers-queue")
async def get_geospatial_layers_queue(
    request: GeospatialImportRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Queue a geospatial layers extraction job for asynchronous processing
    This extracts only layer information without other analysis
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Generate unique job ID
        jobid = f'geospatial-layers-{int(time.time() * 1000)}'
        current_time = datetime.datetime.now().isoformat()
        
        # Create job entry
        job_info = {
            "jobid": jobid,
            "jobtype": "geospatial-layers-extraction",
            "status": "queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info": {
                "file_path": request.file_path,
                "layer_name": request.layer_name,
                "band_index": request.band_index,
                "return_object": request.return_object
            }
        }
        
        # Add job to the main app's job dictionary
        from main import app
        app.jobs[jobid] = job_info
        
        # Create callback function for the queue
        layers_callback = functools.partial(
            process_geospatial_layers_job, 
            jobid, 
            request
        )
        
        # Add to the main app's FIFO queue
        await app.fifo_queue.put(layers_callback)
        
        logger.info(f"Queued geospatial layers extraction job {jobid} for file: {request.file_path}")
        
        return GeospatialJobResponse(
            message="Geospatial layers extraction job is queued",
            job_id=jobid,
            operation_type="geospatial-layers-extraction"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing geospatial layers extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@router.post("/data-queue")
async def extract_geospatial_data_queue(
    request: GeospatialDataExtractionRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Queue a geospatial data extraction job for asynchronous processing
    This extracts raw data and saves it as a CSV file
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Validate CSV output path
        csv_dir = os.path.dirname(request.csv_output_path)
        if csv_dir and not os.path.exists(csv_dir):
            try:
                os.makedirs(csv_dir, exist_ok=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot create output directory: {str(e)}")
        
        # Get file info to validate layer/band selection
        file_info = service.get_file_metadata(request.file_path)
        file_type = file_info.get("type", "unknown")
        available_layers = file_info.get("layers", [])
        
        # Validate layer/band selection
        if request.layer_name_or_band_index is not None:
            if file_type == "vector":
                if request.layer_name_or_band_index not in available_layers:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Layer '{request.layer_name_or_band_index}' not found. Available layers: {available_layers}"
                    )
            elif file_type == "raster":
                try:
                    band_index = int(request.layer_name_or_band_index)
                    if band_index < 1 or band_index > len(available_layers):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Band index {band_index} out of range. Available bands: 1-{len(available_layers)}"
                        )
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid band index '{request.layer_name_or_band_index}'. Must be an integer."
                    )
        
        # Generate unique job ID
        jobid = f'geospatial-data-{int(time.time() * 1000)}'
        current_time = datetime.datetime.now().isoformat()
        
        # Create job entry
        job_info = {
            "jobid": jobid,
            "jobtype": "geospatial-data-extraction",
            "status": "queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info": {
                "file_path": request.file_path,
                "csv_output_path": request.csv_output_path,
                "file_type": file_type,
                "layer_name_or_band_index": request.layer_name_or_band_index,
                "available_layers": available_layers
            }
        }
        
        # Add job to the main app's job dictionary
        from main import app
        app.jobs[jobid] = job_info
        
        # Create callback function for the queue
        data_callback = functools.partial(
            process_geospatial_data_job, 
            jobid, 
            request
        )
        
        # Add to the main app's FIFO queue
        await app.fifo_queue.put(data_callback)
        
        logger.info(f"Queued geospatial data extraction job {jobid} for file: {request.file_path} -> {request.csv_output_path}")
        
        return GeospatialDataExtractionJobResponse(
            message="Geospatial data extraction job is queued",
            job_id=jobid,
            operation_type="geospatial-data-extraction",
            file_path=request.file_path,
            parameters=request.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing geospatial data extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@router.post("/metadata-queue")
async def extract_geospatial_metadata_queue(
    request: GeospatialMetadataRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Queue a geospatial metadata extraction job for asynchronous processing
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get file info to validate layer/band selection
        file_info = service.get_file_metadata(request.file_path)
        file_type = file_info.get("type", "unknown")
        available_layers = file_info.get("layers", [])
        
        # Validate layer/band selection
        if request.layer_name_or_band_index is not None:
            if file_type == "vector":
                if request.layer_name_or_band_index not in available_layers:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Layer '{request.layer_name_or_band_index}' not found. Available layers: {available_layers}"
                    )
            elif file_type == "raster":
                try:
                    band_index = int(request.layer_name_or_band_index)
                    if band_index < 1 or band_index > len(available_layers):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Band index {band_index} out of range. Available bands: 1-{len(available_layers)}"
                        )
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid band index '{request.layer_name_or_band_index}'. Must be an integer."
                    )
        
        # Generate unique job ID
        jobid = f'geospatial-metadata-{int(time.time() * 1000)}'
        current_time = datetime.datetime.now().isoformat()
        
        # Create job entry
        job_info = {
            "jobid": jobid,
            "jobtype": "geospatial-metadata-extraction",
            "status": "queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info": {
                "file_path": request.file_path,
                "file_type": file_type,
                "layer_name_or_band_index": request.layer_name_or_band_index,
                "categorical_allow_list": request.categorical_allow_list or [],
                "categorical_deny_list": request.categorical_deny_list or [],
                "return_object": request.return_object,
                "available_layers": available_layers
            }
        }
        
        # Add job to the main app's job dictionary
        from main import app
        app.jobs[jobid] = job_info
        
        # Create callback function for the queue
        metadata_callback = functools.partial(
            process_geospatial_metadata_job, 
            jobid, 
            request
        )
        
        # Add to the main app's FIFO queue
        await app.fifo_queue.put(metadata_callback)
        
        logger.info(f"Queued geospatial metadata extraction job {jobid} for file: {request.file_path}")
        
        return GeospatialMetadataJobResponse(
            message="Geospatial metadata extraction job is queued",
            job_id=jobid,
            operation_type="geospatial-metadata-extraction",
            file_path=request.file_path,
            parameters=request.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing geospatial metadata extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@router.post("/metadata-queue-batch")
async def extract_geospatial_metadata_batch_queue(
    request: GeospatialBatchMetadataRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Queue multiple geospatial metadata extraction jobs for batch processing
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get file info to validate layer selections
        file_info = service.get_file_metadata(request.file_path)
        file_type = file_info.get("type", "unknown")
        available_layers = file_info.get("layers", [])
        
        # Validate layer selections
        if request.layer_names_or_band_indices:
            if file_type == "vector":
                invalid_layers = [layer for layer in request.layer_names_or_band_indices if layer not in available_layers]
                if invalid_layers:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid layers: {invalid_layers}. Available layers: {available_layers}"
                    )
            elif file_type == "raster":
                try:
                    band_indices = [int(band) for band in request.layer_names_or_band_indices]
                    invalid_bands = [band for band in band_indices if band < 1 or band > len(available_layers)]
                    if invalid_bands:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid band indices: {invalid_bands}. Available bands: 1-{len(available_layers)}"
                        )
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail="All band indices must be integers for raster files."
                    )
        else:
            # If no specific layers specified, process all available layers
            request.layer_names_or_band_indices = available_layers
        
        # Generate batch job ID
        batch_jobid = f'geospatial-batch-metadata-{int(time.time() * 1000)}'
        current_time = datetime.datetime.now().isoformat()
        
        # Create batch job entry
        batch_job_info = {
            "jobid": batch_jobid,
            "jobtype": "geospatial-batch-metadata-extraction",
            "status": "queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info": {
                "file_path": request.file_path,
                "file_type": file_type,
                "layer_names_or_band_indices": request.layer_names_or_band_indices,
                "categorical_allow_list": request.categorical_allow_list or [],
                "categorical_deny_list": request.categorical_deny_list or [],
                "return_object": request.return_object,
                "available_layers": available_layers,
                "total_layers": len(request.layer_names_or_band_indices)
            }
        }
        
        # Add batch job to the main app's job dictionary
        from main import app
        app.jobs[batch_jobid] = batch_job_info
        
        # Create callback function for batch processing
        batch_callback = functools.partial(
            process_geospatial_batch_metadata_job,
            batch_jobid,
            request
        )
        
        # Add to the main app's FIFO queue
        await app.fifo_queue.put(batch_callback)
        
        logger.info(f"Queued batch geospatial metadata extraction job {batch_jobid} for file: {request.file_path}")
        
        return GeospatialBatchJobResponse(
            message="Batch geospatial metadata extraction job is queued",
            job_id=batch_jobid,
            operation_type="geospatial-batch-metadata-extraction",
            file_path=request.file_path,
            total_layers=len(request.layer_names_or_band_indices),
            parameters=request.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing batch geospatial metadata extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue batch job: {str(e)}")


async def process_geospatial_layers_job(jobid: str, request: GeospatialImportRequest):
    """
    Process a geospatial layers extraction job from the queue
    """
    from main import app
    
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    
    try:
        logger.info(f"Processing geospatial layers extraction job {jobid} for file: {request.file_path}")
        
        # Get basic file information to extract layers only
        file_info = await loop.run_in_executor(
            None, 
            geospatial_service.get_file_metadata,
            request.file_path
        )
        
        # Prepare simplified result with only layer information
        result = {
            "file_path": request.file_path,
            "type": file_info.get("type", "unknown"),
            "layers": file_info.get("layers", []),
            "status": "success"
        }
        
        # Mark job as completed
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        # Save result to file
        file_path = os.path.join('jobs', f'{jobid}.json')
        os.makedirs('jobs', exist_ok=True)
        
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile, default=str)
        
        logger.info(f"Successfully completed geospatial layers extraction job {jobid}")
        return {"status": "success", "file_path": file_path}
        
    except Exception as e:
        logger.error(f"Error processing geospatial layers extraction job {jobid}: {str(e)}")
        
        # Mark job as failed
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["error_details"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function": "process_geospatial_layers_job",
            "jobid": jobid,
            "file_path": request.file_path
        }
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        return {"status": "error", "error": str(e)}


async def process_geospatial_data_job(jobid: str, request: GeospatialDataExtractionRequest):
    """
    Process a geospatial data extraction job from the queue
    """
    from main import app
    
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    
    try:
        logger.info(f"Processing geospatial data extraction job {jobid} for file: {request.file_path} -> {request.csv_output_path}")
        
        # Extract data and save as CSV using the service
        result = await loop.run_in_executor(
            None, 
            geospatial_service.extract_data_to_csv,
            request.file_path,
            request.layer_name_or_band_index,
            request.csv_output_path
        )
        
        # Mark job as completed
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        # Save job result metadata to file
        job_result_path = os.path.join('jobs', f'{jobid}.json')
        os.makedirs('jobs', exist_ok=True)
        
        with open(job_result_path, 'w') as outfile:
            json.dump(result, outfile, default=str)
        
        logger.info(f"Successfully completed geospatial data extraction job {jobid}. CSV saved to: {request.csv_output_path}")
        return {"status": "success", "csv_file": request.csv_output_path, "job_result": job_result_path}
        
    except Exception as e:
        logger.error(f"Error processing geospatial data extraction job {jobid}: {str(e)}")
        
        # Mark job as failed
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["error_details"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function": "process_geospatial_data_job",
            "jobid": jobid,
            "file_path": request.file_path,
            "csv_output_path": request.csv_output_path
        }
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        return {"status": "error", "error": str(e)}


async def process_geospatial_metadata_job(jobid: str, request: GeospatialMetadataRequest):
    """
    Process a geospatial metadata extraction job from the queue
    """
    from main import app
    
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    
    try:
        logger.info(f"Processing geospatial metadata extraction job {jobid} for file: {request.file_path}")
        
        # Extract metadata using the service
        result = await loop.run_in_executor(
            None, 
            geospatial_service.extract_comprehensive_metadata,
            request.file_path,
            request.layer_name_or_band_index,
            request.categorical_allow_list,
            request.categorical_deny_list,
            request.return_object,
            False  # generate_images=False to prevent threading issues
        )
        
        # Mark job as completed
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        # Save result to file
        file_path = os.path.join('jobs', f'{jobid}.json')
        os.makedirs('jobs', exist_ok=True)
        
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile, default=str)
        
        logger.info(f"Successfully completed geospatial metadata extraction job {jobid}")
        return {"status": "success", "file_path": file_path}
        
    except Exception as e:
        logger.error(f"Error processing geospatial metadata extraction job {jobid}: {str(e)}")
        
        # Mark job as failed
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["error_details"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function": "process_geospatial_metadata_job",
            "jobid": jobid,
            "file_path": request.file_path
        }
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        return {"status": "error", "error": str(e)}


async def process_geospatial_batch_metadata_job(jobid: str, request: GeospatialBatchMetadataRequest):
    """
    Process a batch geospatial metadata extraction job from the queue
    """
    from main import app
    
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    
    try:
        logger.info(f"Processing batch geospatial metadata extraction job {jobid} for file: {request.file_path}")
        
        # Extract metadata using the service for each layer/band
        results = []
        for layer_name_or_band_index in request.layer_names_or_band_indices:
            result = await loop.run_in_executor(
                None, 
                geospatial_service.extract_comprehensive_metadata,
                request.file_path,
                layer_name_or_band_index,
                request.categorical_allow_list,
                request.categorical_deny_list,
                request.return_object,
                False  # generate_images=False to prevent threading issues
            )
            results.append(result)
        
        # Mark job as completed
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        # Save results to files
        file_paths = []
        for i, result in enumerate(results):
            layer_name_or_band_index = request.layer_names_or_band_indices[i]
            file_name = f"{jobid}-{layer_name_or_band_index}.json"
            file_path = os.path.join('jobs', file_name)
            os.makedirs('jobs', exist_ok=True)
            
            with open(file_path, 'w') as outfile:
                json.dump(result, outfile, default=str)
            file_paths.append(file_path)
        
        logger.info(f"Successfully completed batch geospatial metadata extraction job {jobid}")
        return {"status": "success", "file_paths": file_paths}
        
    except Exception as e:
        logger.error(f"Error processing batch geospatial metadata extraction job {jobid}: {str(e)}")
        
        # Mark job as failed
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["error_details"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function": "process_geospatial_batch_metadata_job",
            "jobid": jobid,
            "file_path": request.file_path
        }
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        return {"status": "error", "error": str(e)}


@router.post("/metadata-queue-with-images")
async def extract_geospatial_metadata_with_images_queue(
    request: GeospatialMetadataRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Queue a geospatial metadata extraction job with image generation enabled
    WARNING: This may cause threading issues on some systems
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get file info to validate layer/band selection
        file_info = service.get_file_metadata(request.file_path)
        file_type = file_info.get("type", "unknown")
        available_layers = file_info.get("layers", [])
        
        # Validate layer/band selection
        if request.layer_name_or_band_index is not None:
            if file_type == "vector":
                if request.layer_name_or_band_index not in available_layers:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Layer '{request.layer_name_or_band_index}' not found. Available layers: {available_layers}"
                    )
            elif file_type == "raster":
                try:
                    band_index = int(request.layer_name_or_band_index)
                    if band_index < 1 or band_index > len(available_layers):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Band index {band_index} out of range. Available bands: 1-{len(available_layers)}"
                        )
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid band index '{request.layer_name_or_band_index}'. Must be an integer."
                    )
        
        # Generate unique job ID
        jobid = f'geospatial-metadata-images-{int(time.time() * 1000)}'
        current_time = datetime.datetime.now().isoformat()
        
        # Create job entry
        job_info = {
            "jobid": jobid,
            "jobtype": "geospatial-metadata-extraction-with-images",
            "status": "queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info": {
                "file_path": request.file_path,
                "file_type": file_type,
                "layer_name_or_band_index": request.layer_name_or_band_index,
                "categorical_allow_list": request.categorical_allow_list or [],
                "categorical_deny_list": request.categorical_deny_list or [],
                "return_object": request.return_object,
                "available_layers": available_layers,
                "generate_images": True,
                "warning": "Image generation enabled - may cause threading issues"
            }
        }
        
        # Add job to the main app's job dictionary
        from main import app
        app.jobs[jobid] = job_info
        
        # Create callback function for the queue
        metadata_callback = functools.partial(
            process_geospatial_metadata_with_images_job, 
            jobid, 
            request
        )
        
        # Add to the main app's FIFO queue
        await app.fifo_queue.put(metadata_callback)
        
        logger.info(f"Queued geospatial metadata extraction job with images {jobid} for file: {request.file_path}")
        
        return GeospatialMetadataJobResponse(
            message="Geospatial metadata extraction job with images is queued (WARNING: May cause threading issues)",
            job_id=jobid,
            operation_type="geospatial-metadata-extraction-with-images",
            file_path=request.file_path,
            parameters=request.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing geospatial metadata extraction with images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


async def process_geospatial_metadata_with_images_job(jobid: str, request: GeospatialMetadataRequest):
    """
    Process a geospatial metadata extraction job with images from the queue
    WARNING: This may cause threading issues
    """
    from main import app
    
    loop = asyncio.get_running_loop()
    app.jobs[jobid]["status"] = "processing"
    
    try:
        logger.info(f"Processing geospatial metadata extraction with images job {jobid} for file: {request.file_path}")
        
        # Extract metadata using the service with images enabled
        result = await loop.run_in_executor(
            None, 
            geospatial_service.extract_comprehensive_metadata,
            request.file_path,
            request.layer_name_or_band_index,
            request.categorical_allow_list,
            request.categorical_deny_list,
            request.return_object,
            True  # generate_images=True (may cause threading issues)
        )
        
        # Mark job as completed
        app.jobs[jobid]["status"] = "done"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        # Save result to file
        file_path = os.path.join('jobs', f'{jobid}.json')
        os.makedirs('jobs', exist_ok=True)
        
        with open(file_path, 'w') as outfile:
            json.dump(result, outfile, default=str)
        
        logger.info(f"Successfully completed geospatial metadata extraction with images job {jobid}")
        return {"status": "success", "file_path": file_path}
        
    except Exception as e:
        logger.error(f"Error processing geospatial metadata extraction with images job {jobid}: {str(e)}")
        
        # Mark job as failed
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["error_details"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function": "process_geospatial_metadata_with_images_job",
            "jobid": jobid,
            "file_path": request.file_path,
            "note": "This error may be related to matplotlib threading issues"
        }
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        
        return {"status": "error", "error": str(e)}


@router.post("/bounding-box")
async def get_geospatial_bounding_box(
    request: GeospatialImportRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Get bounding box for a geospatial file in WGS84 coordinates
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get bounding box
        bbox = service.get_bounding_box(request.file_path)
        
        return {
            "file_path": request.file_path,
            "bounding_box": bbox,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting bounding box: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get bounding box: {str(e)}")


@router.post("/enrich")
async def enrich_geospatial_file(
    request: GeospatialImportRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Enrich a geospatial file with additional metadata and analytics
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Enrich file
        data_obj, enriched_info = service.enrich_file(
            request.file_path,
            layer_name=request.layer_name,
            band_index=request.band_index,
            return_object=request.return_object
        )
        
        return {
            "file_path": request.file_path,
            "enriched_info": enriched_info,
            "has_data_object": request.return_object,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error enriching geospatial file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enrich file: {str(e)}")


@router.post("/comprehensive")
async def get_comprehensive_metadata(
    request: GeospatialImportRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Get comprehensive metadata including file info, bounding box, and analytics
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get comprehensive metadata
        comprehensive_metadata = service.get_comprehensive_metadata(
            request.file_path,
            layer_name=request.layer_name,
            band_index=request.band_index
        )
        
        return GeospatialMetadataResponse(**comprehensive_metadata)
        
    except Exception as e:
        logger.error(f"Error getting comprehensive metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get comprehensive metadata: {str(e)}")


@router.post("/validate")
async def validate_geospatial_file(
    request: GeospatialImportRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Validate if a file is a supported geospatial format
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Validate file format
        validation_result = service.validate_file_format(request.file_path)
        
        return {
            "file_path": request.file_path,
            "validation_result": validation_result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error validating geospatial file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate file: {str(e)}")


@router.get("/formats")
async def get_supported_formats(
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Get list of supported geospatial file formats
    """
    try:
        formats = service.get_supported_formats()
        
        return {
            "supported_formats": formats,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported formats: {str(e)}")


# Queue-based endpoints for long-running operations
@router.post("/enrich-queue")
async def enrich_geospatial_file_queue(
    request: GeospatialImportRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Queue a request to enrich a geospatial file (for long-running operations)
    """
    try:
        # Generate job ID
        jobid = f'geospatial-enrich-{int(time.time())}'
        current_time = datetime.datetime.now().isoformat()
        
        # Add to jobs (assuming app.jobs is accessible)
        job_info = {
            "jobid": jobid,
            "jobtype": "geospatial-enrich",
            "status": "queued",
            "created_at": current_time,
            "completed_at": None,
            "last_accessed": current_time,
            "info": request.dict()
        }
        
        # For now, return immediate response
        # In production, this would be added to your job queue
        return GeospatialJobResponse(
            message="Enrich request queued successfully",
            job_id=jobid,
            operation_type="geospatial-enrich"
        )
        
    except Exception as e:
        logger.error(f"Error queuing geospatial enrich request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue request: {str(e)}")


# Placeholder endpoints for future functionality
@router.post("/transform")
async def transform_coordinate_system(
    request: GeospatialTransformRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Transform coordinate system of a geospatial file
    """
    # TODO: Implement coordinate system transformation
    raise HTTPException(status_code=501, detail="Coordinate system transformation not yet implemented")


@router.post("/clip")
async def clip_geospatial_data(
    request: GeospatialClipRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Clip geospatial data using a geometry
    """
    # TODO: Implement spatial clipping
    raise HTTPException(status_code=501, detail="Spatial clipping not yet implemented")


@router.post("/preview")
async def generate_preview(
    request: GeospatialPreviewRequest,
    service: GeospatialService = Depends(get_geospatial_service)
):
    """
    Generate preview image of geospatial data
    """
    # TODO: Implement preview generation
    raise HTTPException(status_code=501, detail="Preview generation not yet implemented")
