"""Rebuild FIFO job callbacks from persisted jobtype + info."""

from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

JobCallback = Callable[[], Awaitable[Any]]


def build_job_callback(app: Any, jobid: str, jobtype: str, info: dict[str, Any]) -> JobCallback:
    """Return an async zero-arg callback for a persisted job."""
    builders = _callback_builders()
    builder = builders.get(jobtype)
    if builder is None:
        raise ValueError(f"No handler registered for jobtype: {jobtype}")
    return builder(app, jobid, info)


def supported_jobtypes() -> list[str]:
    return sorted(_callback_builders().keys())


def _callback_builders():
    # Lazy registry to avoid import cycles at module load.
    return {
        "data-dictionary": _build_data_dictionary,
        "generate-csv": _build_generate_csv,
        "remove-csv-columns": _build_remove_csv_columns,
        "data-export": _build_data_export,
        "process-microdata": _build_process_microdata,
        "geospatial-layers-extraction": _build_geospatial_layers,
        "geospatial-data-extraction": _build_geospatial_data,
        "geospatial-metadata-extraction": _build_geospatial_metadata,
        "geospatial-batch-metadata-extraction": _build_geospatial_batch_metadata,
        "geospatial-metadata-extraction-with-images": _build_geospatial_metadata_with_images,
        "timeseries-import": _build_timeseries_import,
        "indicator-timeseries-import": _build_timeseries_import,
        "indicator-staging-import": _build_staging_import,
        "indicator-promote": _build_indicator_promote,
        "indicator-replace-from-csv": _build_replace_from_csv,
        "indicator-export-to-file": _build_export_to_file,
        "indicator-timeseries-recompute-ts": _build_recompute_time_derived,
    }


def _partial(handler, jobid: str, model_cls, info: dict[str, Any], *extra) -> JobCallback:
    params = model_cls.model_validate(info)
    return functools.partial(handler, jobid, params, *extra)


def _build_data_dictionary(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from main import write_data_dictionary_file
    from src.DictParams import DictParams

    return _partial(write_data_dictionary_file, jobid, DictParams, info)


def _build_generate_csv(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from main import write_csv_file_callback, FileInfo

    return _partial(write_csv_file_callback, jobid, FileInfo, info)


def _build_remove_csv_columns(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from main import remove_columns_from_csv_callback, RemoveColumnsParams

    return _partial(remove_columns_from_csv_callback, jobid, RemoveColumnsParams, info)


def _build_data_export(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from main import export_data_file
    from src.DictParams import DictParams

    return _partial(export_data_file, jobid, DictParams, info)


def _build_process_microdata(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from main import process_microdata_file, DataProcessingParams

    return _partial(process_microdata_file, jobid, DataProcessingParams, info)


def _build_geospatial_layers(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.geospatial_models import GeospatialImportRequest
    from src.routers.geospatial import process_geospatial_layers_job

    return _partial(process_geospatial_layers_job, jobid, GeospatialImportRequest, info)


def _build_geospatial_data(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.geospatial_models import GeospatialDataExtractionRequest
    from src.routers.geospatial import process_geospatial_data_job

    return _partial(process_geospatial_data_job, jobid, GeospatialDataExtractionRequest, info)


def _build_geospatial_metadata(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.geospatial_models import GeospatialMetadataRequest
    from src.routers.geospatial import process_geospatial_metadata_job

    return _partial(process_geospatial_metadata_job, jobid, GeospatialMetadataRequest, info)


def _build_geospatial_batch_metadata(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.geospatial_models import GeospatialBatchMetadataRequest
    from src.routers.geospatial import process_geospatial_batch_metadata_job

    return _partial(process_geospatial_batch_metadata_job, jobid, GeospatialBatchMetadataRequest, info)


def _build_geospatial_metadata_with_images(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.geospatial_models import GeospatialMetadataRequest
    from src.routers.geospatial import process_geospatial_metadata_with_images_job

    return _partial(process_geospatial_metadata_with_images_job, jobid, GeospatialMetadataRequest, info)


def _build_timeseries_import(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.timeseries_models import TimeseriesImportRequest
    from src.routers.timeseries import process_timeseries_import_job, timeseries_service

    params = TimeseriesImportRequest.model_validate(info)
    return functools.partial(process_timeseries_import_job, jobid, params, timeseries_service)


def _build_staging_import(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.timeseries_models import TimeseriesImportRequest
    from src.routers.timeseries import process_staging_import_job, timeseries_service

    params = TimeseriesImportRequest.model_validate(info)
    return functools.partial(process_staging_import_job, jobid, params, timeseries_service)


def _build_indicator_promote(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.timeseries_models import IndicatorPromoteRequest
    from src.routers.timeseries import process_indicator_promote_job, timeseries_service

    params = IndicatorPromoteRequest.model_validate(info)
    return functools.partial(process_indicator_promote_job, jobid, params, timeseries_service)


def _build_replace_from_csv(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.timeseries_models import IndicatorReplaceFromCsvRequest
    from src.routers.timeseries import process_replace_from_csv_job, timeseries_service

    params = IndicatorReplaceFromCsvRequest.model_validate(info)
    return functools.partial(process_replace_from_csv_job, jobid, params, timeseries_service)


def _build_export_to_file(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.timeseries_models import IndicatorExportToFileRequest
    from src.routers.timeseries import process_export_to_file_job, timeseries_service

    params = IndicatorExportToFileRequest.model_validate(info)
    return functools.partial(process_export_to_file_job, jobid, params, timeseries_service)


def _build_recompute_time_derived(app: Any, jobid: str, info: dict[str, Any]) -> JobCallback:
    from src.models.timeseries_models import RecomputeTimeDerivedRequest
    from src.routers.timeseries import process_recompute_time_derived_job, timeseries_service

    params = RecomputeTimeDerivedRequest.model_validate(info)
    return functools.partial(process_recompute_time_derived_job, jobid, params, timeseries_service)
