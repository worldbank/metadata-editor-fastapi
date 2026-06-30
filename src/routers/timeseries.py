from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Any, Dict, List, Optional
import os
import tempfile
import time
import datetime
import json
import functools
import logging
import traceback
import csv

import duckdb

from ..models.timeseries_models import (
	FacetValueCountItem,
	IndicatorChartAggregateRequest,
	IndicatorObservationKeyValidateRequest,
	IndicatorObservationKeyValidateResponse,
	IndicatorFacetValueCountsRequest,
	IndicatorFacetValueCountsResponse,
	CsvDistinctQueryResponse,
	CsvHeadersValidateResponse,
	IndicatorPromoteRequest,
	IndicatorExportToFileRequest,
	IndicatorReplaceFromCsvRequest,
	IndicatorTimeseriesImportRequest,
	PromoteTimeSpec,
	RecomputeTimeDerivedRequest,
	StagingDescribeColumn,
	StagingDescribeResponse,
	StagingDropResponse,
	TimeseriesDropResponse,
	StagingSampleResponse,
	TimeseriesColumnMeta,
	TimeseriesPageResponse,
	TimeseriesDistinctPair,
	TimeseriesDistinctPairsResponse,
	TimeseriesColumnFreqItem,
	TimeseriesColumnStat,
	TimeseriesColumnStatsResponse,
	StagingDistinctResponse,
	StagingDistinctValueCount,
	TimeseriesImportRequest,
	TimeseriesJobResponse,
	TimeseriesTablesResponse,
	TimeseriesAllTablesResponse,
	TimeseriesTableInfo,
	TimeseriesDescribeResponse,
	TimeseriesColumnInfo,
	TimeseriesDeleteRequest,
	TimeseriesDeleteResponse,
)
from ..services.timeseries_service import TimeseriesService
from ..utils.timeseries_utils import (
	STAGING_TABLE_NAME,
	TIMESERIES_TABLE_NAME,
	build_project_schema_name,
	build_table_name,
	escape_sql_string,
	fetch_table_column_rows,
	quote_identifier,
	resolve_column_name_case_insensitive,
	validate_csv_field_names,
	validate_csv_headers_exact_set,
	validate_dsd_columns_in_csv_headers,
)
from src.job_queue import enqueue_fifo_job
from src.utils.path_security import resolve_safe_path, resolve_safe_path_http
from ..utils.timeseries_ts_derived import (
	assert_staging_time_period_matches_implied_freq,
	build_derived_expressions,
	fetch_table_column_names_ordered,
	filter_user_columns,
	validate_and_resolve_time_spec,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/timeseries", tags=["timeseries"])

_default_db_path = os.getenv("TIMESERIES_DB_PATH", "db/timeseries.duckdb")
timeseries_service = TimeseriesService(_default_db_path)


def get_timeseries_service() -> TimeseriesService:
	return timeseries_service


def _read_csv_headers_or_raise(csv_path: str, delimiter: str) -> list:
	"""Read first row of CSV; validate header names. Returns list of header strings."""
	try:
		with open(csv_path, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter=delimiter)
			headers = next(reader, None)

			if not headers:
				raise HTTPException(status_code=400, detail="CSV file is empty or has no header row")

			is_valid, error_msg = validate_csv_field_names(headers)
			if not is_valid:
				raise HTTPException(status_code=400, detail=f"Invalid CSV headers: {error_msg}")

			return headers
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Error reading CSV headers: {str(e)}")


async def _enqueue_timeseries_import(
	jobid: str,
	jobtype: str,
	operation_type: str,
	import_request: TimeseriesImportRequest,
	service: TimeseriesService,
) -> TimeseriesJobResponse:
	from main import app

	current_time = datetime.datetime.now().isoformat()
	job_info = {
		"jobid": jobid,
		"jobtype": jobtype,
		"status": "queued",
		"created_at": current_time,
		"completed_at": None,
		"last_accessed": current_time,
		"info": {
			"project_id": import_request.project_id,
			"csv_path": import_request.csv_path,
			"delimiter": import_request.delimiter,
			"replace": import_request.replace,
		},
	}
	app.jobs[jobid] = job_info

	callback = functools.partial(
		process_timeseries_import_job,
		jobid,
		import_request,
		service,
	)
	await enqueue_fifo_job(app, jobid, callback)

	return TimeseriesJobResponse(
		message=f"{jobtype} job is queued",
		job_id=jobid,
		operation_type=operation_type,
		project_id=import_request.project_id,
	)


async def _enqueue_staging_import(
	jobid: str,
	import_request: TimeseriesImportRequest,
	service: TimeseriesService,
) -> TimeseriesJobResponse:
	from main import app

	current_time = datetime.datetime.now().isoformat()
	job_info = {
		"jobid": jobid,
		"jobtype": "indicator-staging-import",
		"status": "queued",
		"created_at": current_time,
		"completed_at": None,
		"last_accessed": current_time,
		"info": {
			"project_id": import_request.project_id,
			"csv_path": import_request.csv_path,
			"delimiter": import_request.delimiter,
			"target_table": STAGING_TABLE_NAME,
		},
	}
	app.jobs[jobid] = job_info

	callback = functools.partial(
		process_staging_import_job,
		jobid,
		import_request,
		service,
	)
	await enqueue_fifo_job(app, jobid, callback)

	return TimeseriesJobResponse(
		message="Indicator staging import job is queued",
		job_id=jobid,
		operation_type="indicator-staging-import",
		project_id=import_request.project_id,
	)


async def _enqueue_indicator_promote(
	jobid: str,
	promote_request: IndicatorPromoteRequest,
	service: TimeseriesService,
) -> TimeseriesJobResponse:
	from main import app

	current_time = datetime.datetime.now().isoformat()
	job_info = {
		"jobid": jobid,
		"jobtype": "indicator-promote",
		"status": "queued",
		"created_at": current_time,
		"completed_at": None,
		"last_accessed": current_time,
		"info": promote_request.model_dump(),
	}
	app.jobs[jobid] = job_info

	callback = functools.partial(
		process_indicator_promote_job,
		jobid,
		promote_request,
		service,
	)
	await enqueue_fifo_job(app, jobid, callback)

	return TimeseriesJobResponse(
		message="Indicator promote job is queued",
		job_id=jobid,
		operation_type="indicator-promote",
		project_id=promote_request.project_id,
	)


async def _enqueue_replace_from_csv(
	jobid: str,
	request: IndicatorReplaceFromCsvRequest,
	service: TimeseriesService,
) -> TimeseriesJobResponse:
	from main import app

	current_time = datetime.datetime.now().isoformat()
	job_info = {
		"jobid": jobid,
		"jobtype": "indicator-replace-from-csv",
		"status": "queued",
		"created_at": current_time,
		"completed_at": None,
		"last_accessed": current_time,
		"info": request.model_dump(),
	}
	app.jobs[jobid] = job_info

	callback = functools.partial(
		process_replace_from_csv_job,
		jobid,
		request,
		service,
	)
	await enqueue_fifo_job(app, jobid, callback)

	return TimeseriesJobResponse(
		message="Replace project timeseries from CSV is queued",
		job_id=jobid,
		operation_type="indicator-replace-from-csv",
		project_id=request.project_id,
	)


def _validate_indicator_archive_csv_path(project_id: str, output_csv_path: str) -> str:
	"""Ensure output path is editor data/indicator_data.csv under STORAGE_PATH."""
	if not output_csv_path or not str(output_csv_path).strip():
		raise HTTPException(status_code=400, detail="output_csv_path is required")

	path = resolve_safe_path_http(output_csv_path, label="output_csv_path")
	if os.path.basename(path) != "indicator_data.csv":
		raise HTTPException(
			status_code=400,
			detail="output_csv_path must be named indicator_data.csv",
		)
	if os.path.basename(os.path.dirname(path)) != "data":
		raise HTTPException(
			status_code=400,
			detail="output_csv_path must be under a data/ directory",
		)
	return path


async def _enqueue_export_to_file(
	jobid: str,
	request: IndicatorExportToFileRequest,
	service: TimeseriesService,
) -> TimeseriesJobResponse:
	from main import app

	current_time = datetime.datetime.now().isoformat()
	job_info = {
		"jobid": jobid,
		"jobtype": "indicator-export-to-file",
		"status": "queued",
		"created_at": current_time,
		"completed_at": None,
		"last_accessed": current_time,
		"info": request.model_dump(),
	}
	app.jobs[jobid] = job_info

	callback = functools.partial(
		process_export_to_file_job,
		jobid,
		request,
		service,
	)
	await enqueue_fifo_job(app, jobid, callback)

	return TimeseriesJobResponse(
		message="Export timeseries to project CSV is queued",
		job_id=jobid,
		operation_type="indicator-export-to-file",
		project_id=request.project_id,
	)


def _drop_staging_table(conn, schema_name: str) -> bool:
	qual_st = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"
	exists = conn.execute(
		"""
		SELECT 1
		FROM information_schema.tables
		WHERE table_schema = ? AND table_name = ?
		""",
		[schema_name, STAGING_TABLE_NAME],
	).fetchone()
	if not exists:
		return False
	conn.execute(f"DROP TABLE {qual_st}")
	return True


def _drop_timeseries_table(conn, schema_name: str) -> bool:
	qual_ts = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
	if not _timeseries_table_exists(conn, schema_name):
		return False
	conn.execute(f"DROP TABLE {qual_ts}")
	return True


def _csv_distinct_from_path(
	csv_path: str,
	delimiter: str,
	column: str,
	limit: int = 3000,
) -> tuple[str, List[StagingDistinctValueCount], int, bool]:
	"""Distinct string values for one column via read_csv_auto (no staging table)."""
	headers = _read_csv_headers_or_raise(csv_path, delimiter)
	resolved = resolve_column_name_case_insensitive(column, headers)
	if not resolved:
		raise HTTPException(
			status_code=400,
			detail=f"Column not found in CSV header: {column}",
		)

	escaped = escape_sql_string(csv_path)
	delim = delimiter.replace("'", "''")
	qcol = quote_identifier(resolved)
	limit = max(1, min(int(limit), 3000))

	with duckdb.connect(":memory:") as conn:
		count_row = conn.execute(
			f"SELECT COUNT(*) FROM read_csv_auto('{escaped}', delim='{delim}', header=true, all_varchar=true)"
		).fetchone()
		csv_row_count = int(count_row[0]) if count_row else 0

		rows = conn.execute(
			f"""
			SELECT CAST({qcol} AS VARCHAR) AS v, COUNT(*) AS c
			FROM read_csv_auto('{escaped}', delim='{delim}', header=true, all_varchar=true)
			WHERE {qcol} IS NOT NULL AND TRIM(CAST({qcol} AS VARCHAR)) <> ''
			GROUP BY 1
			ORDER BY c DESC, v ASC
			LIMIT ?
			""",
			[limit + 1],
		).fetchall()

	truncated = len(rows) > limit
	if truncated:
		rows = rows[:limit]

	items = [
		StagingDistinctValueCount(value=str(r[0]), count=int(r[1]))
		for r in rows
		if r[0] is not None
	]

	return resolved, items, csv_row_count, truncated


def _create_timeseries_from_staging(
	conn,
	schema_name: str,
	qual_st: str,
	qual_ts: str,
	indicator_resolved: str,
	indicator_value: str,
	time_spec: Optional[PromoteTimeSpec],
) -> None:
	"""Replace timeseries from filtered staging; optional _ts_year / _ts_freq from time_spec."""
	qcol = quote_identifier(indicator_resolved)
	filter_sql = f"SELECT * FROM {qual_st} WHERE CAST({qcol} AS VARCHAR) = ?"
	params = [str(indicator_value)]
	if time_spec is None:
		conn.execute(f"CREATE OR REPLACE TABLE {qual_ts} AS {filter_sql}", params)
		return
	raw, rt, rf = validate_and_resolve_time_spec(conn, schema_name, STAGING_TABLE_NAME, time_spec)
	year_sql, freq_sql = build_derived_expressions(
		"s",
		rt,
		rf,
		raw.get("time_period_format"),
		raw.get("implied_freq_code"),
		raw.get("default_freq_by_format"),
	)
	qy, qfreq = quote_identifier("_ts_year"), quote_identifier("_ts_freq")
	conn.execute(
		f"CREATE OR REPLACE TABLE {qual_ts} AS "
		f"SELECT s.*, {year_sql} AS {qy}, {freq_sql} AS {qfreq} "
		f"FROM ({filter_sql}) s",
		params,
	)


def _rebuild_timeseries_ts_derived_columns(
	conn,
	schema_name: str,
	time_spec: PromoteTimeSpec,
) -> int:
	"""
	Recompute _ts_year / _ts_freq on existing timeseries. Replaces table in-place via temp name.
	Returns row count of final timeseries.
	"""
	qual_ts = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
	raw, rt, rf = validate_and_resolve_time_spec(conn, schema_name, TIMESERIES_TABLE_NAME, time_spec)
	user_cols = filter_user_columns(
		fetch_table_column_names_ordered(conn, schema_name, TIMESERIES_TABLE_NAME)
	)
	if not user_cols:
		raise ValueError("timeseries has no user columns to preserve")
	year_sql, freq_sql = build_derived_expressions(
		"s",
		rt,
		rf,
		raw.get("time_period_format"),
		raw.get("implied_freq_code"),
		raw.get("default_freq_by_format"),
	)
	select_parts = [f"s.{quote_identifier(c)}" for c in user_cols]
	qy, qfreq = quote_identifier("_ts_year"), quote_identifier("_ts_freq")
	tmp_name = "timeseries__ts_rebuild"
	qual_tmp = f"{quote_identifier(schema_name)}.{quote_identifier(tmp_name)}"
	col_list = ", ".join(select_parts)
	conn.execute(
		f"CREATE OR REPLACE TABLE {qual_tmp} AS "
		f"SELECT {col_list}, {year_sql} AS {qy}, {freq_sql} AS {qfreq} "
		f"FROM {qual_ts} s"
	)
	conn.execute(f"DROP TABLE {qual_ts}")
	conn.execute(f"ALTER TABLE {qual_tmp} RENAME TO {quote_identifier(TIMESERIES_TABLE_NAME)}")
	row_count = conn.execute(f"SELECT COUNT(*) FROM {qual_ts}").fetchone()[0]
	return int(row_count)


async def _enqueue_recompute_time_derived(
	jobid: str,
	request: RecomputeTimeDerivedRequest,
	service: TimeseriesService,
) -> TimeseriesJobResponse:
	from main import app

	current_time = datetime.datetime.now().isoformat()
	job_info = {
		"jobid": jobid,
		"jobtype": "indicator-timeseries-recompute-ts",
		"status": "queued",
		"created_at": current_time,
		"completed_at": None,
		"last_accessed": current_time,
		"info": request.model_dump(),
	}
	app.jobs[jobid] = job_info

	callback = functools.partial(
		process_recompute_time_derived_job,
		jobid,
		request,
		service,
	)
	await enqueue_fifo_job(app, jobid, callback)

	return TimeseriesJobResponse(
		message="Timeseries _ts_year/_ts_freq recompute is queued",
		job_id=jobid,
		operation_type="indicator-timeseries-recompute-ts",
		project_id=request.project_id,
	)


def _duckdb_import_csv_table(
	db_path: str,
	project_id: str,
	csv_path: str,
	delimiter: str,
	replace: bool,
	table_name: str,
) -> dict:
	schema_name = build_project_schema_name(project_id)
	qualified_table = f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"
	clean_qualified_table = f"{schema_name}.{table_name}"
	escaped_path = escape_sql_string(csv_path)

	db_dir = os.path.dirname(db_path)
	if db_dir:
		os.makedirs(db_dir, exist_ok=True)

	with duckdb.connect(db_path) as conn:
		conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema_name)}")

		existing = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, table_name],
		).fetchone()

		if existing and not replace:
			raise ValueError(
				f"Table already exists: {clean_qualified_table}. Set replace=true to overwrite."
			)

		if existing and replace:
			conn.execute(f"DROP TABLE {qualified_table}")

		create_sql = (
			"CREATE TABLE "
			f"{qualified_table} AS "
			"SELECT * FROM read_csv_auto('"
			f"{escaped_path}"
			"', delim='"
			f"{delimiter}"
			"', header=true, all_varchar=true"
			")"
		)
		conn.execute(create_sql)

		row_count = conn.execute(
			f"SELECT COUNT(*) FROM {qualified_table}"
		).fetchone()[0]
		col_rows = fetch_table_column_rows(conn, schema_name, table_name)

	columns = [
		{
			"name": r[0],
			"type": r[1],
			"not_null": (r[2] == "NO") if len(r) > 2 else False,
			"default": None,
		}
		for r in col_rows
	]

	return {
		"project_id": project_id,
		"schema": schema_name,
		"table": table_name,
		"qualified_table": clean_qualified_table,
		"row_count": row_count,
		"column_count": len(columns),
		"columns": columns,
	}


@router.get("/")
async def timeseries_root():
	return {
		"message": "Timeseries API - Use /docs for API documentation",
		"db_path": timeseries_service.get_db_path(),
		"db_exists": os.path.exists(timeseries_service.get_db_path()),
		"endpoints": [
			"POST /timeseries/indicators/draft-queue",
			"GET /timeseries/indicators/draft/describe",
			"GET /timeseries/indicators/draft/sample",
			"GET /timeseries/indicators/draft/distinct",
			"DELETE /timeseries/indicators/draft",
			"POST /timeseries/indicators/timeseries/import-queue",
			"GET /timeseries/indicators/timeseries/page",
			"GET /timeseries/indicators/timeseries/distinct-pairs",
			"GET /timeseries/indicators/timeseries/column-stats",
			"POST /timeseries/indicators/timeseries/chart-aggregate",
			"POST /timeseries/indicators/timeseries/observation-key-validate",
			"POST /timeseries/indicators/timeseries/facet-value-counts",
			"GET /timeseries/indicators/timeseries/export",
			"POST /timeseries/indicators/timeseries/export-to-file-queue",
			"DELETE /timeseries/indicators/timeseries",
			"POST /timeseries/indicators/timeseries/recompute-queue",
		],
		"notes": {
			"draft_flow": "POST draft-queue → GET draft/distinct → POST timeseries/import-queue; poll GET /jobs/{jobid}",
			"draft": "project_{sid}.staging — raw CSV upload buffer before publishing to project_{sid}.timeseries",
			"legacy": "/timeseries/tables, /timeseries/describe, /timeseries/tables-queue, /timeseries/indicators/queue are deprecated",
		}
	}


@router.get("/describe", deprecated=True, response_model=TimeseriesDescribeResponse)
async def describe_timeseries_table(
	project_id: str,
	service: TimeseriesService = Depends(get_timeseries_service)
):
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	table_name = build_table_name(project_id)
	qualified_table = f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"
	clean_qualified_table = f"{schema_name}.{table_name}"

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, table_name]
		).fetchone()

		if not exists:
			raise HTTPException(status_code=404, detail="Timeseries table not found")

		row_count = conn.execute(
			f"SELECT COUNT(*) FROM {qualified_table}"
		).fetchone()[0]
		columns = conn.execute(
			f"PRAGMA table_info({quote_identifier(table_name)})"
		).fetchall()

	column_info = [
		TimeseriesColumnInfo(
			name=col[1],
			type=col[2],
			not_null=bool(col[3]),
			default=col[4]
		)
		for col in columns
	]

	return TimeseriesDescribeResponse(
		project_id=project_id,
		schema_name=schema_name,
		table=table_name,
		qualified_table=clean_qualified_table,
		row_count=row_count,
		column_count=len(columns),
		columns=column_info
	)


@router.get("/tables", deprecated=True, response_model=TimeseriesAllTablesResponse)
async def list_timeseries_tables(
	service: TimeseriesService = Depends(get_timeseries_service)
):
	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		return TimeseriesAllTablesResponse(total_tables=0, tables=[])

	with duckdb.connect(db_path) as conn:
		tables = conn.execute(
			"""
			SELECT table_schema
			FROM information_schema.tables
			WHERE table_schema LIKE 'project_%' AND table_name = 'timeseries'
			ORDER BY table_schema
			"""
		).fetchall()

	items = [
		TimeseriesTableInfo(
			project_id=table[0].replace('project_', ''),
			qualified_table=f"{table[0]}.timeseries"
		)
		for table in tables
	]

	return TimeseriesAllTablesResponse(total_tables=len(items), tables=items)


@router.delete("/tables", deprecated=True, response_model=TimeseriesDeleteResponse)
async def delete_timeseries_table(
	request: TimeseriesDeleteRequest,
	service: TimeseriesService = Depends(get_timeseries_service)
):
	if not request.project_id and not request.table_name:
		raise HTTPException(status_code=400, detail="Either project_id or table_name must be provided")

	if request.project_id and request.table_name:
		raise HTTPException(status_code=400, detail="Provide either project_id or table_name, not both")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	if request.project_id:
		if not request.project_id.isdigit():
			raise HTTPException(status_code=400, detail="project_id must be numeric")
		schema_name = build_project_schema_name(request.project_id)
		table_name = build_table_name(request.project_id)
	else:
		parts = request.table_name.split(".")
		if len(parts) != 2:
			raise HTTPException(status_code=400, detail="table_name must be in format 'schema.table'")
		schema_name, table_name = parts

	qualified_table = f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"
	clean_qualified_table = f"{schema_name}.{table_name}"

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, table_name]
		).fetchone()

		if not exists:
			raise HTTPException(status_code=404, detail=f"Table not found: {clean_qualified_table}")

		row_count = conn.execute(
			f"SELECT COUNT(*) FROM {qualified_table}"
		).fetchone()[0]

		conn.execute(f"DROP TABLE {qualified_table}")

	return TimeseriesDeleteResponse(
		message=f"Table {schema_name}.timeseries successfully deleted",
		schema_name=schema_name,
		table=table_name,
		qualified_table=f"{schema_name}.{table_name}",
		rows_deleted=row_count
	)


@router.post("/tables-queue", deprecated=True, response_model=TimeseriesJobResponse)
async def import_timeseries_table_queue(
	request: TimeseriesImportRequest,
	service: TimeseriesService = Depends(get_timeseries_service)
):
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	csv_path = resolve_safe_path_http(request.csv_path, label="csv_path")
	request = request.model_copy(update={"csv_path": csv_path})
	if not os.path.exists(csv_path):
		raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")

	_read_csv_headers_or_raise(csv_path, request.delimiter)

	# Check if table already exists before queuing
	schema_name = build_project_schema_name(request.project_id)
	table_name = build_table_name(request.project_id)
	db_path = service.get_db_path()

	with duckdb.connect(db_path) as conn:
		existing = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, table_name]
		).fetchone()

		if existing and not request.replace:
			clean_qualified_table = f"{schema_name}.{table_name}"
			raise HTTPException(
				status_code=409,
				detail=f"Table already exists: {clean_qualified_table}. Set replace=true to overwrite."
			)

	jobid = f"timeseries-import-{int(time.time() * 1000)}"
	return await _enqueue_timeseries_import(
		jobid,
		"timeseries-import",
		"timeseries-import",
		request,
		service,
	)


@router.post("/indicators/queue", deprecated=True, response_model=TimeseriesJobResponse)
async def import_indicator_timeseries_queue(
	request: IndicatorTimeseriesImportRequest,
	service: TimeseriesService = Depends(get_timeseries_service)
):
	"""
	Editor indicator pipeline: always replaces project_{sid}.timeseries.

	Optional dsd_columns lists DSD column names that must each appear in the CSV
	header (case-insensitive), matching indicator_dsd.name from PHP.
	"""
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	csv_path = resolve_safe_path_http(request.csv_path, label="csv_path")
	request = request.model_copy(update={"csv_path": csv_path})
	if not os.path.exists(csv_path):
		raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")

	headers = _read_csv_headers_or_raise(csv_path, request.delimiter)

	if request.dsd_columns:
		names = [c.name for c in request.dsd_columns]
		ok, err = validate_dsd_columns_in_csv_headers(headers, names)
		if not ok:
			raise HTTPException(status_code=400, detail=err)

	import_body = TimeseriesImportRequest(
		project_id=request.project_id,
		csv_path=csv_path,
		delimiter=request.delimiter,
		replace=True,
	)

	jobid = f"indicator-import-{int(time.time() * 1000)}"
	return await _enqueue_timeseries_import(
		jobid,
		"indicator-timeseries-import",
		"indicator-timeseries-import",
		import_body,
		service,
	)


@router.post("/indicators/draft-queue", response_model=TimeseriesJobResponse)
async def import_indicator_staging_queue(
	request: IndicatorTimeseriesImportRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Load CSV into project_{sid}.staging (draft buffer). Use GET .../draft/distinct
	then POST .../timeseries/import-queue to publish to project_{sid}.timeseries.
	"""
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	csv_path = resolve_safe_path_http(request.csv_path, label="csv_path")
	request = request.model_copy(update={"csv_path": csv_path})
	if not os.path.exists(csv_path):
		raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")

	headers = _read_csv_headers_or_raise(csv_path, request.delimiter)

	if request.dsd_columns:
		names = [c.name for c in request.dsd_columns]
		ok, err = validate_dsd_columns_in_csv_headers(headers, names)
		if not ok:
			raise HTTPException(status_code=400, detail=err)

	import_body = TimeseriesImportRequest(
		project_id=request.project_id,
		csv_path=csv_path,
		delimiter=request.delimiter,
		replace=True,
	)

	jobid = f"indicator-staging-{int(time.time() * 1000)}"
	return await _enqueue_staging_import(jobid, import_body, service)


@router.get("/indicators/draft/describe", response_model=StagingDescribeResponse)
async def indicator_staging_describe(
	project_id: str = Query(..., description="Editor sid"),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""Metadata for project_{sid}.staging (resume import UI)."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, STAGING_TABLE_NAME],
		).fetchone()

		if not exists:
			return StagingDescribeResponse(exists=False, row_count=0, columns=[])

		col_rows = fetch_table_column_rows(conn, schema_name, STAGING_TABLE_NAME)
		qual = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"
		row_count = conn.execute(f"SELECT COUNT(*) FROM {qual}").fetchone()[0]

	columns = [StagingDescribeColumn(name=str(r[0])) for r in col_rows]

	return StagingDescribeResponse(
		exists=True,
		row_count=int(row_count),
		columns=columns,
	)


def _cell_to_preview_str(v) -> str:
	if v is None:
		return ""
	if isinstance(v, bool):
		return "true" if v else "false"
	return str(v)


@router.get("/indicators/draft/sample", response_model=StagingSampleResponse)
async def indicator_staging_sample(
	project_id: str = Query(..., description="Editor sid"),
	limit: int = Query(20, ge=1, le=500),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""First `limit` rows from staging (data preview in import wizard)."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, STAGING_TABLE_NAME],
		).fetchone()

		if not exists:
			raise HTTPException(status_code=404, detail="Staging table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, STAGING_TABLE_NAME)
		col_names = [str(r[0]) for r in col_rows]
		if not col_names:
			return StagingSampleResponse(
				project_id=project_id,
				columns=[],
				rows=[],
				row_count_returned=0,
			)

		col_sql = ", ".join(quote_identifier(c) for c in col_names)
		rel = conn.execute(f"SELECT {col_sql} FROM {qual} LIMIT {int(limit)}")
		raw_rows = rel.fetchall()

	out_rows = []
	for tup in raw_rows:
		out_rows.append({col_names[i]: _cell_to_preview_str(tup[i]) for i in range(len(col_names))})

	return StagingSampleResponse(
		project_id=project_id,
		columns=col_names,
		rows=out_rows,
		row_count_returned=len(out_rows),
	)


@router.get("/indicators/draft/distinct", response_model=StagingDistinctResponse)
async def indicator_staging_distinct(
	project_id: str = Query(..., description="Editor sid"),
	column: str = Query(..., min_length=1, description="Physical column name in staging"),
	limit: int = Query(3000, ge=1, le=3000),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""Distinct non-null values in a staging column (for indicator id picker). Max 3000 values."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, STAGING_TABLE_NAME],
		).fetchone()

		if not exists:
			raise HTTPException(status_code=404, detail="Staging table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, STAGING_TABLE_NAME)
		names = [r[0] for r in col_rows]
		resolved = resolve_column_name_case_insensitive(column, names)
		if not resolved:
			raise HTTPException(
				status_code=400,
				detail=f"Column not found in staging: {column}",
			)

		q = quote_identifier(resolved)
		qual = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"
		staging_row_count = conn.execute(f"SELECT COUNT(*) FROM {qual}").fetchone()[0]
		cap = int(limit) + 1
		# Per-value counts; order by frequency so the dropdown surfaces common codes first
		rows = conn.execute(
			f"SELECT CAST({q} AS VARCHAR) AS v, COUNT(*) AS cnt FROM {qual} "
			f"WHERE {q} IS NOT NULL GROUP BY v ORDER BY cnt DESC, v ASC LIMIT {cap}"
		).fetchall()

	truncated = len(rows) > limit
	if truncated:
		rows = rows[:limit]

	items: list = []
	values: list = []
	for r in rows:
		if r[0] is None:
			continue
		v = str(r[0])
		c = int(r[1]) if len(r) > 1 and r[1] is not None else 0
		values.append(v)
		items.append(StagingDistinctValueCount(value=v, count=c))

	return StagingDistinctResponse(
		project_id=project_id,
		column_resolved=resolved,
		values=values,
		items=items,
		truncated=truncated,
		staging_row_count=int(staging_row_count),
	)


@router.delete("/indicators/draft", response_model=StagingDropResponse)
async def indicator_staging_drop(
	project_id: str = Query(..., description="Editor sid"),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""Drop project_{sid}.staging if it exists (after promote + PHP import)."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, STAGING_TABLE_NAME],
		).fetchone()

		if not exists:
			return StagingDropResponse(project_id=project_id, dropped=False)

		conn.execute(f"DROP TABLE {qual}")

	return StagingDropResponse(project_id=project_id, dropped=True)


@router.get("/indicators/csv/distinct", response_model=CsvDistinctQueryResponse)
async def indicator_csv_distinct(
	project_id: str = Query(..., description="Editor sid"),
	csv_path: str = Query(..., description="Absolute path to CSV on this host"),
	column: str = Query(..., min_length=1, description="Column name (indicator_id)"),
	delimiter: str = Query(",", min_length=1, max_length=1),
	limit: int = Query(3000, ge=1, le=3000),
):
	"""
	Distinct non-null values in a CSV column without loading staging.
	Used for the indicator_id picker before replace import.
	"""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")
	csv_path = resolve_safe_path_http(csv_path, label="csv_path")
	if not os.path.isfile(csv_path):
		raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")

	resolved, items, csv_row_count, truncated = _csv_distinct_from_path(
		csv_path, delimiter, column, limit
	)

	return CsvDistinctQueryResponse(
		project_id=project_id,
		column_resolved=resolved,
		values=[i.value for i in items],
		items=items,
		truncated=truncated,
		csv_row_count=csv_row_count,
	)


@router.get("/indicators/csv/validate-headers", response_model=CsvHeadersValidateResponse)
async def indicator_csv_validate_headers(
	project_id: str = Query(..., description="Editor sid"),
	csv_path: str = Query(..., description="Absolute path to CSV on this host"),
	expected_columns: str = Query(
		...,
		description="Comma-separated DSD column names (must match CSV header set exactly)",
	),
	delimiter: str = Query(",", min_length=1, max_length=1),
):
	"""Validate CSV header row against expected DSD columns (exact set, case-insensitive)."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")
	csv_path = resolve_safe_path_http(csv_path, label="csv_path")
	if not os.path.isfile(csv_path):
		raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")

	headers = _read_csv_headers_or_raise(csv_path, delimiter)
	names = [n.strip() for n in expected_columns.split(",") if n.strip()]
	ok, msg, missing, extra = validate_csv_headers_exact_set(headers, names)

	return CsvHeadersValidateResponse(
		project_id=project_id,
		valid=ok,
		message=msg,
		missing_in_csv=missing,
		extra_in_csv=extra,
		csv_headers=headers,
	)


@router.post("/indicators/timeseries/replace-from-csv-queue", response_model=TimeseriesJobResponse)
async def indicator_replace_from_csv_queue(
	request: IndicatorReplaceFromCsvRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Validate CSV headers, replace project timeseries from rows matching indicator_value.
	Uses staging only inside the job; staging is dropped on success or failure.
	"""
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")
	csv_path = resolve_safe_path_http(request.csv_path, label="csv_path")
	request = request.model_copy(update={"csv_path": csv_path})
	if not os.path.isfile(csv_path):
		raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")

	headers = _read_csv_headers_or_raise(csv_path, request.delimiter)
	names = [c.name for c in request.expected_columns]
	ok, msg, _, _ = validate_csv_headers_exact_set(headers, names)
	if not ok:
		raise HTTPException(status_code=400, detail=msg or "CSV headers do not match DSD columns")

	jobid = f"indicator-replace-csv-{int(time.time() * 1000)}"
	return await _enqueue_replace_from_csv(jobid, request, service)


@router.post("/indicators/timeseries/export-to-file-queue", response_model=TimeseriesJobResponse)
async def indicator_timeseries_export_to_file_queue(
	request: IndicatorExportToFileRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Export project_{sid}.timeseries to indicator_data.csv on the shared editor filesystem.
	Used after import so the project folder archive matches DuckDB without streaming CSV through PHP.
	"""
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	output_path = _validate_indicator_archive_csv_path(
		request.project_id,
		request.output_csv_path,
	)
	request = request.model_copy(update={"output_csv_path": output_path})

	jobid = f"indicator-export-file-{int(time.time() * 1000)}"
	return await _enqueue_export_to_file(jobid, request, service)


@router.delete("/indicators/timeseries", response_model=TimeseriesDropResponse)
async def indicator_timeseries_drop(
	project_id: str = Query(..., description="Editor sid"),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""Drop project_{sid}.timeseries if it exists (delete all published indicator data)."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, TIMESERIES_TABLE_NAME],
		).fetchone()

		if not exists:
			return TimeseriesDropResponse(project_id=project_id, dropped=False, row_count=0)

		row_count = conn.execute(f"SELECT COUNT(*) FROM {qual}").fetchone()[0]
		conn.execute(f"DROP TABLE {qual}")

	return TimeseriesDropResponse(project_id=project_id, dropped=True, row_count=int(row_count))


def _timeseries_table_exists(conn, schema_name: str) -> bool:
	row = conn.execute(
		"""
		SELECT 1
		FROM information_schema.tables
		WHERE table_schema = ? AND table_name = ?
		""",
		[schema_name, TIMESERIES_TABLE_NAME],
	).fetchone()
	return row is not None


def _fetch_timeseries_column_meta_ordered(conn, schema_name: str) -> list:
	"""Column metadata in table ordinal order."""
	try:
		return conn.execute(
			"""
			SELECT column_name, data_type, is_nullable
			FROM information_schema.columns
			WHERE table_schema = ? AND table_name = ?
			ORDER BY ordinal_position
			""",
			[schema_name, TIMESERIES_TABLE_NAME],
		).fetchall()
	except Exception:
		return fetch_table_column_rows(conn, schema_name, TIMESERIES_TABLE_NAME)


_TIMESERIES_PAGE_FILTERS_MAX_BYTES = 20000
_TIMESERIES_PAGE_MAX_FILTER_COLUMNS = 40
_TIMESERIES_PAGE_MAX_VALUES_PER_COLUMN = 500


def _parse_timeseries_page_filters_query(raw: Optional[str]) -> Dict[str, List[str]]:
	"""JSON object: physical column name -> list of string values (trimmed VARCHAR match, IN (...))."""
	if raw is None:
		return {}
	s = str(raw).strip()
	if s == "":
		return {}
	if len(s) > _TIMESERIES_PAGE_FILTERS_MAX_BYTES:
		raise HTTPException(status_code=400, detail="filters query parameter is too large")
	try:
		obj = json.loads(s)
	except json.JSONDecodeError:
		raise HTTPException(status_code=400, detail="filters must be valid JSON")
	if not isinstance(obj, dict):
		raise HTTPException(status_code=400, detail="filters must be a JSON object")
	if len(obj) > _TIMESERIES_PAGE_MAX_FILTER_COLUMNS:
		raise HTTPException(
			status_code=400,
			detail=f"at most {_TIMESERIES_PAGE_MAX_FILTER_COLUMNS} filter columns allowed",
		)
	out: Dict[str, List[str]] = {}
	for k, v in obj.items():
		key = str(k).strip()
		if not key:
			raise HTTPException(status_code=400, detail="filter column name cannot be empty")
		if not isinstance(v, list):
			raise HTTPException(status_code=400, detail=f"filter values for {key!r} must be a JSON array")
		if len(v) > _TIMESERIES_PAGE_MAX_VALUES_PER_COLUMN:
			raise HTTPException(
				status_code=400,
				detail=f"at most {_TIMESERIES_PAGE_MAX_VALUES_PER_COLUMN} values per filter column",
			)
		str_vals: List[str] = []
		for item in v:
			str_vals.append(str(item).strip())
		out[key] = str_vals
	return out


def _timeseries_page_where_from_filters(
	filters: Dict[str, List[str]],
	table_col_names: List[str],
) -> tuple[str, List[Any]]:
	where_parts: List[str] = []
	params: List[Any] = []
	for col_key, vals in filters.items():
		if not vals:
			continue
		col_res = resolve_column_name_case_insensitive(str(col_key).strip(), table_col_names)
		if not col_res:
			raise HTTPException(status_code=400, detail=f"filter column not in table: {col_key}")
		qcol = quote_identifier(col_res)
		placeholders = ", ".join(["?" for _ in vals])
		where_parts.append(f"trim(cast(t.{qcol} AS VARCHAR)) IN ({placeholders})")
		for v in vals:
			params.append(str(v).strip())
	if not where_parts:
		return "", []
	return " WHERE " + " AND ".join(where_parts), params


@router.get("/indicators/timeseries/page", response_model=TimeseriesPageResponse)
async def indicator_timeseries_page(
	project_id: str = Query(..., description="Editor sid"),
	offset: int = Query(0, ge=0),
	limit: int = Query(50, ge=1, le=200),
	filters: Optional[str] = Query(
		None,
		description=(
			"Optional JSON object: { \"PhysicalCol\": [\"v1\", \"v2\"], ... } — "
			"rows where each column’s trimmed string value is in its list (AND across columns). "
			"Empty arrays are ignored."
		),
	),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""Paginated rows from project_{sid}.timeseries for editor data explorer."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	filter_map = _parse_timeseries_page_filters_query(filters)

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(status_code=404, detail="Timeseries table not found for this project")

		col_rows = _fetch_timeseries_column_meta_ordered(conn, schema_name)
		meta_list: List[TimeseriesColumnMeta] = []
		col_names: List[str] = []
		for r in col_rows:
			cname = str(r[0])
			dt = str(r[1]) if len(r) > 1 and r[1] is not None else "UNKNOWN"
			nul = str(r[2]) if len(r) > 2 and r[2] is not None else "YES"
			col_names.append(cname)
			meta_list.append(TimeseriesColumnMeta(name=cname, data_type=dt, is_nullable=nul))

		table_col_names = list(col_names)

		if not col_names:
			total = int(conn.execute(f"SELECT COUNT(*) FROM {qual}").fetchone()[0])
			return TimeseriesPageResponse(
				project_id=project_id,
				columns=[],
				rows=[],
				total_row_count=total,
				offset=offset,
				limit=limit,
				row_count_returned=0,
			)

		where_sql, where_params = _timeseries_page_where_from_filters(filter_map, table_col_names)

		col_sql = ", ".join(quote_identifier(c) for c in col_names)
		count_sql = f"SELECT COUNT(*) FROM {qual} AS t{where_sql}"
		total = int(conn.execute(count_sql, where_params).fetchone()[0])
		data_sql = f"SELECT {col_sql} FROM {qual} AS t{where_sql} LIMIT {int(limit)} OFFSET {int(offset)}"
		rel = conn.execute(data_sql, list(where_params))
		raw_rows = rel.fetchall()

		out_rows = []
		for tup in raw_rows:
			out_rows.append({col_names[i]: _cell_to_preview_str(tup[i]) for i in range(len(col_names))})

	return TimeseriesPageResponse(
		project_id=project_id,
		columns=meta_list,
		rows=out_rows,
		total_row_count=total,
		offset=offset,
		limit=limit,
		row_count_returned=len(out_rows),
	)


@router.get("/indicators/timeseries/distinct-pairs", response_model=TimeseriesDistinctPairsResponse)
async def indicator_timeseries_distinct_pairs(
	project_id: str = Query(..., description="Editor sid"),
	code_column: str = Query(..., min_length=1, description="Physical column name in timeseries (code)"),
	label_column: Optional[str] = Query(
		None,
		description="Optional physical column for display label; if omitted, label equals code",
	),
	limit: int = Query(5000, ge=1, le=20000),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Distinct non-empty codes from project_{sid}.timeseries via DuckDB aggregation
	(one label per code using any_value when multiple labels exist).
	"""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
	cap = int(limit) + 1

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(status_code=404, detail="Timeseries table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, TIMESERIES_TABLE_NAME)
		names = [r[0] for r in col_rows]
		code_resolved = resolve_column_name_case_insensitive(code_column, names)
		if not code_resolved:
			raise HTTPException(
				status_code=400,
				detail=f"Column not found in timeseries: {code_column}",
			)
		code_q = quote_identifier(code_resolved)

		label_resolved: Optional[str] = None
		label_q: Optional[str] = None
		if label_column is not None and str(label_column).strip() != "":
			label_resolved = resolve_column_name_case_insensitive(str(label_column).strip(), names)
			if not label_resolved:
				raise HTTPException(
					status_code=400,
					detail=f"Column not found in timeseries: {label_column}",
				)
			label_q = quote_identifier(label_resolved)

		if label_q is None:
			sql = (
				f"SELECT DISTINCT trim(cast({code_q} AS VARCHAR)) AS code "
				f"FROM {qual} "
				f"WHERE {code_q} IS NOT NULL AND trim(cast({code_q} AS VARCHAR)) != '' "
				f"ORDER BY code LIMIT {cap}"
			)
			rows = conn.execute(sql).fetchall()
		else:
			# One row per trimmed code; any_value picks an arbitrary label among rows for that code
			label_expr = (
				f"CASE "
				f"WHEN {label_q} IS NULL THEN trim(cast({code_q} AS VARCHAR)) "
				f"WHEN trim(cast({label_q} AS VARCHAR)) = '' THEN trim(cast({code_q} AS VARCHAR)) "
				f"ELSE trim(cast({label_q} AS VARCHAR)) END"
			)
			sql = (
				f"SELECT trim(cast({code_q} AS VARCHAR)) AS code, any_value({label_expr}) AS label "
				f"FROM {qual} "
				f"WHERE {code_q} IS NOT NULL AND trim(cast({code_q} AS VARCHAR)) != '' "
				f"GROUP BY trim(cast({code_q} AS VARCHAR)) "
				f"ORDER BY code LIMIT {cap}"
			)
			rows = conn.execute(sql).fetchall()

	truncated = len(rows) > limit
	if truncated:
		rows = rows[:limit]

	items: List[TimeseriesDistinctPair] = []
	for r in rows:
		if not r or r[0] is None:
			continue
		c = str(r[0]).strip()
		if c == "":
			continue
		if label_q is None:
			items.append(TimeseriesDistinctPair(code=c, label=c))
		else:
			lb = str(r[1]).strip() if len(r) > 1 and r[1] is not None else c
			if lb == "":
				lb = c
			items.append(TimeseriesDistinctPair(code=c, label=lb))

	return TimeseriesDistinctPairsResponse(
		project_id=project_id,
		code_column_resolved=code_resolved,
		label_column_resolved=label_resolved,
		items=items,
		truncated=truncated,
	)


COLUMN_STATS_FREQ_MAX = 100
COLUMN_STATS_MAX_REQUEST = 200


def _timeseries_missing_predicate(col_q: str) -> str:
	"""SQL: NULL or blank/whitespace-only after trim(cast AS VARCHAR))."""
	return f"({col_q} IS NULL OR trim(cast({col_q} AS VARCHAR)) = '')"


@router.get("/indicators/timeseries/column-stats", response_model=TimeseriesColumnStatsResponse)
async def indicator_timeseries_column_stats(
	project_id: str = Query(..., description="Editor sid"),
	columns: Optional[str] = Query(
		None,
		description="Comma-separated physical column names; omit for all columns in timeseries",
	),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Non-null / null / distinct counts and top frequencies per column on project_{sid}.timeseries.
	Missing cell: NULL or trim(cast AS VARCHAR) = ''.
	"""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	computed_at = (
		datetime.datetime.now(datetime.timezone.utc)
		.replace(microsecond=0)
		.isoformat()
		.replace("+00:00", "Z")
	)

	out_stats: List[TimeseriesColumnStat] = []

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(status_code=404, detail="Timeseries table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, TIMESERIES_TABLE_NAME)
		table_col_names = [str(r[0]) for r in col_rows]

		if columns is None or str(columns).strip() == "":
			target_resolved = list(table_col_names)
		else:
			parts = [p.strip() for p in str(columns).split(",") if p.strip()]
			if len(parts) > COLUMN_STATS_MAX_REQUEST:
				raise HTTPException(
					status_code=400,
					detail=f"At most {COLUMN_STATS_MAX_REQUEST} columns per request",
				)
			target_resolved = []
			for p in parts:
				resolved = resolve_column_name_case_insensitive(p, table_col_names)
				if not resolved:
					raise HTTPException(
						status_code=400,
						detail=f"Column not found in timeseries: {p}",
					)
				if resolved not in target_resolved:
					target_resolved.append(resolved)

		for col_resolved in target_resolved:
			col_q = quote_identifier(col_resolved)
			miss = _timeseries_missing_predicate(col_q)
			agg_sql = (
				f"SELECT COUNT(*) AS rc, "
				f"SUM(CASE WHEN {miss} THEN 1 ELSE 0 END) AS n_null, "
				f"COUNT(DISTINCT CASE WHEN NOT ({miss}) THEN trim(cast({col_q} AS VARCHAR)) END) AS n_dist "
				f"FROM {qual}"
			)
			row = conn.execute(agg_sql).fetchone()
			rc = int(row[0] or 0)
			n_null = int(row[1] or 0)
			n_non_null = rc - n_null
			n_dist = int(row[2] or 0)

			cap = COLUMN_STATS_FREQ_MAX + 1
			freq_sql = (
				f"SELECT trim(cast({col_q} AS VARCHAR)) AS v, COUNT(*) AS c FROM {qual} "
				f"WHERE NOT ({miss}) GROUP BY 1 ORDER BY c DESC, v ASC LIMIT {cap}"
			)
			frows = conn.execute(freq_sql).fetchall()
			trunc = len(frows) > COLUMN_STATS_FREQ_MAX
			frows = frows[:COLUMN_STATS_FREQ_MAX]
			freq_list: List[TimeseriesColumnFreqItem] = []
			for fr in frows:
				if not fr or fr[0] is None:
					continue
				freq_list.append(
					TimeseriesColumnFreqItem(value=str(fr[0]), count=int(fr[1] or 0)),
				)

			out_stats.append(
				TimeseriesColumnStat(
					field=col_resolved,
					row_count=rc,
					non_null_count=n_non_null,
					null_count=n_null,
					distinct_count=n_dist,
					freq_max=COLUMN_STATS_FREQ_MAX,
					freq_truncated=trunc,
					freq=freq_list,
				),
			)

	return TimeseriesColumnStatsResponse(
		project_id=project_id,
		source="timeseries",
		computed_at=computed_at,
		columns=out_stats,
	)


_DUP_KEY_CHART_MSG = (
	"Duplicate rows for the same time period and dimension key (SDMX observation key must be "
	"unique). Remove duplicates in the source data or adjust the DSD."
)

_FACET_VALUE_COUNTS_MAX = 5000


@router.post(
	"/indicators/timeseries/facet-value-counts",
	response_model=IndicatorFacetValueCountsResponse,
)
async def indicator_timeseries_facet_value_counts(
	body: IndicatorFacetValueCountsRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
) -> IndicatorFacetValueCountsResponse:
	"""
	Row counts per distinct trimmed string value for each requested physical column (full table).
	Used for chart filter labels on initial page load; not sensitive to other UI filters.
	"""
	project_id = body.project_id.strip()
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	seen_req: set = set()
	ordered_req: List[str] = []
	for c in body.columns:
		k = str(c).strip()
		if not k or k in seen_req:
			continue
		seen_req.add(k)
		ordered_req.append(k)

	column_counts: Dict[str, List[FacetValueCountItem]] = {}
	columns_truncated: Dict[str, bool] = {}

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(status_code=404, detail="Timeseries table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, TIMESERIES_TABLE_NAME)
		table_col_names = [str(r[0]) for r in col_rows]

		for req_col in ordered_req:
			resolved = resolve_column_name_case_insensitive(req_col, table_col_names)
			if not resolved:
				raise HTTPException(status_code=400, detail=f"column not in timeseries table: {req_col}")

			cq = quote_identifier(resolved)
			expr = f"trim(cast(t.{cq} AS VARCHAR))"
			sql = f"""
				SELECT {expr} AS v, COUNT(*)::BIGINT AS c
				FROM {qual} AS t
				WHERE t.{cq} IS NOT NULL AND {expr} <> ''
				GROUP BY 1
				ORDER BY c DESC, v ASC
				LIMIT {_FACET_VALUE_COUNTS_MAX + 1}
			"""
			rows = conn.execute(sql).fetchall()
			truncated = len(rows) > _FACET_VALUE_COUNTS_MAX
			if truncated:
				rows = rows[:_FACET_VALUE_COUNTS_MAX]

			items: List[FacetValueCountItem] = []
			for row in rows:
				if not row or row[0] is None:
					continue
				sv = str(row[0]).strip()
				if sv == "":
					continue
				items.append(FacetValueCountItem(value=sv, count=int(row[1] or 0)))

			column_counts[resolved] = items
			columns_truncated[resolved] = truncated

	return IndicatorFacetValueCountsResponse(
		project_id=project_id,
		column_counts=column_counts,
		columns_truncated=columns_truncated,
	)


@router.post("/indicators/timeseries/chart-aggregate")
async def indicator_timeseries_chart_aggregate(
	body: IndicatorChartAggregateRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
) -> Dict[str, Any]:
	"""
	Return one row per observation key (time × slice dimensions); filters narrow rows.
	Raises 400 if duplicate keys exist after filters. No SUM/AVG on the measure.

	Each record: time_period, observation_value, series_key, slice_values (same order as
	metadata.slice_columns; used by PHP for labels). PHP strips slice_values and sets geography /
	series_key_label from DSD codelists.
	"""
	project_id = body.project_id.strip()
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(status_code=404, detail="Timeseries table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, TIMESERIES_TABLE_NAME)
		table_col_names = [str(r[0]) for r in col_rows]

		tc_res = resolve_column_name_case_insensitive(body.time_column, table_col_names)
		vc_res = resolve_column_name_case_insensitive(body.value_column, table_col_names)
		if not tc_res or not vc_res:
			raise HTTPException(status_code=400, detail="time_column or value_column not in timeseries table")

		slice_resolved: List[str] = []
		for sc in body.slice_columns:
			r = resolve_column_name_case_insensitive(str(sc).strip(), table_col_names)
			if not r:
				raise HTTPException(status_code=400, detail=f"slice column not in table: {sc}")
			if r == tc_res or r == vc_res:
				raise HTTPException(
					status_code=400,
					detail="slice_columns must not include time or value column",
				)
			slice_resolved.append(r)

		for fk in body.filters.keys():
			r = resolve_column_name_case_insensitive(str(fk).strip(), table_col_names)
			if not r:
				raise HTTPException(status_code=400, detail=f"filter key not in table: {fk}")

		use_ts_year = body.use_ts_year_for_time_filter
		if use_ts_year is None:
			use_ts_year = resolve_column_name_case_insensitive("_ts_year", table_col_names) is not None

		t_q = quote_identifier(tc_res)
		v_q = quote_identifier(vc_res)
		time_expr = f"trim(cast(t.{t_q} AS VARCHAR))"
		slice_trim: List[str] = []
		for r in slice_resolved:
			sq = quote_identifier(r)
			slice_trim.append(f"trim(cast(t.{sq} AS VARCHAR))")

		where_parts: List[str] = [
			f"t.{v_q} IS NOT NULL",
			f"trim(cast(t.{v_q} AS VARCHAR)) <> ''",
		]
		params: List[Any] = []

		for col_key, vals in body.filters.items():
			if not vals:
				continue
			col_res = resolve_column_name_case_insensitive(str(col_key).strip(), table_col_names)
			assert col_res
			qcol = quote_identifier(col_res)
			placeholders = ", ".join(["?" for _ in vals])
			where_parts.append(f"trim(cast(t.{qcol} AS VARCHAR)) IN ({placeholders})")
			for v in vals:
				params.append(str(v).strip())

		ts = body.time_period_start
		te = body.time_period_end
		if ts is not None and str(ts).strip() != "":
			if use_ts_year and resolve_column_name_case_insensitive("_ts_year", table_col_names):
				where_parts.append('t."_ts_year" >= ?')
				params.append(int(str(ts).strip()[:4]))
			else:
				where_parts.append(f"{time_expr} >= ?")
				params.append(str(ts).strip())
		if te is not None and str(te).strip() != "":
			if use_ts_year and resolve_column_name_case_insensitive("_ts_year", table_col_names):
				where_parts.append('t."_ts_year" <= ?')
				params.append(int(str(te).strip()[:4]))
			else:
				where_parts.append(f"{time_expr} <= ?")
				params.append(str(te).strip())

		where_sql = " AND ".join(where_parts)
		pbind = list(params)

		range_sql = f"SELECT MIN({time_expr}), MAX({time_expr}) FROM {qual} AS t WHERE {where_sql}"
		range_row = conn.execute(range_sql, pbind).fetchone()
		tp_min = range_row[0] if range_row else None
		tp_max = range_row[1] if range_row else None

		partition_exprs = [time_expr] + slice_trim
		group_by_keys = ", ".join(partition_exprs)
		dup_sql = f"""
			SELECT COUNT(*) FROM (
				SELECT 1 FROM {qual} AS t
				WHERE {where_sql}
				GROUP BY {group_by_keys}
				HAVING COUNT(*) > 1
			) AS dups
		"""
		dup_row = conn.execute(dup_sql, list(pbind)).fetchone()
		dup_n = int(dup_row[0]) if dup_row and dup_row[0] is not None else 0
		if dup_n > 0:
			raise HTTPException(status_code=400, detail=_DUP_KEY_CHART_MSG)

		select_dims = ", ".join(f"{st} AS dim_{i}" for i, st in enumerate(slice_trim))
		value_expr = f"TRY_CAST(t.{v_q} AS DOUBLE)"
		if select_dims:
			select_list = f"{time_expr} AS time_period, {select_dims}, {value_expr} AS observation_value"
		else:
			select_list = f"{time_expr} AS time_period, {value_expr} AS observation_value"

		order_parts = [time_expr] + slice_trim
		order_sql = ", ".join(order_parts)

		sql = f"""
			SELECT {select_list}
			FROM {qual} AS t
			WHERE {where_sql}
			ORDER BY {order_sql}
		"""

		cur = conn.execute(sql, list(pbind))
		rows = cur.fetchall()
		desc = cur.description or []
		col_names = [d[0] for d in desc]

		records: List[Dict[str, Any]] = []
		for row in rows:
			rec: Dict[str, Any] = {}
			dims: Dict[str, str] = {}
			for name, val in zip(col_names, row):
				if name == "time_period":
					rec["time_period"] = val if val is None else str(val)
				elif name == "observation_value":
					rec["observation_value"] = float(val) if val is not None else None
				elif name.startswith("dim_"):
					idx = int(name.split("_", 1)[1])
					sc = slice_resolved[idx]
					sval = "" if val is None else str(val)
					dims[sc] = sval
			if dims:
				rec["series_key"] = " | ".join(dims[c] for c in slice_resolved if c in dims)
				rec["slice_values"] = [dims[c] if c in dims else "" for c in slice_resolved]
			else:
				rec["series_key"] = "series"
				rec["slice_values"] = []
			records.append(rec)

	return {
		"records": records,
		"filter_options": {
			"time_period": {
				"min": str(tp_min) if tp_min is not None else None,
				"max": str(tp_max) if tp_max is not None else None,
				"values": [],
			}
		},
		"metadata": {
			"source": "duckdb",
			"observation_semantics": "unique_key",
			"time_column": tc_res,
			"value_column": vc_res,
			"slice_columns": slice_resolved,
			"total_records": len(records),
		},
	}


@router.post(
	"/indicators/timeseries/observation-key-validate",
	response_model=IndicatorObservationKeyValidateResponse,
)
async def indicator_timeseries_observation_key_validate(
	body: IndicatorObservationKeyValidateRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
) -> IndicatorObservationKeyValidateResponse:
	"""
	SDMX observation-key uniqueness for the full published timeseries: DuckDB GROUP BY / aggregates only.
	Semantics match chart-aggregate (trim(cast AS VARCHAR)) on time + slice columns; value non-null and non-empty.
	"""
	project_id = body.project_id.strip()
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(status_code=404, detail="Timeseries table not found for this project")

		col_rows = fetch_table_column_rows(conn, schema_name, TIMESERIES_TABLE_NAME)
		table_col_names = [str(r[0]) for r in col_rows]

		tc_res = resolve_column_name_case_insensitive(body.time_column, table_col_names)
		vc_res = resolve_column_name_case_insensitive(body.value_column, table_col_names)
		if not tc_res or not vc_res:
			raise HTTPException(status_code=400, detail="time_column or value_column not in timeseries table")

		slice_resolved: List[str] = []
		for sc in body.slice_columns:
			r = resolve_column_name_case_insensitive(str(sc).strip(), table_col_names)
			if not r:
				raise HTTPException(status_code=400, detail=f"slice column not in table: {sc}")
			if r == tc_res or r == vc_res:
				raise HTTPException(
					status_code=400,
					detail="slice_columns must not include time or value column",
				)
			slice_resolved.append(r)

		t_q = quote_identifier(tc_res)
		v_q = quote_identifier(vc_res)
		time_expr = f"trim(cast(t.{t_q} AS VARCHAR))"
		slice_trim: List[str] = []
		for r in slice_resolved:
			sq = quote_identifier(r)
			slice_trim.append(f"trim(cast(t.{sq} AS VARCHAR))")

		where_parts: List[str] = [
			f"t.{v_q} IS NOT NULL",
			f"trim(cast(t.{v_q} AS VARCHAR)) <> ''",
		]
		where_sql = " AND ".join(where_parts)

		partition_exprs = [time_expr] + slice_trim
		group_by_keys = ", ".join(partition_exprs)

		total_row = conn.execute(f"SELECT COUNT(*)::BIGINT FROM {qual} AS t").fetchone()
		table_total = int(total_row[0]) if total_row and total_row[0] is not None else 0

		rw_row = conn.execute(
			f"SELECT COUNT(*)::BIGINT FROM {qual} AS t WHERE {where_sql}",
		).fetchone()
		rows_with_value = int(rw_row[0]) if rw_row and rw_row[0] is not None else 0

		uniq_sql = f"""
			SELECT COUNT(*)::BIGINT FROM (
				SELECT 1 FROM {qual} AS t
				WHERE {where_sql}
				GROUP BY {group_by_keys}
			) AS u
		"""
		uniq_row = conn.execute(uniq_sql).fetchone()
		unique_keys = int(uniq_row[0]) if uniq_row and uniq_row[0] is not None else 0

		dup_sql = f"""
			SELECT COUNT(*)::BIGINT FROM (
				SELECT 1 FROM {qual} AS t
				WHERE {where_sql}
				GROUP BY {group_by_keys}
				HAVING COUNT(*) > 1
			) AS dups
		"""
		dup_row = conn.execute(dup_sql).fetchone()
		dup_groups = int(dup_row[0]) if dup_row and dup_row[0] is not None else 0

		duplicate_rows = rows_with_value - unique_keys
		if duplicate_rows < 0:
			duplicate_rows = 0

	return IndicatorObservationKeyValidateResponse(
		project_id=project_id,
		time_column=tc_res,
		value_column=vc_res,
		slice_columns=slice_resolved,
		table_total_row_count=table_total,
		rows_with_observation_value=rows_with_value,
		unique_observation_key_count=unique_keys,
		duplicate_key_group_count=dup_groups,
		duplicate_row_count=duplicate_rows,
		source="duckdb",
	)


@router.get("/indicators/timeseries/export")
async def indicator_timeseries_export_csv(
	background_tasks: BackgroundTasks,
	project_id: str = Query(..., description="Editor sid"),
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""Export full project_{sid}.timeseries as CSV (attachment)."""
	if not project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(project_id)
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

	tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix=f"ts_{project_id}_")
	tmp_path = tmp.name
	tmp.close()
	path_norm = tmp_path.replace("\\", "/").replace("'", "''")

	try:
		with duckdb.connect(db_path) as conn:
			if not _timeseries_table_exists(conn, schema_name):
				raise HTTPException(status_code=404, detail="Timeseries table not found for this project")
			conn.execute(
				f"COPY (SELECT * FROM {qual}) TO '{path_norm}' (HEADER, DELIMITER ',')"
			)
	except HTTPException:
		if os.path.isfile(tmp_path):
			os.unlink(tmp_path)
		raise
	except Exception as e:
		if os.path.isfile(tmp_path):
			os.unlink(tmp_path)
		raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}") from e

	def _unlink_if_exists(p: str) -> None:
		if os.path.isfile(p):
			os.unlink(p)

	background_tasks.add_task(_unlink_if_exists, tmp_path)
	filename = f"indicator_timeseries_{project_id}.csv"
	return FileResponse(
		tmp_path,
		media_type="text/csv; charset=utf-8",
		filename=filename,
	)


@router.post("/indicators/timeseries/import-queue", response_model=TimeseriesJobResponse)
async def indicator_promote_queue(
	request: IndicatorPromoteRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Replace project_{sid}.timeseries with rows from staging where indicator_column = indicator_value.
	"""
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(request.project_id)

	with duckdb.connect(db_path) as conn:
		exists = conn.execute(
			"""
			SELECT 1
			FROM information_schema.tables
			WHERE table_schema = ? AND table_name = ?
			""",
			[schema_name, STAGING_TABLE_NAME],
		).fetchone()

		if not exists:
			raise HTTPException(status_code=404, detail="Staging table not found for this project")

	jobid = f"indicator-promote-{int(time.time() * 1000)}"
	return await _enqueue_indicator_promote(jobid, request, service)


@router.post("/indicators/timeseries/recompute-queue", response_model=TimeseriesJobResponse)
async def indicator_timeseries_recompute_time_derived_queue(
	request: RecomputeTimeDerivedRequest,
	service: TimeseriesService = Depends(get_timeseries_service),
):
	"""
	Queue recomputation of _ts_year / _ts_freq on existing project_{sid}.timeseries
	using the supplied time_spec (same shape as import-queue time_spec).
	Staging is not modified.
	"""
	if not request.project_id.isdigit():
		raise HTTPException(status_code=400, detail="project_id must be numeric")

	db_path = service.get_db_path()
	if not os.path.exists(db_path):
		raise HTTPException(status_code=404, detail="Timeseries database not found")

	schema_name = build_project_schema_name(request.project_id)

	with duckdb.connect(db_path) as conn:
		if not _timeseries_table_exists(conn, schema_name):
			raise HTTPException(
				status_code=404,
				detail="Timeseries table not found for this project. Promote staging first.",
			)

	jobid = f"indicator-recompute-ts-{int(time.time() * 1000)}"
	return await _enqueue_recompute_time_derived(jobid, request, service)


async def process_timeseries_import_job(
	jobid: str,
	request: TimeseriesImportRequest,
	service: TimeseriesService,
):
	from main import app

	app.jobs[jobid]["status"] = "processing"

	try:
		csv_path = resolve_safe_path(request.csv_path)
		result = _duckdb_import_csv_table(
			service.get_db_path(),
			request.project_id,
			csv_path,
			request.delimiter,
			request.replace,
			build_table_name(request.project_id),
		)

		app.jobs[jobid]["status"] = "done"
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

		os.makedirs("jobs", exist_ok=True)
		job_result_path = os.path.join("jobs", f"{jobid}.json")
		with open(job_result_path, "w") as outfile:
			json.dump(result, outfile)

		logger.info(
			"Timeseries import job %s completed: %s",
			jobid,
			result.get("qualified_table"),
		)

	except Exception as e:
		logger.error("Timeseries import job %s failed: %s", jobid, str(e))
		app.jobs[jobid]["status"] = "error"
		app.jobs[jobid]["error"] = str(e)
		app.jobs[jobid]["error_details"] = {
			"function": "process_timeseries_import_job",
			"jobid": jobid,
			"traceback": traceback.format_exc(),
		}
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()


async def process_staging_import_job(
	jobid: str,
	request: TimeseriesImportRequest,
	service: TimeseriesService,
):
	from main import app

	app.jobs[jobid]["status"] = "processing"

	try:
		csv_path = resolve_safe_path(request.csv_path)
		result = _duckdb_import_csv_table(
			service.get_db_path(),
			request.project_id,
			csv_path,
			request.delimiter,
			True,
			STAGING_TABLE_NAME,
		)

		app.jobs[jobid]["status"] = "done"
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

		os.makedirs("jobs", exist_ok=True)
		job_result_path = os.path.join("jobs", f"{jobid}.json")
		with open(job_result_path, "w") as outfile:
			json.dump(result, outfile)

		logger.info(
			"Staging import job %s completed: %s",
			jobid,
			result.get("qualified_table"),
		)

	except Exception as e:
		logger.error("Staging import job %s failed: %s", jobid, str(e))
		app.jobs[jobid]["status"] = "error"
		app.jobs[jobid]["error"] = str(e)
		app.jobs[jobid]["error_details"] = {
			"function": "process_staging_import_job",
			"jobid": jobid,
			"traceback": traceback.format_exc(),
		}
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()


async def process_indicator_promote_job(
	jobid: str,
	request: IndicatorPromoteRequest,
	service: TimeseriesService,
):
	from main import app

	app.jobs[jobid]["status"] = "processing"

	try:
		db_path = service.get_db_path()
		schema_name = build_project_schema_name(request.project_id)
		qual_st = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"
		qual_ts = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

		with duckdb.connect(db_path) as conn:
			exists = conn.execute(
				"""
				SELECT 1
				FROM information_schema.tables
				WHERE table_schema = ? AND table_name = ?
				""",
				[schema_name, STAGING_TABLE_NAME],
			).fetchone()

			if not exists:
				raise ValueError("Staging table not found. Run staging import first.")

			col_rows = fetch_table_column_rows(conn, schema_name, STAGING_TABLE_NAME)
			names = [r[0] for r in col_rows]
			resolved = resolve_column_name_case_insensitive(request.indicator_column, names)
			if not resolved:
				raise ValueError(f"Column not found in staging: {request.indicator_column}")

			assert_staging_time_period_matches_implied_freq(
				conn,
				schema_name,
				STAGING_TABLE_NAME,
				qual_st,
				request.time_spec,
				resolved,
				str(request.indicator_value),
			)

			_create_timeseries_from_staging(
				conn,
				schema_name,
				qual_st,
				qual_ts,
				resolved,
				str(request.indicator_value),
				request.time_spec,
			)

			row_count = conn.execute(f"SELECT COUNT(*) FROM {qual_ts}").fetchone()[0]
			staging_count = conn.execute(f"SELECT COUNT(*) FROM {qual_st}").fetchone()[0]

		result = {
			"project_id": request.project_id,
			"schema": schema_name,
			"timeseries_qualified": f"{schema_name}.{TIMESERIES_TABLE_NAME}",
			"staging_qualified": f"{schema_name}.{STAGING_TABLE_NAME}",
			"row_count": int(row_count),
			"staging_row_count": int(staging_count),
			"indicator_column_resolved": resolved,
			"indicator_value": request.indicator_value,
		}

		app.jobs[jobid]["status"] = "done"
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

		os.makedirs("jobs", exist_ok=True)
		job_result_path = os.path.join("jobs", f"{jobid}.json")
		with open(job_result_path, "w") as outfile:
			json.dump(result, outfile)

		logger.info("Promote job %s completed: %s rows in timeseries", jobid, row_count)

	except Exception as e:
		logger.error("Promote job %s failed: %s", jobid, str(e))
		app.jobs[jobid]["status"] = "error"
		app.jobs[jobid]["error"] = str(e)
		app.jobs[jobid]["error_details"] = {
			"function": "process_indicator_promote_job",
			"jobid": jobid,
			"traceback": traceback.format_exc(),
		}
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()


async def process_recompute_time_derived_job(
	jobid: str,
	request: RecomputeTimeDerivedRequest,
	service: TimeseriesService,
):
	from main import app

	app.jobs[jobid]["status"] = "processing"

	try:
		db_path = service.get_db_path()
		schema_name = build_project_schema_name(request.project_id)
		qual_ts = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

		with duckdb.connect(db_path) as conn:
			if not _timeseries_table_exists(conn, schema_name):
				raise ValueError("Timeseries table not found. Promote staging first.")

			row_count = _rebuild_timeseries_ts_derived_columns(conn, schema_name, request.time_spec)

		result = {
			"project_id": request.project_id,
			"schema": schema_name,
			"timeseries_qualified": f"{schema_name}.{TIMESERIES_TABLE_NAME}",
			"row_count": int(row_count),
		}

		app.jobs[jobid]["status"] = "done"
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

		os.makedirs("jobs", exist_ok=True)
		job_result_path = os.path.join("jobs", f"{jobid}.json")
		with open(job_result_path, "w") as outfile:
			json.dump(result, outfile)

		logger.info("Recompute ts-derived job %s completed: %s rows", jobid, row_count)

	except Exception as e:
		logger.error("Recompute ts-derived job %s failed: %s", jobid, str(e))
		app.jobs[jobid]["status"] = "error"
		app.jobs[jobid]["error"] = str(e)
		app.jobs[jobid]["error_details"] = {
			"function": "process_recompute_time_derived_job",
			"jobid": jobid,
			"traceback": traceback.format_exc(),
		}
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()


async def process_replace_from_csv_job(
	jobid: str,
	request: IndicatorReplaceFromCsvRequest,
	service: TimeseriesService,
):
	from main import app

	app.jobs[jobid]["status"] = "processing"
	db_path = service.get_db_path()
	schema_name = build_project_schema_name(request.project_id)

	try:
		csv_path = resolve_safe_path(request.csv_path)
		headers = _read_csv_headers_or_raise(csv_path, request.delimiter)
		names = [c.name for c in request.expected_columns]
		ok, msg, _, _ = validate_csv_headers_exact_set(headers, names)
		if not ok:
			raise ValueError(msg or "CSV headers do not match DSD columns")

		with duckdb.connect(db_path) as conn:
			conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema_name)}")
			_drop_timeseries_table(conn, schema_name)
			_drop_staging_table(conn, schema_name)

		_duckdb_import_csv_table(
			db_path,
			request.project_id,
			csv_path,
			request.delimiter,
			True,
			STAGING_TABLE_NAME,
		)

		qual_st = f"{quote_identifier(schema_name)}.{quote_identifier(STAGING_TABLE_NAME)}"
		qual_ts = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"

		with duckdb.connect(db_path) as conn:
			col_rows = fetch_table_column_rows(conn, schema_name, STAGING_TABLE_NAME)
			col_names = [r[0] for r in col_rows]
			resolved = resolve_column_name_case_insensitive(
				request.indicator_column, col_names
			)
			if not resolved:
				raise ValueError(
					f"Column not found in CSV: {request.indicator_column}"
				)

			assert_staging_time_period_matches_implied_freq(
				conn,
				schema_name,
				STAGING_TABLE_NAME,
				qual_st,
				request.time_spec,
				resolved,
				str(request.indicator_value),
			)

			_create_timeseries_from_staging(
				conn,
				schema_name,
				qual_st,
				qual_ts,
				resolved,
				str(request.indicator_value),
				request.time_spec,
			)

			row_count = int(conn.execute(f"SELECT COUNT(*) FROM {qual_ts}").fetchone()[0])
			staging_dropped = _drop_staging_table(conn, schema_name)

		result = {
			"project_id": request.project_id,
			"row_count": row_count,
			"indicator_column_resolved": resolved,
			"indicator_value": request.indicator_value,
			"staging_dropped": staging_dropped,
		}

		app.jobs[jobid]["status"] = "done"
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

		os.makedirs("jobs", exist_ok=True)
		with open(os.path.join("jobs", f"{jobid}.json"), "w") as outfile:
			json.dump(result, outfile)

		logger.info(
			"Replace-from-csv job %s completed: %s rows",
			jobid,
			row_count,
		)

	except Exception as e:
		logger.error("Replace-from-csv job %s failed: %s", jobid, str(e))
		try:
			with duckdb.connect(db_path) as conn:
				conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema_name)}")
				_drop_timeseries_table(conn, schema_name)
				_drop_staging_table(conn, schema_name)
		except Exception as cleanup_err:
			logger.warning(
				"Replace-from-csv cleanup failed for %s: %s",
				jobid,
				cleanup_err,
			)

		app.jobs[jobid]["status"] = "error"
		app.jobs[jobid]["error"] = str(e)
		app.jobs[jobid]["error_details"] = {
			"function": "process_replace_from_csv_job",
			"jobid": jobid,
			"traceback": traceback.format_exc(),
		}
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()


async def process_export_to_file_job(
	jobid: str,
	request: IndicatorExportToFileRequest,
	service: TimeseriesService,
):
	from main import app

	app.jobs[jobid]["status"] = "processing"
	db_path = service.get_db_path()
	schema_name = build_project_schema_name(request.project_id)
	output_path = _validate_indicator_archive_csv_path(
		request.project_id,
		request.output_csv_path,
	)
	tmp_path = None

	try:
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		if os.path.isfile(output_path):
			os.unlink(output_path)

		qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
		tmp_path = output_path + ".export_" + str(int(time.time() * 1000))
		path_norm = tmp_path.replace("\\", "/").replace("'", "''")

		with duckdb.connect(db_path) as conn:
			if not _timeseries_table_exists(conn, schema_name):
				raise ValueError("Timeseries table not found for this project")
			row_count = int(conn.execute(f"SELECT COUNT(*) FROM {qual}").fetchone()[0])
			conn.execute(
				f"COPY (SELECT * FROM {qual}) TO '{path_norm}' (HEADER, DELIMITER ',')"
			)

		if not os.path.isfile(tmp_path):
			raise ValueError("Export did not create output file")

		os.replace(tmp_path, output_path)
		bytes_written = int(os.path.getsize(output_path))

		result = {
			"project_id": request.project_id,
			"output_csv_path": output_path,
			"row_count": row_count,
			"bytes_written": bytes_written,
		}

		app.jobs[jobid]["status"] = "done"
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

		os.makedirs("jobs", exist_ok=True)
		with open(os.path.join("jobs", f"{jobid}.json"), "w") as outfile:
			json.dump(result, outfile)

		logger.info(
			"Export-to-file job %s completed: %s rows -> %s",
			jobid,
			row_count,
			output_path,
		)

	except Exception as e:
		logger.error("Export-to-file job %s failed: %s", jobid, str(e))
		if tmp_path and os.path.isfile(tmp_path):
			try:
				os.unlink(tmp_path)
			except OSError:
				pass

		app.jobs[jobid]["status"] = "error"
		app.jobs[jobid]["error"] = str(e)
		app.jobs[jobid]["error_details"] = {
			"function": "process_export_to_file_job",
			"jobid": jobid,
			"traceback": traceback.format_exc(),
		}
		app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
