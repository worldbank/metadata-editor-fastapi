"""
Indicator CSV import into DuckDB.

Blocking functions: callers run them in a worker thread (asyncio.to_thread) so a large CSV
does not stall the event loop.

The CSV on disk is never rewritten. Header renames (column_names) and column selection
(select_columns) are applied while DuckDB reads the file. The project timeseries table is
replaced atomically: a failed import leaves the previously published data in place.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb

from ..utils.timeseries_ts_derived import (
	assert_staging_time_period_matches_implied_freq,
	build_derived_expressions,
	validate_and_resolve_time_spec,
)
from ..utils.timeseries_utils import (
	STAGING_TABLE_NAME,
	TIMESERIES_TABLE_NAME,
	build_project_schema_name,
	escape_sql_string,
	quote_identifier,
	resolve_column_name_case_insensitive,
	validate_csv_field_names,
	validate_csv_headers_exact_set,
)

NEW_TIMESERIES_TABLE_NAME = "timeseries__import_new"
DEFAULT_NOTE_TUPLE_LIMIT = 20000


class CsvImportError(ValueError):
	"""Import input problem (bad headers, missing columns, file changed)."""


def read_csv_header_row(csv_path: str, delimiter: str) -> List[str]:
	"""Raw first row of the CSV (BOM stripped). No name validation."""
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		headers = next(csv.reader(f, delimiter=delimiter), None)
	if not headers:
		raise CsvImportError("CSV file is empty or has no header row")
	headers[0] = headers[0].lstrip("﻿")
	return headers


def resolve_csv_column_names(
	csv_path: str,
	delimiter: str,
	column_names: Optional[Sequence[str]],
) -> List[str]:
	"""
	Column names DuckDB will use for the CSV.

	column_names: names for every CSV column in file order (the editor's normalized headers).
	When omitted, the file's own header row is used.
	"""
	headers = read_csv_header_row(csv_path, delimiter)
	if column_names:
		names = [str(n) for n in column_names]
		if len(names) != len(headers):
			raise CsvImportError(
				f"column_names has {len(names)} names but the CSV has {len(headers)} columns"
			)
	else:
		names = headers
	ok, msg = validate_csv_field_names(names)
	if not ok:
		raise CsvImportError(f"Invalid CSV headers: {msg}")
	return names


def csv_source_sql(csv_path: str, delimiter: str, column_names: Optional[Sequence[str]]) -> str:
	"""read_csv_auto(...) table function; all values as VARCHAR; optional column renames."""
	escaped = escape_sql_string(csv_path)
	delim = delimiter.replace("'", "''")
	sql = f"read_csv_auto('{escaped}', delim='{delim}', header=true, all_varchar=true"
	if column_names:
		quoted = ", ".join("'" + str(n).replace("'", "''") + "'" for n in column_names)
		sql += f", names=[{quoted}]"
	return sql + ")"


def assert_file_unchanged(csv_path: str, expected_size: Optional[int], expected_mtime: Optional[int]) -> None:
	"""Refuse to import a file that changed after the editor validated it."""
	if expected_size is None and expected_mtime is None:
		return
	st = os.stat(csv_path)
	if expected_size is not None and int(st.st_size) != int(expected_size):
		raise CsvImportError("CSV file changed after it was validated. Upload and validate it again.")
	if expected_mtime is not None and int(st.st_mtime) != int(expected_mtime):
		raise CsvImportError("CSV file changed after it was validated. Upload and validate it again.")


def csv_distinct_values(
	csv_path: str,
	delimiter: str,
	column: str,
	limit: int = 3000,
	column_names: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Tuple[str, int]], int, bool]:
	"""
	Distinct non-blank values of one CSV column with counts, in one scan of the file.

	Returns (resolved_column, [(value, count)], csv_row_count, truncated).
	"""
	names = resolve_csv_column_names(csv_path, delimiter, column_names)
	resolved = resolve_column_name_case_insensitive(column, names)
	if not resolved:
		raise CsvImportError(f"Column not found in CSV header: {column}")

	limit = max(1, min(int(limit), 3000))
	qcol = quote_identifier(resolved)
	src = csv_source_sql(csv_path, delimiter, column_names)

	with duckdb.connect(":memory:") as conn:
		conn.execute(
			f"CREATE TEMP TABLE g AS SELECT CAST({qcol} AS VARCHAR) AS v, COUNT(*) AS c FROM {src} GROUP BY 1"
		)
		total = conn.execute("SELECT COALESCE(SUM(c), 0) FROM g").fetchone()[0]
		rows = conn.execute(
			"""
			SELECT v, c FROM g
			WHERE v IS NOT NULL AND TRIM(v) <> ''
			ORDER BY c DESC, v ASC
			LIMIT ?
			""",
			[limit + 1],
		).fetchall()

	truncated = len(rows) > limit
	items = [(str(r[0]), int(r[1])) for r in rows[:limit]]
	return resolved, items, int(total), truncated


def _table_exists(conn, schema_name: str, table_name: str) -> bool:
	row = conn.execute(
		"SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
		[schema_name, table_name],
	).fetchone()
	return row is not None


def _drop_if_exists(conn, schema_name: str, table_name: str) -> bool:
	if not _table_exists(conn, schema_name, table_name):
		return False
	conn.execute(f"DROP TABLE {quote_identifier(schema_name)}.{quote_identifier(table_name)}")
	return True


def _attr(obj: Any, name: str, default: Any = None) -> Any:
	if isinstance(obj, dict):
		return obj.get(name, default)
	return getattr(obj, name, default)


def _build_select(
	request: Any,
	csv_names: List[str],
) -> Tuple[List[str], List[str], Dict[str, Any], Dict[str, str]]:
	"""
	Build staging SELECT parts.

	Returns (select_parts, timeseries_columns, note_specs_resolved, source_expr_by_target).
	timeseries_columns: target names kept in the timeseries table.
	source_expr_by_target: upper(target) -> SQL expression read from the CSV (for filtering while loading).
	"""
	select_columns = _attr(request, "select_columns")
	note_columns = _attr(request, "note_columns") or []

	parts: List[str] = []
	ts_columns: List[str] = []
	taken: Dict[str, str] = {}  # upper target -> target
	source_expr: Dict[str, str] = {}

	if select_columns:
		missing: List[str] = []
		for col in select_columns:
			target = str(_attr(col, "name")).strip()
			source = _attr(col, "source")
			if not target:
				raise CsvImportError("select_columns entry without a name")
			if target.upper() in taken:
				continue
			if source is None or str(source).strip() == "":
				parts.append(f"CAST(NULL AS VARCHAR) AS {quote_identifier(target)}")
				source_expr[target.upper()] = "CAST(NULL AS VARCHAR)"
			else:
				physical = resolve_column_name_case_insensitive(str(source), csv_names)
				if not physical:
					missing.append(str(source))
					continue
				parts.append(f"{quote_identifier(physical)} AS {quote_identifier(target)}")
				source_expr[target.upper()] = quote_identifier(physical)
			taken[target.upper()] = target
			ts_columns.append(target)
		if missing:
			raise CsvImportError("CSV is missing required data structure columns: " + ", ".join(missing))
	else:
		expected = [str(_attr(c, "name")) for c in (_attr(request, "expected_columns") or [])]
		ok, msg, _, _ = validate_csv_headers_exact_set(csv_names, expected)
		if not ok:
			raise CsvImportError(msg or "CSV headers do not match DSD columns")
		for name in csv_names:
			parts.append(quote_identifier(name))
			source_expr[name.upper()] = quote_identifier(name)
			taken[name.upper()] = name
			ts_columns.append(name)

	# Note text columns are loaded into staging (for note tuples) but not kept in timeseries.
	notes_resolved: Dict[str, Any] = {}
	for spec in note_columns:
		column = str(_attr(spec, "column")).strip()
		id_target = taken.get(column.upper())
		if not id_target:
			continue
		text_targets: List[str] = []
		for text_src in _attr(spec, "text_columns") or []:
			physical = resolve_column_name_case_insensitive(str(text_src), csv_names)
			if not physical:
				continue
			if physical.upper() not in taken:
				parts.append(quote_identifier(physical))
				taken[physical.upper()] = physical
			text_targets.append(taken[physical.upper()])
		notes_resolved[column] = {"id_column": id_target, "text_columns": text_targets}

	return parts, ts_columns, notes_resolved, source_expr


def _create_timeseries_table(
	conn,
	schema_name: str,
	qual_st: str,
	qual_new: str,
	ts_columns: List[str],
	time_spec: Any,
) -> None:
	"""Create the new timeseries table from (already filtered) staging."""
	col_list = ", ".join(f"s.{quote_identifier(c)}" for c in ts_columns)
	if time_spec is None:
		conn.execute(f"CREATE TABLE {qual_new} AS SELECT {col_list} FROM {qual_st} s")
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
		f"CREATE TABLE {qual_new} AS "
		f"SELECT {col_list}, {year_sql} AS {qy}, {freq_sql} AS {qfreq} FROM {qual_st} s"
	)


def _swap_in_new_timeseries(conn, schema_name: str) -> None:
	"""Replace timeseries with the new table in one transaction."""
	qual_ts = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
	qual_new = f"{quote_identifier(schema_name)}.{quote_identifier(NEW_TIMESERIES_TABLE_NAME)}"
	conn.execute("BEGIN TRANSACTION")
	try:
		if _table_exists(conn, schema_name, TIMESERIES_TABLE_NAME):
			conn.execute(f"DROP TABLE {qual_ts}")
		conn.execute(f"ALTER TABLE {qual_new} RENAME TO {quote_identifier(TIMESERIES_TABLE_NAME)}")
		conn.execute("COMMIT")
	except Exception:
		conn.execute("ROLLBACK")
		raise


def _note_tuples(conn, qual_st: str, notes_resolved: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
	"""
	Distinct (note ID cell, note text cells) per footnote column, in order of first appearance.
	Rows with neither an ID nor a text are left out.
	"""
	out: List[Dict[str, Any]] = []
	for column, spec in notes_resolved.items():
		cols = [spec["id_column"]] + list(spec["text_columns"])
		casts = [f"CAST({quote_identifier(c)} AS VARCHAR)" for c in cols]
		non_blank = " OR ".join(f"TRIM(COALESCE({c}, '')) <> ''" for c in casts)
		select_vals = ", ".join(f"{c} AS v{i}" for i, c in enumerate(casts))
		group_by = ", ".join(f"v{i}" for i in range(len(casts)))
		rows = conn.execute(
			f"SELECT {select_vals}, MIN(rowid) AS first_row, COUNT(*) AS n "
			f"FROM {qual_st} WHERE {non_blank} GROUP BY {group_by} ORDER BY first_row LIMIT ?",
			[limit + 1],
		).fetchall()
		truncated = len(rows) > limit
		items = []
		for r in rows[:limit]:
			items.append({
				"id": r[0] if r[0] is not None else "",
				"texts": [v if v is not None else "" for v in r[1:len(cols)]],
				"count": int(r[len(cols) + 1]),
			})
		out.append({
			"column": column,
			"text_columns": list(spec["text_columns"]),
			"rows": items,
			"truncated": truncated,
		})
	return out


def export_timeseries_to_file(conn, schema_name: str, output_path: str) -> Dict[str, Any]:
	"""COPY timeseries to output_path via a temp file + atomic replace. Returns row_count, bytes_written."""
	qual = f"{quote_identifier(schema_name)}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
	if not _table_exists(conn, schema_name, TIMESERIES_TABLE_NAME):
		raise ValueError("Timeseries table not found for this project")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	tmp_path = output_path + ".export_" + str(int(time.time() * 1000))
	path_norm = tmp_path.replace("\\", "/").replace("'", "''")
	try:
		row_count = int(conn.execute(f"SELECT COUNT(*) FROM {qual}").fetchone()[0])
		conn.execute(f"COPY (SELECT * FROM {qual}) TO '{path_norm}' (HEADER, DELIMITER ',')")
		if not os.path.isfile(tmp_path):
			raise ValueError("Export did not create output file")
		os.replace(tmp_path, output_path)
	finally:
		if os.path.isfile(tmp_path):
			try:
				os.unlink(tmp_path)
			except OSError:
				pass
	return {
		"output_csv_path": output_path,
		"row_count": row_count,
		"bytes_written": int(os.path.getsize(output_path)),
	}


def export_timeseries_to_file_db(db_path: str, project_id: str, output_path: str) -> Dict[str, Any]:
	"""Open the DB and export (for the standalone export-to-file job)."""
	schema_name = build_project_schema_name(project_id)
	with duckdb.connect(db_path) as conn:
		return export_timeseries_to_file(conn, schema_name, output_path)


def run_indicator_csv_import(db_path: str, request: Any, export_csv_path: Optional[str] = None) -> Dict[str, Any]:
	"""
	Replace project timeseries from rows of the CSV where indicator_column = indicator_value.

	Steps (one DuckDB connection):
	  1. load matching rows into staging, applying column renames/selection while reading;
	  2. check TIME_PERIOD against implied FREQ;
	  3. build timeseries__import_new and swap it in atomically;
	  4. collect distinct note tuples; 5. export the archive CSV (optional, failure is reported, not raised).
	Staging and the temporary table are always dropped; the live timeseries is never dropped on failure.
	"""
	project_id = str(_attr(request, "project_id"))
	csv_path = str(_attr(request, "csv_path"))
	delimiter = str(_attr(request, "delimiter", ",") or ",")
	indicator_value = str(_attr(request, "indicator_value"))
	time_spec = _attr(request, "time_spec")
	column_names = _attr(request, "column_names")
	note_limit = int(_attr(request, "note_tuple_limit") or DEFAULT_NOTE_TUPLE_LIMIT)

	assert_file_unchanged(csv_path, _attr(request, "expected_size"), _attr(request, "expected_mtime"))

	csv_names = resolve_csv_column_names(csv_path, delimiter, column_names)
	parts, ts_columns, notes_resolved, source_expr = _build_select(request, csv_names)

	indicator_col = resolve_column_name_case_insensitive(str(_attr(request, "indicator_column")), ts_columns)
	if not indicator_col:
		raise CsvImportError(f"Column not found in CSV: {_attr(request, 'indicator_column')}")

	schema_name = build_project_schema_name(project_id)
	qs = quote_identifier(schema_name)
	qual_st = f"{qs}.{quote_identifier(STAGING_TABLE_NAME)}"
	qual_new = f"{qs}.{quote_identifier(NEW_TIMESERIES_TABLE_NAME)}"
	src = csv_source_sql(csv_path, delimiter, column_names)

	db_dir = os.path.dirname(db_path)
	if db_dir:
		os.makedirs(db_dir, exist_ok=True)

	with duckdb.connect(db_path) as conn:
		conn.execute(f"CREATE SCHEMA IF NOT EXISTS {qs}")
		_drop_if_exists(conn, schema_name, STAGING_TABLE_NAME)
		_drop_if_exists(conn, schema_name, NEW_TIMESERIES_TABLE_NAME)
		try:
			# Filter while reading so only the indicator's rows are stored (same match as promote).
			ind_source = source_expr[indicator_col.upper()]
			conn.execute(
				f"CREATE TABLE {qual_st} AS SELECT {', '.join(parts)} FROM {src} "
				f"WHERE CAST({ind_source} AS VARCHAR) = ?",
				[indicator_value],
			)

			assert_staging_time_period_matches_implied_freq(
				conn, schema_name, STAGING_TABLE_NAME, qual_st, time_spec, indicator_col, indicator_value,
			)

			_create_timeseries_table(conn, schema_name, qual_st, qual_new, ts_columns, time_spec)
			_swap_in_new_timeseries(conn, schema_name)

			qual_ts = f"{qs}.{quote_identifier(TIMESERIES_TABLE_NAME)}"
			row_count = int(conn.execute(f"SELECT COUNT(*) FROM {qual_ts}").fetchone()[0])

			result: Dict[str, Any] = {
				"project_id": project_id,
				"row_count": row_count,
				"indicator_column_resolved": indicator_col,
				"indicator_value": indicator_value,
				"staging_dropped": True,
			}
			if notes_resolved:
				result["note_tuples"] = _note_tuples(conn, qual_st, notes_resolved, note_limit)

			if export_csv_path:
				try:
					result["export"] = export_timeseries_to_file(conn, schema_name, export_csv_path)
				except Exception as e:  # data is already committed; report, and do not leave a stale archive
					result["export_error"] = str(e)
					if os.path.isfile(export_csv_path):
						try:
							os.unlink(export_csv_path)
						except OSError:
							pass
			return result
		finally:
			for table in (STAGING_TABLE_NAME, NEW_TIMESERIES_TABLE_NAME):
				try:
					_drop_if_exists(conn, schema_name, table)
				except Exception:
					pass
