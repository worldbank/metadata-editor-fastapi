"""
Derive DuckDB _ts_year and _ts_freq from TIME_PERIOD (+ optional FREQ) columns.

Contract aligns with PHP Indicator_dsd_model::build_duckdb_promote_time_spec / docs/dsd-duckdb.md.
Staging tables must NOT get these columns; only project_{sid}.timeseries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .timeseries_utils import escape_sql_string, quote_identifier, resolve_column_name_case_insensitive


def fetch_table_column_names_ordered(conn, schema_name: str, table_name: str) -> List[str]:
	"""Physical column order (ordinal_position when available)."""
	rows = conn.execute(
		"""
		SELECT column_name
		FROM information_schema.columns
		WHERE table_schema = ? AND table_name = ?
		ORDER BY ordinal_position
		""",
		[schema_name, table_name],
	).fetchall()
	return [str(r[0]) for r in rows if r and r[0] is not None]


def _norm_reserved(name: str) -> str:
	return (name or "").strip().upper()


def filter_user_columns(column_names: List[str]) -> List[str]:
	"""Drop platform-derived columns so we can rebuild them."""
	out: List[str] = []
	for c in column_names:
		if _norm_reserved(c) in ("_TS_YEAR", "_TS_FREQ"):
			continue
		out.append(c)
	return out


def resolve_freq_constant(
	time_period_format: Optional[str],
	implied_freq_code: Optional[str],
	default_freq_by_format: Optional[Dict[str, str]],
) -> Optional[str]:
	"""Single constant FREQ code when there is no freq_column."""
	if implied_freq_code and str(implied_freq_code).strip():
		return str(implied_freq_code).strip()
	if not default_freq_by_format or not time_period_format:
		return None
	return default_freq_by_format.get(time_period_format)


def sql_ts_year_expr(qualified_time_column: str) -> str:
	"""
	qualified_time_column e.g. s."TIME" — extract 4-digit year from start of string.
	"""
	inner = f"TRIM(CAST({qualified_time_column} AS VARCHAR))"
	return f"TRY_CAST(SUBSTRING({inner}, 1, 4) AS INTEGER)"


def sql_ts_freq_expr(
	qualified_freq_column: Optional[str],
	time_period_format: Optional[str],
	implied_freq_code: Optional[str],
	default_freq_by_format: Optional[Dict[str, str]],
) -> str:
	if qualified_freq_column:
		return f"TRIM(CAST({qualified_freq_column} AS VARCHAR))"
	code = resolve_freq_constant(time_period_format, implied_freq_code, default_freq_by_format)
	if not code:
		return "CAST(NULL AS VARCHAR)"
	return f"'{escape_sql_string(code)}'"


def build_derived_expressions(
	table_alias: str,
	resolved_time_col: str,
	resolved_freq_col: Optional[str],
	time_period_format: Optional[str],
	implied_freq_code: Optional[str],
	default_freq_by_format: Optional[Dict[str, str]],
) -> Tuple[str, str]:
	qt = f'{table_alias}.{quote_identifier(resolved_time_col)}'
	qf = f'{table_alias}.{quote_identifier(resolved_freq_col)}' if resolved_freq_col else None
	year_sql = sql_ts_year_expr(qt)
	freq_sql = sql_ts_freq_expr(qf, time_period_format, implied_freq_code, default_freq_by_format)
	return year_sql, freq_sql


def time_spec_dict_from_pydantic(spec: Any) -> Dict[str, Any]:
	"""Normalize PromoteTimeSpec-like model to dict."""
	if spec is None:
		return {}
	if hasattr(spec, "model_dump"):
		return spec.model_dump(exclude_none=True)
	return dict(spec)


def validate_and_resolve_time_spec(
	conn,
	schema_name: str,
	table_name: str,
	spec: Any,
) -> Tuple[Dict[str, Any], str, Optional[str]]:
	"""
	Load column names from schema.table, resolve time_column / freq_column (case-insensitive).
	Returns (spec_dict, resolved_time, resolved_freq_or_none).
	Raises ValueError on missing columns or missing time_column in spec.
	"""
	raw = time_spec_dict_from_pydantic(spec)
	time_col = (raw.get("time_column") or "").strip()
	if not time_col:
		raise ValueError("time_spec.time_column is required")

	col_rows = fetch_table_column_names_ordered(conn, schema_name, table_name)
	if not col_rows:
		raise ValueError(f"No columns found on {schema_name}.{table_name}")

	user_cols = filter_user_columns(col_rows)
	resolved_time = resolve_column_name_case_insensitive(time_col, user_cols)
	if not resolved_time:
		raise ValueError(f"time_spec.time_column not found in table: {time_col}")

	resolved_freq: Optional[str] = None
	fc = raw.get("freq_column")
	if fc and str(fc).strip():
		resolved_freq = resolve_column_name_case_insensitive(str(fc).strip(), user_cols)
		if not resolved_freq:
			raise ValueError(f"time_spec.freq_column not found in table: {fc}")

	return raw, resolved_time, resolved_freq
