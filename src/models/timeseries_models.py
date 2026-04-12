from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional


class DsdColumnRef(BaseModel):
	"""Reference to one row of indicator_dsd (column identity in CSV)."""

	name: str = Field(
		...,
		min_length=1,
		max_length=255,
		description="DSD column name (sid); must match a CSV header case-insensitively",
	)


class IndicatorTimeseriesImportRequest(BaseModel):
	"""
	Load CSV into project_{sid}.staging (wizard: distinct IDs → pick indicator → promote).

	Caller (PHP) saves the CSV to a path readable by this service, then POSTs here.
	Optional dsd_columns enforces that every listed name exists in the CSV header
	before the job is queued (aligns with MySQL indicator_dsd).

	Use POST /timeseries/indicators/queue only for legacy direct loads into timeseries.
	"""

	project_id: str = Field(..., description="Same as editor sid (numeric string)")
	csv_path: str = Field(..., description="Absolute path to CSV on this host")
	delimiter: str = Field(",", min_length=1, max_length=1, description="CSV delimiter")
	dsd_columns: Optional[List[DsdColumnRef]] = Field(
		None,
		description="If provided, every name must exist in the CSV header (case-insensitive)",
	)


class PromoteTimeSpec(BaseModel):
	"""
	How to derive _ts_year / _ts_freq on project_{sid}.timeseries (not on staging).
	Mirrors PHP Indicator_dsd_model::build_duckdb_promote_time_spec.
	"""

	time_column: str = Field(
		...,
		min_length=1,
		max_length=255,
		description="Physical column holding TIME_PERIOD (case-insensitive match on promote source)",
	)
	time_period_format: Optional[str] = Field(
		None,
		description="One of YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SSZ",
	)
	default_freq_by_format: Optional[Dict[str, str]] = Field(
		None,
		description="When freq_column absent: map time_period_format -> default FREQ code",
	)
	freq_column: Optional[str] = Field(
		None,
		description="Physical FREQ column when DSD has periodicity (case-insensitive)",
	)
	implied_freq_code: Optional[str] = Field(
		None,
		description="Constant FREQ when no freq_column (metadata.import_freq_code)",
	)
	global_time_codelist_id: Optional[int] = Field(None, description="Reserved for future validation")
	global_freq_codelist_id: Optional[int] = Field(None, description="Reserved for future validation")


class IndicatorPromoteRequest(BaseModel):
	"""Copy rows from staging into timeseries where indicator_column = indicator_value (replace timeseries)."""

	project_id: str = Field(..., description="Same as editor sid (numeric string)")
	indicator_column: str = Field(
		...,
		min_length=1,
		max_length=255,
		description="Physical CSV column name in staging (case-insensitive match)",
	)
	indicator_value: str = Field(
		...,
		description="Value to keep; other indicator ids are excluded from timeseries",
	)
	time_spec: Optional[PromoteTimeSpec] = Field(
		None,
		description="When set with time_column, promoted timeseries includes _ts_year and _ts_freq",
	)


class RecomputeTimeDerivedRequest(BaseModel):
	"""Recompute _ts_year / _ts_freq on existing project_{sid}.timeseries from current time_spec."""

	project_id: str = Field(..., description="Same as editor sid (numeric string)")
	time_spec: PromoteTimeSpec = Field(..., description="Full spec; time_column must exist on timeseries")


class StagingDistinctValueCount(BaseModel):
	"""One distinct indicator code and its row count in staging."""

	value: str = Field(..., description="Cast to string; matches promote filter")
	count: int = Field(..., ge=0, description="Rows in staging with this value")


class StagingDistinctResponse(BaseModel):
	project_id: str
	column_resolved: str
	values: List[str] = Field(
		...,
		description="Legacy flat list; same order as items[].value",
	)

	items: List[StagingDistinctValueCount] = Field(
		default_factory=list,
		description="Distinct values with counts (sorted by count desc, then value)",
	)
	truncated: bool
	staging_row_count: int


class StagingDescribeColumn(BaseModel):
	"""Physical column name in project_{sid}.staging."""

	name: str = Field(..., description="Column name as stored in DuckDB")


class StagingDescribeResponse(BaseModel):
	exists: bool = Field(..., description="True if project_{sid}.staging is present")
	row_count: int = Field(0, description="Row count when exists")
	columns: List[StagingDescribeColumn] = Field(
		default_factory=list,
		description="Column names (order matches table)",
	)


class StagingDropResponse(BaseModel):
	"""Result of removing project_{sid}.staging after a successful import."""

	project_id: str = Field(..., description="Editor sid")
	dropped: bool = Field(..., description="True if the staging table existed and was dropped")


class TimeseriesDropResponse(BaseModel):
	"""Result of dropping project_{sid}.timeseries (delete all published data)."""

	project_id: str = Field(..., description="Editor sid")
	dropped: bool = Field(..., description="True if the timeseries table existed and was dropped")
	row_count: int = Field(0, description="Number of rows that were in the table before dropping")


class StagingSampleResponse(BaseModel):
	"""First N rows from project_{sid}.staging for UI preview."""

	project_id: str
	columns: List[str] = Field(default_factory=list, description="Column names in result order")
	rows: List[Dict[str, str]] = Field(
		default_factory=list,
		description="Row dicts keyed by physical column name; values as strings",
	)
	row_count_returned: int = Field(0, description="Number of rows in rows")


class TimeseriesColumnMeta(BaseModel):
	"""Published timeseries table column from information_schema (physical order)."""

	name: str = Field(..., description="Column name as in DuckDB")
	data_type: str = Field(..., description="SQL data type")
	is_nullable: str = Field("YES", description="YES or NO")


class TimeseriesPageResponse(BaseModel):
	"""Paginated rows from project_{sid}.timeseries (DuckDB) for data explorer."""

	project_id: str
	columns: List[TimeseriesColumnMeta] = Field(default_factory=list)
	rows: List[Dict[str, str]] = Field(default_factory=list)
	total_row_count: int = Field(
		0,
		description="Row count for the current query (full table when no filters; filtered count when filters are applied)",
	)
	offset: int = 0
	limit: int = 0
	row_count_returned: int = 0


class TimeseriesDistinctPair(BaseModel):
	"""One distinct dimension code with a representative label from published timeseries."""

	code: str = Field(..., description="Trimmed string; empty codes omitted")
	label: str = Field(..., description="Label column or same as code when label omitted")


class TimeseriesDistinctPairsResponse(BaseModel):
	"""Distinct codes from project_{sid}.timeseries (DuckDB), one row per code."""

	project_id: str
	code_column_resolved: str = Field(..., description="Physical column used as code")
	label_column_resolved: Optional[str] = Field(
		None,
		description="Physical column used as label when provided",
	)
	items: List[TimeseriesDistinctPair] = Field(default_factory=list)
	truncated: bool = Field(
		False,
		description="True when more distinct codes exist than returned (limit applied)",
	)


class TimeseriesColumnFreqItem(BaseModel):
	"""One value and its row count (non-missing cells only)."""

	value: str = Field(..., description="Trimmed string as stored for profiling")
	count: int = Field(..., ge=0, description="Rows with this value among present cells")


class TimeseriesColumnStat(BaseModel):
	"""Summary statistics for one timeseries column (DuckDB)."""

	field: str = Field(..., description="Resolved physical column name")
	row_count: int = Field(..., ge=0)
	non_null_count: int = Field(..., ge=0, description="Present: not NULL and trim(cast AS VARCHAR) != ''")
	null_count: int = Field(..., ge=0, description="Missing: NULL or blank/whitespace-only string")
	distinct_count: int = Field(..., ge=0, description="Distinct trimmed values among present cells")
	freq_max: int = Field(100, description="Maximum number of frequency rows returned")
	freq_truncated: bool = Field(False, description="True if more than freq_max distinct values exist")
	freq: List[TimeseriesColumnFreqItem] = Field(
		default_factory=list,
		description="Top values by count, then value asc; at most freq_max entries",
	)


class TimeseriesColumnStatsResponse(BaseModel):
	"""Batch column profiles for editor sum_stats (MySQL indicator_dsd.sum_stats)."""

	project_id: str
	source: str = Field("timeseries", description="Table layer used")
	computed_at: str = Field(..., description="UTC ISO-8601 timestamp")
	columns: List[TimeseriesColumnStat] = Field(default_factory=list)


class TimeseriesBaseModel(BaseModel):
    """Base class for timeseries request/response models."""

    pass


class TimeseriesImportRequest(TimeseriesBaseModel):
    project_id: str = Field(..., description="Project identifier")
    csv_path: str = Field(..., description="Path to the source CSV file")
    delimiter: str = Field(",", description="CSV delimiter")
    replace: bool = Field(False, description="Whether to replace an existing table")


class TimeseriesJobResponse(TimeseriesBaseModel):
    message: str = Field(..., description="Status message")
    job_id: str = Field(..., description="Unique job identifier")
    operation_type: str = Field(..., description="Type of operation")
    project_id: str = Field(..., description="Project identifier")


class TimeseriesTableInfo(TimeseriesBaseModel):
    project_id: str = Field(..., description="Project identifier")
    qualified_table: str = Field(..., description="Schema-qualified table name")


class TimeseriesTablesResponse(TimeseriesBaseModel):
    project_id: str = Field(..., description="Project identifier")
    tables: List[TimeseriesTableInfo] = Field(..., description="Tables for the project")


class TimeseriesAllTablesResponse(TimeseriesBaseModel):
    total_tables: int = Field(..., description="Total number of tables")
    tables: List[TimeseriesTableInfo] = Field(..., description="All timeseries tables across all projects")


class TimeseriesColumnInfo(TimeseriesBaseModel):
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column type")
    not_null: bool = Field(..., description="Whether the column is NOT NULL")
    default: str = Field(None, description="Default value")


class TimeseriesDescribeResponse(TimeseriesBaseModel):
    project_id: str = Field(..., description="Project identifier")
    schema: str = Field(..., description="Schema name")
    table: str = Field(..., description="Table name")
    qualified_table: str = Field(..., description="Schema-qualified table name")
    row_count: int = Field(..., description="Number of rows")
    column_count: int = Field(..., description="Number of columns")
    columns: List[TimeseriesColumnInfo] = Field(..., description="Column metadata")


class TimeseriesDeleteRequest(TimeseriesBaseModel):
    project_id: Optional[str] = Field(None, description="Project identifier (numeric)")
    table_name: Optional[str] = Field(None, description="Full table name (schema.table)")


class TimeseriesDeleteResponse(TimeseriesBaseModel):
    message: str = Field(..., description="Status message")
    schema: str = Field(..., description="Schema name")
    table: str = Field(..., description="Table name")
    qualified_table: str = Field(..., description="Schema-qualified table name")
    rows_deleted: int = Field(..., description="Number of rows that were in the table")


class IndicatorChartAggregateRequest(BaseModel):
	"""
	SDMX-style chart rows: one observation per (time × slice dimensions).
	PHP Indicator_dsd_model::build_chart_aggregate_spec — unknown fields ignored.
	"""

	model_config = ConfigDict(extra="ignore")

	project_id: str = Field(..., description="Editor sid (numeric string)")
	time_column: str = Field(..., min_length=1, description="Physical TIME_PERIOD column")
	value_column: str = Field(..., min_length=1, description="Physical observation measure column")
	slice_columns: List[str] = Field(
		default_factory=list,
		description="Physical columns for series identity (excludes time and value)",
	)
	filters: Dict[str, List[str]] = Field(
		default_factory=dict,
		description="Physical column -> selected codes (trimmed string IN filter)",
	)
	time_period_start: Optional[str] = None
	time_period_end: Optional[str] = None
	use_ts_year_for_time_filter: Optional[bool] = None


class IndicatorObservationKeyValidateRequest(BaseModel):
	"""
	Full published timeseries: SDMX observation-key uniqueness via DuckDB aggregates (no row streaming).
	Must use the same trim(cast AS VARCHAR)) key parts as chart-aggregate.
	"""

	model_config = ConfigDict(extra="ignore")

	project_id: str = Field(..., description="Editor sid (numeric string)")
	time_column: str = Field(..., min_length=1, description="Physical time period column")
	value_column: str = Field(..., min_length=1, description="Physical observation value column")
	slice_columns: List[str] = Field(
		default_factory=list,
		description="Physical slice columns (geography, dimensions, periodicity); excludes time and value",
	)


class IndicatorObservationKeyValidateResponse(BaseModel):
	project_id: str
	time_column: str
	value_column: str
	slice_columns: List[str] = Field(default_factory=list)
	table_total_row_count: int = Field(..., ge=0, description="COUNT(*) on full timeseries table")
	rows_with_observation_value: int = Field(
		...,
		ge=0,
		description="Rows with non-null non-empty trimmed observation value (same WHERE as chart-aggregate)",
	)
	unique_observation_key_count: int = Field(
		...,
		ge=0,
		description="Distinct (time × slice) keys among those rows",
	)
	duplicate_key_group_count: int = Field(
		...,
		ge=0,
		description="Number of keys with more than one row (HAVING COUNT(*) > 1)",
	)
	duplicate_row_count: int = Field(
		...,
		ge=0,
		description="rows_with_observation_value − unique_observation_key_count",
	)
	source: str = Field(default="duckdb", description="Aggregate computed in DuckDB")


class FacetValueCountItem(BaseModel):
	"""One trimmed string value and row count in published timeseries (chart facet histogram)."""

	value: str = Field(..., description="trim(cast AS VARCHAR))")
	count: int = Field(..., ge=0)


class IndicatorFacetValueCountsRequest(BaseModel):
	"""Dataset-wide GROUP BY counts per column (no filter cross-effects)."""

	project_id: str = Field(..., description="Editor sid (numeric string)")
	columns: List[str] = Field(
		...,
		min_length=1,
		description="Physical column names; resolved case-insensitively on timeseries table",
	)


class IndicatorFacetValueCountsResponse(BaseModel):
	project_id: str
	column_counts: Dict[str, List[FacetValueCountItem]] = Field(
		default_factory=dict,
		description="Resolved physical column name -> value/count rows",
	)
	columns_truncated: Dict[str, bool] = Field(
		default_factory=dict,
		description="True when more distinct values exist than returned for that column",
	)
