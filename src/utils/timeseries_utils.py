import re
from typing import List, Optional, Sequence, Tuple, Union


def validate_csv_field_names(field_names: List[str]) -> tuple[bool, str]:
	"""
	Validate CSV field names according to rules:
	- Must be unique
	- Cannot be numeric characters only
	- Allowed: alphanumeric with underscore and dashes
	- Max length: 50 characters
	
	Returns: (is_valid, error_message)
	"""
	if not field_names:
		return False, "No field names found in CSV header"
	
	seen = set()
	for name in field_names:
		# Check length
		if len(name) > 50:
			return False, f"Field name exceeds 50 characters: '{name}' ({len(name)} chars)"
		
		# Check for duplicates
		if name in seen:
			return False, f"Duplicate field name: '{name}'"
		seen.add(name)
		
		# Check if only numeric
		if name.isdigit():
			return False, f"Field name cannot be numeric only: '{name}'"
		
		# Check for allowed characters (alphanumeric, underscore, dash)
		if not re.match(r"^[a-zA-Z0-9_-]+$", name):
			return False, f"Field name contains invalid characters: '{name}'. Allowed: alphanumeric, underscore, and dashes"
	
	return True, ""


def normalize_dsd_column_name(name: str) -> str:
	"""Strip and uppercase for comparison with editor DSD / CSV headers."""
	return (name or "").strip().upper()


def csv_headers_normalized_lookup(headers: List[str]) -> dict[str, str]:
	"""
	Map UPPERCASE stripped header -> original header string (first occurrence wins).
	Use for case-insensitive DSD name resolution.
	"""
	out: dict[str, str] = {}
	for h in headers:
		if h is None:
			continue
		key = normalize_dsd_column_name(str(h))
		if key and key not in out:
			out[key] = h
	return out


def validate_dsd_columns_in_csv_headers(
	headers: List[str],
	dsd_column_names: List[str],
) -> tuple[bool, str]:
	"""
	Ensure every DSD column name appears in the CSV header row (case-insensitive),
	matching editor convention (DSD names are typically uppercase).

	Returns (True, "") or (False, error_message).
	"""
	if not dsd_column_names:
		return True, ""

	lookup = csv_headers_normalized_lookup(headers)
	missing: List[str] = []
	for raw in dsd_column_names:
		key = normalize_dsd_column_name(raw)
		if not key:
			return False, "DSD column name is empty"
		if key not in lookup:
			missing.append(raw.strip())

	if missing:
		preview = ", ".join(missing[:10])
		suffix = " …" if len(missing) > 10 else ""
		return False, f"DSD columns missing from CSV header: {preview}{suffix}"

	return True, ""


def sanitize_identifier(value: str, default: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", value or "").strip("_").lower()
    if not cleaned:
        cleaned = default
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier to avoid parsing issues."""
    return f'"{identifier}"'


def build_project_schema_name(project_id: str) -> str:
    """Build schema name from numeric project_id. Format: project_<id>"""
    return f"project_{project_id}"


# Canonical table names inside each project_* schema.
TIMESERIES_TABLE_NAME = "timeseries"
STAGING_TABLE_NAME = "staging"


def build_table_name(project_id: str) -> str:
    """Return the published indicator/timeseries table name (same for every project)."""
    return TIMESERIES_TABLE_NAME


def build_staging_table_name(project_id: str) -> str:
    """
    Return the staging table name for raw CSV loads before promote/filter.

    Qualified: project_{sid}.staging (e.g. project_877.staging).
    The project_id parameter is accepted for API symmetry; staging name is fixed.
    """
    return STAGING_TABLE_NAME


def resolve_column_name_case_insensitive(
	requested: str,
	column_names: Sequence[Union[str, Tuple, List]],
) -> Optional[str]:
	"""
	Match requested column name to a physical column name (case-insensitive).
	Pass column_names as strings or information_schema rows (column_name first).
	"""
	if not requested or not str(requested).strip():
		return None
	req = str(requested).strip().upper()
	names: List[str] = []
	for item in column_names:
		if item is None:
			continue
		if isinstance(item, (tuple, list)) and len(item) > 0:
			names.append(str(item[0]))
		else:
			names.append(str(item))
	for n in names:
		if n.strip().upper() == req:
			return n
	return None


def fetch_table_column_rows(conn, schema_name: str, table_name: str) -> list:
	"""information_schema rows for columns of schema.table."""
	return conn.execute(
		"""
		SELECT column_name, data_type, is_nullable
		FROM information_schema.columns
		WHERE table_schema = ? AND table_name = ?
		ORDER BY column_name
		""",
		[schema_name, table_name],
	).fetchall()
