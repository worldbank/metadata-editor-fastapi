"""
Normalize an uploaded CSV in place to UTF-8 (no BOM), comma-separated.

Detects the source encoding (UTF-8, then cp1252, then latin-1) and delimiter
(comma, semicolon, tab, pipe). A file that is already UTF-8 without a BOM and
comma-separated is left untouched.

The returned dict is the `csv_normalization` contract used by the editor:
    converted               bool  - file was rewritten
    source_encoding         str   - "utf-8", "utf-8-sig", "cp1252" or "latin-1"
    source_delimiter        str   - the detected delimiter character
    source_delimiter_label  str   - "comma", "semicolon", "tab" or "pipe"
    rows                    int   - data rows written (only when converted)
    warnings                list  - human-readable notes for the import report
"""

import codecs
import csv
import os
import tempfile

DELIMITERS = [
    (",", "comma"),
    (";", "semicolon"),
    ("\t", "tab"),
    ("|", "pipe"),
]

_CHUNK = 1024 * 1024
_SAMPLE_LINES = 50


def _is_valid_encoding(path, encoding):
    """Stream the whole file through an incremental decoder."""
    decoder = codecs.getincrementaldecoder(encoding)()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                decoder.decode(chunk)
        decoder.decode(b"", final=True)
        return True
    except UnicodeDecodeError:
        return False


def detect_encoding(path):
    with open(path, "rb") as f:
        head = f.read(3)
    if head == codecs.BOM_UTF8:
        return "utf-8-sig"
    if _is_valid_encoding(path, "utf-8"):
        return "utf-8"
    if _is_valid_encoding(path, "cp1252"):
        return "cp1252"
    return "latin-1"  # decodes any byte sequence


def detect_delimiter(lines):
    """
    Pick the delimiter that splits the sample into the most columns with a
    consistent column count. Ties go to the order in DELIMITERS (comma first).
    """
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        return ",", "comma"

    best = None
    for delim, label in DELIMITERS:
        try:
            counts = [len(row) for row in csv.reader(lines, delimiter=delim)]
        except csv.Error:
            continue
        if not counts or counts[0] < 2:
            continue
        header = counts[0]
        # The sample can cut a quoted multi-line field; score on rows matching the header
        consistent = sum(1 for c in counts if c == header) / len(counts)
        if consistent < 0.8:
            continue
        score = (header, consistent)
        if best is None or score > best[0]:
            best = (score, delim, label)

    if best is None:
        return ",", "comma"  # single column or undecidable: treat as comma
    return best[1], best[2]


def _read_sample(path, encoding):
    lines = []
    with open(path, "r", encoding=encoding, newline="") as f:
        for line in f:
            lines.append(line)
            if len(lines) >= _SAMPLE_LINES:
                break
    return lines


def normalize_csv_file(path):
    """
    Normalize `path` in place. Raises ValueError if the file cannot be parsed.
    """
    if not os.path.isfile(path):
        raise ValueError("CSV file not found: " + str(path))

    encoding = detect_encoding(path)
    sample = _read_sample(path, encoding)
    if not sample:
        raise ValueError("CSV file is empty")

    delimiter, label = detect_delimiter(sample)

    info = {
        "converted": False,
        "source_encoding": encoding,
        "source_delimiter": delimiter,
        "source_delimiter_label": label,
        "warnings": [],
    }

    if encoding == "utf-8" and delimiter == ",":
        return info

    if encoding in ("cp1252", "latin-1"):
        info["warnings"].append(
            "The file was not UTF-8; it was read as {} and converted. "
            "Check that accented and special characters look correct.".format(encoding)
        )

    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".normalize-", suffix=".csv", dir=folder)
    rows = 0
    ragged = 0
    try:
        with open(path, "r", encoding=encoding, newline="") as src, \
                os.fdopen(fd, "w", encoding="utf-8", newline="") as dst:
            reader = csv.reader(src, delimiter=delimiter)
            writer = csv.writer(dst, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            header_len = None
            for row in reader:
                if header_len is None:
                    header_len = len(row)
                else:
                    rows += 1
                    if len(row) != header_len and row:
                        ragged += 1
                writer.writerow(row)
        try:
            os.chmod(tmp_path, os.stat(path).st_mode & 0o777)
        except OSError:
            pass
        os.replace(tmp_path, path)
    except csv.Error as e:
        _silent_remove(tmp_path)
        raise ValueError("Could not parse CSV file: " + str(e)) from e
    except Exception:
        _silent_remove(tmp_path)
        raise

    if ragged:
        info["warnings"].append(
            "{} row(s) have a different number of values than the header.".format(ragged)
        )

    info["converted"] = True
    info["rows"] = rows
    return info


def _silent_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
