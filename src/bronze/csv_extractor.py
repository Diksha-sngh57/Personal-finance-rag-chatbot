"""
CSV statement extractor — mirrors pdf_extractor.py's output contract so
both sources feed the same Bronze schema downstream.
"""
from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Any
 
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# Bank/UPI-app CSV exports vary wildly in header naming. We map every
# alias we've seen (or are likely to see) onto our canonical raw columns
# instead of assuming one fixed header layout.
COLUMN_ALIASES = {
    "transaction_date_raw": {"date", "txn date", "transaction date", "value date"},
    "description_raw": {"description", "narration", "particulars", "remarks", "details"},
    "debit_raw": {"debit", "withdrawal", "debit amount", "dr"},
    "credit_raw": {"credit", "deposit", "credit amount", "cr"},
    "balance_raw": {"balance", "closing balance", "available balance"},
}
 
 
def _map_columns(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for original in columns:
        key = original.strip().lower()
        for canonical, aliases in COLUMN_ALIASES.items():
            if key in aliases and canonical not in mapping.values():
                mapping[original] = canonical
                break
    return mapping
 
 
def extract_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """
    Returns raw row dicts with the same 5 canonical fields as
    extract_rows_from_pdf, plus source_row_index (source_page is always
    None for CSVs). Unmapped/unexpected columns are dropped for Bronze
    purposes — nothing is inferred — but a warning is logged so the user
    knows a column went unrecognized.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
 
    df: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding, dtype=str, keep_default_na=False)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
 
    if df is None:
        raise ValueError(f"Could not decode '{csv_path.name}' with utf-8/latin-1/cp1252.") from last_error
 
    if df.empty:
        logger.warning("'%s' has a header but zero data rows.", csv_path.name)
        return []
 
    column_mapping = _map_columns(list(df.columns))
    unmapped = [c for c in df.columns if c not in column_mapping]
    if unmapped:
        logger.warning("Unrecognized columns in '%s' ignored for Bronze: %s", csv_path.name, unmapped)
 
    missing_required = {"transaction_date_raw", "description_raw"} - set(column_mapping.values())
    if missing_required:
        raise ValueError(
            f"'{csv_path.name}' is missing required column(s) for {missing_required}. "
            f"Detected headers: {list(df.columns)}"
        )
 
    df = df.rename(columns=column_mapping)
    for canonical in COLUMN_ALIASES:
        if canonical not in df.columns:
            df[canonical] = None  # keep schema uniform even when the source lacked this column
 
    rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(df.to_dict(orient="records")):
        rows.append(
            {
                "transaction_date_raw": record.get("transaction_date_raw") or None,
                "description_raw": record.get("description_raw") or None,
                "debit_raw": record.get("debit_raw") or None,
                "credit_raw": record.get("credit_raw") or None,
                "balance_raw": record.get("balance_raw") or None,
                "source_page": None,
                "source_row_index": row_index,
            }
        )
    return rows
