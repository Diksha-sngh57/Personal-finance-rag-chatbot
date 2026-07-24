"""
PDF statement extractor.
 
Bronze-layer philosophy: pull raw table rows out of the PDF exactly as
printed, and do NOT clean, cast, or validate values here. Type casting,
null-handling business rules, and de-duplication of "real" duplicate
transactions all belong to the Silver layer. This stage owns exactly one
job: turn PDF bytes into structured rows without losing or inventing data.
"""
from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Any, Optional
 
import pdfplumber
 
logger = logging.getLogger(__name__)
 
EXPECTED_HEADER_TOKENS = {"date", "description", "debit", "credit", "balance"}
 
 
def _looks_like_header_row(row: list[Optional[str]]) -> bool:
    normalized = {(cell or "").strip().lower() for cell in row}
    return len(normalized & EXPECTED_HEADER_TOKENS) >= 3
 
 
def _clean_cell(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.replace("\n", " ").strip()
    return value if value else None
 
 
def extract_rows_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Returns a list of raw row dicts:
        {
            "transaction_date_raw": str | None,
            "description_raw": str | None,
            "debit_raw": str | None,
            "credit_raw": str | None,
            "balance_raw": str | None,
            "source_page": int,
            "source_row_index": int,
        }
    Rows that are clearly headers or fully blank are skipped. Everything
    else is passed through untouched — no assumptions are made about
    what counts as "valid" at this layer.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
 
    rows: list[dict[str, Any]] = []
 
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) == 0:
            logger.warning("'%s' has zero pages.", pdf_path.name)
            return rows
 
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                logger.warning("No tables detected on page %d of '%s'.", page_number, pdf_path.name)
                continue
 
            for table in tables:
                for row_index, raw_row in enumerate(table):
                    if raw_row is None:
                        continue
                    if _looks_like_header_row(raw_row):
                        continue
 
                    cleaned = [_clean_cell(c) for c in raw_row]
                    if all(c is None for c in cleaned):
                        continue  # blank spacer rows are common in real bank PDFs
 
                    # Pad/truncate defensively — a malformed table row should
                    # degrade gracefully, not crash the whole ingestion run.
                    cleaned = (cleaned + [None] * 5)[:5]
 
                    rows.append(
                        {
                            "transaction_date_raw": cleaned[0],
                            "description_raw": cleaned[1],
                            "debit_raw": cleaned[2],
                            "credit_raw": cleaned[3],
                            "balance_raw": cleaned[4],
                            "source_page": page_number,
                            "source_row_index": row_index,
                        }
                    )
 
    if not rows:
        logger.warning("Extracted zero transaction rows from '%s'.", pdf_path.name)
 
    return rows
