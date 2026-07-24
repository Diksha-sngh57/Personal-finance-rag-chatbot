"""
Bronze layer ETL orchestrator.
 
Flow:
  1. Discover source files (PDF + CSV) in a given source directory.
  2. Skip any file already recorded in the manifest by content hash —
     idempotent reruns never duplicate data or churn out extra files.
  3. Extract raw rows per file with the appropriate extractor
     (PDF -> pdfplumber, CSV -> pandas), tagging each row with its
     source file name in plain Python, before Spark ever gets involved.
  4. Hand the combined row list to Spark for the transformations Bronze
     is actually responsible for: enforcing one uniform schema across
     both source types, stamping ingestion metadata, and computing a
     row-level hash for downstream dedup. This is the part of the stage
     that meaningfully benefits from Spark's DataFrame API — and the
     same code scales unchanged if statement volume grows later.
  5. Materialize to a single clean Parquet file (see writer.py) and
     update the manifest so step 2 can skip these files next run.
"""
from __future__ import annotations
 
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
 
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType
 
from src.bronze.csv_extractor import extract_rows_from_csv
from src.bronze.pdf_extractor import extract_rows_from_pdf
from src.bronze.spark_session import get_spark
from src.bronze.writer import load_manifest, update_manifest, write_bronze_batch
from src.config import SYNTHETIC_DIR
 
logger = logging.getLogger(__name__)
 
# All fields are strings on purpose — Bronze is schema-on-read. Date
# parsing and numeric casting happen in Silver, where we can apply real
# validation rules instead of guessing here.
RAW_ROW_SCHEMA = StructType(
    [
        StructField("transaction_date_raw", StringType(), True),
        StructField("description_raw", StringType(), True),
        StructField("debit_raw", StringType(), True),
        StructField("credit_raw", StringType(), True),
        StructField("balance_raw", StringType(), True),
        StructField("source_page", StringType(), True),
        StructField("source_row_index", StringType(), True),
        StructField("source_file", StringType(), True),
        StructField("source_type", StringType(), True),
    ]
)
 
 
def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
 
 
def _discover_source_files(source_dir: Path) -> list[Path]:
    return sorted(p for p in source_dir.glob("*") if p.is_file() and p.suffix.lower() in (".pdf", ".csv"))
 
 
def _extract_all_rows(files: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Returns (rows, file_hash_to_name) where file_hash_to_name only
    contains files actually included in this batch — already-processed
    files (per the manifest) are excluded here, not just filtered later.
    """
    manifest = load_manifest()
    already_processed = set(manifest.get("processed_files", {}).keys())
 
    all_rows: list[dict[str, Any]] = []
    file_hash_to_name: dict[str, str] = {}
 
    for file_path in files:
        file_hash = _sha256_of_file(file_path)
        if file_hash in already_processed:
            logger.info("Skipping '%s' — already ingested (content hash match).", file_path.name)
            continue
 
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                raw_rows = extract_rows_from_pdf(file_path)
                source_type = "PDF"
            else:
                raw_rows = extract_rows_from_csv(file_path)
                source_type = "CSV"
        except Exception:
            logger.exception("Failed to extract rows from '%s' — skipping this file.", file_path.name)
            continue
 
        if not raw_rows:
            logger.warning("'%s' produced zero rows — excluded from this batch.", file_path.name)
            continue
 
        for row in raw_rows:
            row["source_file"] = file_path.name
            row["source_type"] = source_type
            row["source_page"] = str(row["source_page"]) if row.get("source_page") is not None else None
            row["source_row_index"] = str(row["source_row_index"])
            all_rows.append(row)
 
        file_hash_to_name[file_hash] = file_path.name
 
    return all_rows, file_hash_to_name
 
 
def run_bronze_pipeline(source_dir: Path = SYNTHETIC_DIR) -> Optional[Path]:
    files = _discover_source_files(source_dir)
    if not files:
        logger.warning("No PDF/CSV files found in '%s'.", source_dir)
        return None
 
    rows, file_hash_to_name = _extract_all_rows(files)
    if not rows:
        logger.warning("Nothing new to ingest — all files already processed or produced zero rows.")
        return None
 
    spark = get_spark()
    df = spark.createDataFrame(rows, schema=RAW_ROW_SCHEMA)
 
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
 
    df = (
        df.withColumn("batch_id", F.lit(batch_id))
        .withColumn("ingested_at", F.current_timestamp())
        # Row hash gives Silver a stable dedup / idempotency key without
        # us having to guess at business-level uniqueness rules here.
        .withColumn(
            "row_hash",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("source_file"), F.lit("")),
                    F.coalesce(F.col("source_page"), F.lit("")),
                    F.coalesce(F.col("source_row_index"), F.lit("")),
                    F.coalesce(F.col("transaction_date_raw"), F.lit("")),
                    F.coalesce(F.col("description_raw"), F.lit("")),
                    F.coalesce(F.col("debit_raw"), F.lit("")),
                    F.coalesce(F.col("credit_raw"), F.lit("")),
                ),
                256,
            ),
        )
        .dropDuplicates(["row_hash"])
    )
 
    row_count = df.count()
    logger.info(
        "Bronze batch %s: %d raw row(s) extracted from %d file(s).",
        batch_id, row_count, len(file_hash_to_name),
    )
 
    output_path = write_bronze_batch(df, batch_id)
    update_manifest(file_hash_to_name, batch_id)
 
    return output_path