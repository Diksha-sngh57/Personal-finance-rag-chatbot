"""
Silver layer ETL orchestrator.
 
Flow:
  1. Read every Bronze batch parquet file at once
     (spark.read.parquet(*paths) — this is a case where Spark's native
     multi-file read genuinely earns its keep over hand-rolling a loop
     and union in Python).
  2. Apply the Silver transform (type casting, cleaning, business key,
     quarantine flagging) — see transform.py.
  3. Split into valid vs. quarantined rows, and de-duplicate ONLY the
     valid rows on the business key. Quarantined rows are deliberately
     never deduplicated: two bad rows can look identical purely because
     both are missing the same field (e.g. two rows with no date at all
     hash the same), and collapsing those would hide review-worthy
     problems instead of surfacing them.
  4. Write both outputs as single, clean Parquet files (see writer.py).
"""
from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Optional
 
from pyspark.sql import functions as F
 
from src.bronze.spark_session import get_spark
from src.config import BRONZE_DIR
from src.silver.transform import apply_silver_transform
from src.silver.writer import write_silver_quarantine, write_silver_transactions
 
logger = logging.getLogger(__name__)
 
 
def _discover_bronze_files() -> list[Path]:
    return sorted(BRONZE_DIR.glob("bronze_*.parquet"))
 
 
def run_silver_pipeline() -> tuple[Path, Optional[Path]]:
    files = _discover_bronze_files()
    if not files:
        raise FileNotFoundError(
            f"No Bronze parquet files found in '{BRONZE_DIR}'. Run "
            f"scripts/run_bronze_pipeline.py at least once before running Silver."
        )
 
    spark = get_spark()
    bronze_df = spark.read.parquet(*[str(f) for f in files])
    bronze_row_count = bronze_df.count()
    logger.info("Read %d raw row(s) from %d Bronze file(s).", bronze_row_count, len(files))
 
    transformed_df = apply_silver_transform(bronze_df)
 
    valid_df = transformed_df.filter(F.col("quarantine_reason").isNull()).dropDuplicates(["txn_key_hash"])
    quarantine_df = transformed_df.filter(F.col("quarantine_reason").isNotNull())
 
    valid_count = valid_df.count()
    quarantine_count = quarantine_df.count()
    logger.info(
        "Silver transform: %d clean row(s) after dedup, %d row(s) quarantined.",
        valid_count,
        quarantine_count,
    )
 
    if valid_count == 0:
        raise ValueError(
            "Every single Bronze row failed Silver validation. This almost certainly "
            "means the transform logic doesn't match your actual data shape (e.g. an "
            "unexpected date format), not that all your transactions are genuinely "
            "invalid — check data/silver/silver_quarantine.parquet for the specific "
            "reasons before assuming this result is correct."
        )
 
    transactions_path = write_silver_transactions(valid_df)
    quarantine_path = write_silver_quarantine(quarantine_df) if quarantine_count > 0 else None
 
    return transactions_path, quarantine_path
 
