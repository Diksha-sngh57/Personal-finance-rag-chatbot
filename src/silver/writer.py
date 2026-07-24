"""
Silver output writer.
 
Silver is fully recomputed from ALL Bronze batches on every run (see
pipeline.py), not incrementally merged. At personal-finance data volumes
this is simpler and provably correct — there's no upsert/merge logic to
get subtly wrong — and recomputing from scratch costs nothing noticeable
locally. If this pipeline is ever pointed at genuinely large, multi-year,
multi-user data, this is the point where true incremental merge logic
would replace the "reread everything" approach; everything upstream
(transform.py, the Bronze layer) stays the same either way.
 
Writes here go through a temp-file + atomic-rename pattern rather than
writing straight to the final path, so a crash or Ctrl+C mid-write can
never leave a half-written, corrupt Parquet file sitting at the path
other code expects to read cleanly.
"""
from __future__ import annotations
 
import logging
import os
from pathlib import Path
from typing import Optional
 
import pandas as pd
from pyspark.sql import DataFrame
 
from src.config import SILVER_QUARANTINE_PATH, SILVER_TRANSACTIONS_PATH, ensure_dir
 
logger = logging.getLogger(__name__)
 
 
def _atomic_write_parquet(pandas_df: pd.DataFrame, final_path: Path) -> None:
    """
    Writes to a temp file in the same directory, then renames over the
    final path. os.replace() is atomic on both Windows and POSIX when
    source and destination share a filesystem — which they always do
    here, since the temp file lives right next to its final destination.
    """
    ensure_dir(final_path.parent)
    tmp_path = final_path.with_suffix(".tmp")
    pandas_df.to_parquet(
        tmp_path,
        engine="pyarrow",
        index=False,
        # Same defensive fix as src/bronze/writer.py: force microsecond
        # (not PyArrow's nanosecond default) timestamp precision, since
        # Spark's Parquet reader can't read nanosecond timestamps back.
        # Silver doesn't currently write a timestamp column, but this
        # keeps that true by construction if one gets added later.
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    os.replace(tmp_path, final_path)
 
 
def write_silver_transactions(df: DataFrame) -> Path:
    pandas_df = df.toPandas()
 
    if pandas_df.empty:
        raise ValueError(
            "Refusing to write an empty Silver transactions table — "
            "check the Bronze data and the quarantine output for why nothing passed validation."
        )
 
    _atomic_write_parquet(pandas_df, SILVER_TRANSACTIONS_PATH)
    logger.info("Wrote %d clean transaction(s) to %s", len(pandas_df), SILVER_TRANSACTIONS_PATH)
    return SILVER_TRANSACTIONS_PATH
 
 
def write_silver_quarantine(df: DataFrame) -> Optional[Path]:
    pandas_df = df.toPandas()
 
    if pandas_df.empty:
        # Nothing quarantined this run. Remove any stale quarantine file
        # from a previous run so it doesn't look like there are still
        # unresolved bad rows sitting around when there aren't.
        if SILVER_QUARANTINE_PATH.exists():
            SILVER_QUARANTINE_PATH.unlink()
        logger.info("No quarantined rows this run.")
        return None
 
    _atomic_write_parquet(pandas_df, SILVER_QUARANTINE_PATH)
    logger.warning(
        "Wrote %d quarantined row(s) needing review to %s", len(pandas_df), SILVER_QUARANTINE_PATH
    )
    return SILVER_QUARANTINE_PATH
