"""
Gold output writer — same atomic single-file pattern as Silver
(src/silver/writer.py): write to a temp file, then os.replace() over the
final path, so a crash mid-write can never leave a corrupt file behind.
 
No Spark involved here — see pipeline.py for why the Gold layer doesn't
use Spark at all.
"""
from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
 
from src.config import GOLD_DIR, GOLD_TRANSACTIONS_PATH, ensure_dir
 
 
def write_gold_transactions(df: pd.DataFrame) -> Path:
    if df.empty:
        raise ValueError("Refusing to write an empty Gold transactions table.")
 
    ensure_dir(GOLD_DIR)
    tmp_path = GOLD_TRANSACTIONS_PATH.with_suffix(".tmp")
    df.to_parquet(
        tmp_path,
        engine="pyarrow",
        index=False,
        # Same nanosecond-timestamp fix as Bronze/Silver's writers — see
        # those files for the full explanation of why this matters.
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    os.replace(tmp_path, GOLD_TRANSACTIONS_PATH)
    return GOLD_TRANSACTIONS_PATH
