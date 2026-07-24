"""
Bronze output writer.
 
We deliberately do NOT call DataFrame.write.parquet() here. On a local
Windows box that produces a *directory* containing a part-00000-....parquet
file, a _SUCCESS marker, and .crc checksum files per part — clutter that
also depends on Hadoop's native winutils bits for the commit protocol
(see spark_session.py). Personal finance statements are a few hundred to
a few thousand rows; there is no distributed-storage benefit to Spark's
writer at this scale, only downside.
 
Instead, we let Spark do the actual transformation work (schema
enforcement, ingestion metadata, dedup — see pipeline.py), then
materialize the already-small result to Pandas and write exactly one
clean, predictably-named Parquet file via PyArrow. If this pipeline is
ever pointed at genuinely large data later, this is the one function that
would need to change back to a distributed writer — everything upstream
of it stays the same either way.
"""
from __future__ import annotations
 
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
 
from pyspark.sql import DataFrame
 
from src.config import BRONZE_DIR, BRONZE_MANIFEST_PATH, ensure_dir
 
logger = logging.getLogger(__name__)
 
 
def _assert_dir_really_exists(path: Path, retries: int = 3, delay_seconds: float = 0.5) -> None:
    """
    mkdir() not raising an exception is not proof a directory is actually
    usable. On some Windows setups — most notably the Microsoft Store
    build of Python (paths containing 'WindowsApps') combined with a
    OneDrive-synced project folder — file I/O can be silently redirected
    to a virtualized location: mkdir() "succeeds" against that virtual
    view, while pandas/pyarrow (and even the JVM, for Spark's own temp
    files) check the real path a moment later and find nothing.
 
    We retry briefly in case this is only a OneDrive sync timing race,
    then fail with a message that names the actual likely cause instead
    of surfacing pandas' generic "non-existent directory" OSError.
    """
    for attempt in range(retries):
        ensure_dir(path)
        if path.is_dir():
            return
        time.sleep(delay_seconds * (attempt + 1))
 
    is_store_python = "WindowsApps" in sys.executable
    raise RuntimeError(
        f"'{path}' could not be verified as a real directory after {retries} attempts, "
        f"even though creating it raised no error.\n"
        f"Python executable in use: {sys.executable}\n"
        + (
            "This interpreter is the Microsoft Store build of Python, which is the "
            "known cause of exactly this symptom when the project also lives inside "
            "a OneDrive-synced folder: file writes get silently redirected to a "
            "virtualized location instead of the real path. Fix: install Python from "
            "https://python.org (NOT the Microsoft Store), recreate this venv with "
            "that interpreter, and ideally move the project outside OneDrive."
            if is_store_python
            else "Check antivirus / 'Controlled Folder Access' settings for this path, "
            "and confirm no cloud-sync tool (OneDrive, Dropbox, etc.) is interfering "
            "with this folder."
        )
    )
 
 
def write_bronze_batch(df: DataFrame, batch_id: str) -> Path:
    """
    Writes a single Parquet file named bronze_<batch_id>.parquet and
    returns its path. Raises if the batch would be empty — an empty
    Bronze file is almost always a silent upstream extraction bug, not a
    legitimate outcome, so we fail loudly instead of writing a 0-row file.
    """
    pandas_df = df.toPandas()  # small-scale by design; safe materialization point
 
    if pandas_df.empty:
        raise ValueError("Refusing to write an empty Bronze batch — check the source extraction step.")
 
    _assert_dir_really_exists(BRONZE_DIR)
 
    output_path = BRONZE_DIR / f"bronze_{batch_id}.parquet"
    pandas_df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        # PyArrow defaults to nanosecond-precision Parquet timestamps for
        # pandas datetime64[ns] columns (our `ingested_at` column, from
        # Spark's current_timestamp()). Spark's own Parquet reader cannot
        # read that back — it raises "Illegal Parquet type: INT64
        # (TIMESTAMP(NANOS,false))". Forcing microsecond precision here
        # keeps this file readable by the very engine that produced it.
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
 
    logger.info("Wrote %d rows to %s", len(pandas_df), output_path)
    return output_path
 
 
def load_manifest() -> dict:
    _assert_dir_really_exists(BRONZE_DIR)
    if not BRONZE_MANIFEST_PATH.exists():
        return {"processed_files": {}}
    with open(BRONZE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def update_manifest(file_hash_to_name: dict[str, str], batch_id: str) -> None:
    """
    file_hash_to_name: {sha256_of_source_file: original_file_name}
 
    Recording the batch a source file was ingested into lets the pipeline
    skip re-ingesting a file it has already processed byte-for-byte, so
    reruns never create duplicate Bronze rows or duplicate output files.
    """
    _assert_dir_really_exists(BRONZE_DIR)
 
    manifest = load_manifest()
    now = datetime.now(timezone.utc).isoformat()
    for file_hash, file_name in file_hash_to_name.items():
        manifest["processed_files"][file_hash] = {
            "file_name": file_name,
            "batch_id": batch_id,
            "processed_at": now,
        }
    with open(BRONZE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)