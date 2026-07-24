"""
Spark session factory tuned for a single Windows/local machine.
 
Design goals:
  1. Never scatter spark-warehouse/ or shuffle temp files across the
     filesystem — everything Spark touches lives under .spark_work/,
     which is safe to delete between runs.
  2. Keep resource usage sane for a laptop-class CPU (i3, 2-4 cores) —
     the Spark defaults (200 shuffle partitions, unlimited UI, etc.) are
     tuned for clusters, not a single dev machine, and quietly create a
     lot of clutter and slowness if left as-is.
  3. Fail with a clear, actionable message if the Windows Hadoop native
     binaries (winutils.exe) are missing, instead of a cryptic Java stack
     trace three layers deep in a shuffle stage.
"""
from __future__ import annotations
 
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional
 
from pyspark.sql import SparkSession
 
from src.config import SPARK_LOCAL_TMP_DIR, SPARK_WAREHOUSE_DIR
 
logger = logging.getLogger(__name__)
 
 
def _warn_if_windows_hadoop_missing() -> None:
    """
    PySpark's local-mode execution still shells out to Hadoop's native
    file-permission code on Windows for shuffle spill files. If HADOOP_HOME
    isn't set (winutils.exe present under %HADOOP_HOME%\\bin), Spark can
    throw a NullPointerException / UnsatisfiedLinkError deep in a shuffle
    stage that has nothing obviously to do with Hadoop. We only *warn*
    here rather than hard-fail, because small local jobs sometimes get
    away without ever touching that code path — but if this pipeline dies
    with a strange error, this is the first thing to check.
    """
    if platform.system() != "Windows":
        return
 
    hadoop_home = os.environ.get("HADOOP_HOME")
    winutils_present = bool(hadoop_home) and (Path(hadoop_home) / "bin" / "winutils.exe").exists()
 
    if not winutils_present:
        logger.warning(
            "HADOOP_HOME / winutils.exe not detected. If this run fails with an "
            "obscure error during a shuffle stage, download winutils.exe matching "
            "your PySpark's bundled Hadoop version, place it at C:\\hadoop\\bin\\winutils.exe, "
            "and set HADOOP_HOME=C:\\hadoop before rerunning. This pipeline avoids "
            "Spark's own file writer specifically to sidestep this issue where "
            "possible, but internal shuffle stages can still need it."
        )
 
 
_spark_singleton: Optional[SparkSession] = None
 
 
def get_spark(app_name: str = "finance-bronze-etl", num_cores: int = 2) -> SparkSession:
    """
    Returns a memoized local SparkSession, reused across a single process
    run so repeated calls don't pay JVM startup cost or leak sessions.
 
    num_cores defaults to 2 — deliberately conservative for an i3-class
    CPU. Bump it (e.g. local[*]) once you've confirmed the pipeline runs
    cleanly and want more parallelism.
    """
    global _spark_singleton
    if _spark_singleton is not None:
        return _spark_singleton
 
    _warn_if_windows_hadoop_missing()
 
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
 
    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{num_cores}]")
        # Keep every scratch artifact inside the project instead of
        # scattering it across the working directory or system temp.
        .config("spark.sql.warehouse.dir", str(SPARK_WAREHOUSE_DIR))
        .config("spark.local.dir", str(SPARK_LOCAL_TMP_DIR))
        # A laptop CPU does not need 200 shuffle partitions (the Spark
        # default) for a few thousand rows of personal transactions —
        # left high, that alone produces hundreds of tiny output files
        # on every shuffle.
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        # No need for the web UI in a short-lived local ETL job, and
        # skipping it avoids "port 4040 already in use" noise when the
        # pipeline is run repeatedly during development.
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
    )
 
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")  # default INFO logging is extremely noisy
 
    _spark_singleton = spark
    return spark
 
 
def stop_spark() -> None:
    """Explicitly tear down the session and clear the singleton."""
    global _spark_singleton
    if _spark_singleton is not None:
        _spark_singleton.stop()
        _spark_singleton = None