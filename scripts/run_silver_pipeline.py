"""
CLI entrypoint for Part 3: run the Silver layer transform against
whatever Bronze batches already exist on disk.
 
Usage (run from the project root):
    python scripts/run_silver_pipeline.py
"""
from __future__ import annotations
 
import logging
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.bronze.spark_session import stop_spark  # noqa: E402
from src.silver.pipeline import run_silver_pipeline  # noqa: E402
 
 
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
 
    try:
        transactions_path, quarantine_path = run_silver_pipeline()
    finally:
        stop_spark()
 
    print(f"\n✅ Silver transactions written: {transactions_path}")
    if quarantine_path:
        print(f"⚠️  Some rows need review — see: {quarantine_path}")
    else:
        print("✅ No quarantined rows this run.")
 
 
if __name__ == "__main__":
    main()
