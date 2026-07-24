"""
CLI entrypoint for Part 2: generate synthetic statements (optional) and
run the Bronze layer ETL against them.
 
Usage (run from the project root):
    python scripts/run_bronze_pipeline.py                    # ingest data/synthetic
    python scripts/run_bronze_pipeline.py --generate 3       # generate 3 synthetic
                                                                 statements, then ingest
    python scripts/run_bronze_pipeline.py --source uploads   # ingest data/uploads instead
"""
from __future__ import annotations
 
import argparse
import logging
import sys
from pathlib import Path
 
# Allow `python scripts/run_bronze_pipeline.py` to resolve `src.*` imports
# regardless of the caller's current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.bronze.pipeline import run_bronze_pipeline  # noqa: E402
from src.bronze.spark_session import stop_spark  # noqa: E402
from src.config import SYNTHETIC_DIR, UPLOADS_DIR  # noqa: E402
from src.synthetic_data.generate_statements import generate_batch  # noqa: E402
 
 
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
 
    parser = argparse.ArgumentParser(description="Run the Bronze layer ETL.")
    parser.add_argument(
        "--generate", type=int, default=0, metavar="N",
        help="Generate N synthetic PDF statements (+1 CSV) before ingesting.",
    )
    parser.add_argument(
        "--source", choices=["synthetic", "uploads"], default="synthetic",
        help="Which folder to ingest from (default: synthetic).",
    )
    args = parser.parse_args()
 
    if args.generate > 0:
        generate_batch(count=args.generate)
 
    source_dir = SYNTHETIC_DIR if args.source == "synthetic" else UPLOADS_DIR
 
    try:
        output_path = run_bronze_pipeline(source_dir=source_dir)
    finally:
        stop_spark()
 
    if output_path:
        print(f"\n✅ Bronze batch written: {output_path}")
    else:
        print("\n⚠️  No new Bronze batch written (nothing new to ingest — see log above for why).")
 
 
if __name__ == "__main__":
    main()
