"""
Centralized path & constant configuration for the Personal Finance
Intelligence pipeline. Every other module imports paths from here so we
never hardcode strings that could drift between the Streamlit app, the
synthetic data generator, and the Spark pipeline.
"""
from __future__ import annotations
 
from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
 
# Spark keeps ALL of its scratch state under one folder inside the project
# instead of scattering spark-warehouse/ and shuffle temp files wherever the
# process happens to be launched from. This whole directory is safe to
# delete at any time between runs.
SPARK_WORK_DIR = BASE_DIR / ".spark_work"
SPARK_WAREHOUSE_DIR = SPARK_WORK_DIR / "warehouse"
SPARK_LOCAL_TMP_DIR = SPARK_WORK_DIR / "tmp"
 
BRONZE_MANIFEST_PATH = BRONZE_DIR / "_manifest.json"
 
# Silver is fully recomputed from all Bronze batches on every run (see
# src/silver/pipeline.py for why), so — unlike Bronze — these are fixed
# filenames, not one-per-batch.
SILVER_TRANSACTIONS_PATH = SILVER_DIR / "silver_transactions.parquet"
SILVER_QUARANTINE_PATH = SILVER_DIR / "silver_quarantine.parquet"
 
# Gold is derived from Silver via LLM categorization (see src/gold/). The
# category cache is what makes reruns cheap — it's keyed by txn_key_hash,
# so a transaction already categorized in a prior run is never re-sent to
# the LLM, even though the Gold table itself is fully rebuilt each run.
GOLD_TRANSACTIONS_PATH = GOLD_DIR / "gold_transactions.parquet"
CATEGORY_CACHE_PATH = GOLD_DIR / "_category_cache.json"
 
# The single canonical category list — used by BOTH the Streamlit manual-
# entry dropdown (app.py) and the LLM categorizer (src/gold/categorizer.py).
# This lives here, not duplicated in each place, specifically so the LLM
# can never be asked to choose from a different set of categories than the
# ones a human sees in the UI.
CATEGORY_OPTIONS = (
    "Uncategorized",
    "Food & Dining",
    "Groceries",
    "Transport",
    "Shopping",
    "Bills & Utilities",
    "Rent",
    "Entertainment",
    "Health & Wellness",
    "Travel",
    "Investments",
    "Income",
    "Other",
)
 
 
# RAG layer — ChromaDB persists here. Deleting this folder is always safe;
# it gets fully rebuilt from data/gold/gold_transactions.parquet by
# src/rag/indexer.py (delivered next).
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
CHROMA_COLLECTION_NAME = "transactions"
 
# If this changes, VECTOR_STORE_DIR must be deleted and reindexed — a
# collection's embedding dimensionality is fixed at creation time, and a
# model swap will raise InvalidDimensionException on the next .add()/.query()
# otherwise. all-MiniLM-L6-v2 = 384 dims, CPU-fast, no API key required.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
 
 
def ensure_dir(path: Path) -> None:
    """
    Ensures `path` exists as a real directory.
 
    If a *file* (not a folder) already sits at this exact path — e.g. from
    `New-Item data\\bronze` in PowerShell without `-ItemType Directory`,
    which silently creates a file — mkdir(exist_ok=True) will keep raising
    FileExistsError forever, since exist_ok only suppresses that error when
    the existing path is already a directory. We surface that case
    immediately and specifically, rather than swallowing it and letting a
    much more confusing failure show up later, several calls deep, at the
    point something finally tries to write into the (non-)directory.
    """
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(
            f"Expected '{path}' to be a folder, but a FILE already exists at that "
            f"exact path. This project only ever creates this path as a directory, "
            f"so this is almost always a leftover from manually creating it wrong "
            f"(e.g. `New-Item {path.name}` in PowerShell without -ItemType Directory "
            f"creates a file, not a folder). Delete that file and rerun."
        )
 
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
 
 
for _dir in (
    UPLOADS_DIR,
    SYNTHETIC_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    VECTOR_STORE_DIR,
    SPARK_WAREHOUSE_DIR,
    SPARK_LOCAL_TMP_DIR,
):
    ensure_dir(_dir)
 