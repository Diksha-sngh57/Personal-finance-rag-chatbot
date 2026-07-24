from __future__ import annotations
 
import logging
 
import pandas as pd
 
from src.config import GOLD_TRANSACTIONS_PATH
from src.rag.vector_store import get_indexed_ids, get_transactions_collection, upsert_transactions
 
logger = logging.getLogger(__name__)
 
REQUIRED_GOLD_COLUMNS = {"txn_key_hash", "txn_date", "description", "amount", "category", "source_type"}
 
 
def build_or_refresh_index() -> dict[str, int]:
    if not GOLD_TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(f"'{GOLD_TRANSACTIONS_PATH}' not found. Run the Gold pipeline first.")
 
    gold_df = pd.read_parquet(GOLD_TRANSACTIONS_PATH, engine="pyarrow")
    if gold_df.empty:
        raise ValueError("Gold table is empty — nothing to index.")
 
    missing = REQUIRED_GOLD_COLUMNS - set(gold_df.columns)
    if missing:
        raise ValueError(f"Gold table is missing expected column(s): {missing}")
 
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
 
    for _, row in gold_df.iterrows():
        date_str = pd.Timestamp(row["txn_date"]).strftime("%Y-%m-%d")
        direction = "received" if row["amount"] > 0 else "spent"
        amount_abs = abs(row["amount"])
        document = (
            f"On {date_str}, {direction} Rs {amount_abs:.2f} "
            f"({row['description']}), category: {row['category']}, source: {row['source_type']}."
        )
        ids.append(row["txn_key_hash"])
        documents.append(document)
        metadatas.append(
            {
                "txn_date": date_str,
                "category": row["category"],
                "amount": float(row["amount"]),
                "source_type": row["source_type"],
            }
        )
 
    upsert_transactions(ids, documents, metadatas)
 
    current_id_set = set(ids)
    already_indexed = get_indexed_ids()
    stale_ids = list(already_indexed - current_id_set)
    if stale_ids:
        get_transactions_collection().delete(ids=stale_ids)
        logger.info("Removed %d stale vector(s) no longer present in Gold.", len(stale_ids))
 
    logger.info("Indexed %d transaction(s) (existing ids updated in place, not duplicated).", len(ids))
 
    return {"upserted": len(ids), "removed_stale": len(stale_ids)}
