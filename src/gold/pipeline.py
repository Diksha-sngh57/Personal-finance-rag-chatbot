"""
Gold layer orchestrator: LLM-based categorization on top of Silver +
manual entries.

Design note — this layer deliberately does NOT use Spark (see original
docstring reasoning, unchanged). As of Part 5, it also merges in manual
transactions logged directly in the UI, which never touch Bronze/Silver
at all (see src/gold/manual_overrides.py for why).

GOLD_OUTPUT_COLUMNS is a fixed, narrow contract — Gold no longer passes
through every raw Bronze/Silver column verbatim. This decouples the
Gold schema (and therefore the dashboard) from extraction implementation
details, and is what lets manual entries (which never had a "row_hash"
or "source_page" to begin with) merge in cleanly.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.config import CATEGORY_CACHE_PATH, GOLD_DIR, SILVER_TRANSACTIONS_PATH, ensure_dir
from src.gold.categorizer import BATCH_SIZE, TransactionToCategorize, categorize_batch
from src.gold.manual_overrides import get_manual_category_overrides, manual_transactions_as_dataframe
from src.gold.writer import write_gold_transactions

logger = logging.getLogger(__name__)

BATCH_PAUSE_SECONDS = 13  # unchanged — see original docstring for the Groq TPM math

GOLD_OUTPUT_COLUMNS = ["txn_key_hash", "txn_date", "description", "amount", "source_type", "source_file"]


def _load_cache() -> dict[str, str]:
    if not CATEGORY_CACHE_PATH.exists():
        return {}
    with open(CATEGORY_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache: dict[str, str]) -> None:
    ensure_dir(GOLD_DIR)
    tmp_path = CATEGORY_CACHE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp_path, CATEGORY_CACHE_PATH)


def run_gold_pipeline() -> Path:
    if not SILVER_TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"'{SILVER_TRANSACTIONS_PATH}' not found. Run the Bronze + Silver pipeline first "
            f"(upload a statement, then run the full pipeline)."
        )

    silver_df = pd.read_parquet(SILVER_TRANSACTIONS_PATH, engine="pyarrow")
    logger.info("Read %d transaction(s) from Silver.", len(silver_df))
    silver_df = silver_df[GOLD_OUTPUT_COLUMNS].copy()

    manual_df = manual_transactions_as_dataframe()
    if not manual_df.empty:
        logger.info("Including %d manual transaction(s) in this Gold run.", len(manual_df))

    combined_df = pd.concat([silver_df, manual_df], ignore_index=True)

    # Spark's toPandas() and pandas' native `date` objects don't reliably
    # land on the same dtype — normalize once here so the Parquet writer
    # and any downstream date filtering see one consistent dtype.
    combined_df["txn_date"] = pd.to_datetime(combined_df["txn_date"])

    before = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["txn_key_hash"], keep="last").reset_index(drop=True)
    if len(combined_df) != before:
        logger.warning(
            "Dropped %d duplicate txn_key_hash row(s) when combining Silver + manual sources.",
            before - len(combined_df),
        )

    cache = _load_cache()

    overrides = get_manual_category_overrides()
    if overrides:
        cache.update(overrides)
        # Persist immediately. Do NOT wait for the batch loop below to do
        # it — if overrides cover every currently-uncached row, that loop
        # never executes, and the override would otherwise vanish on the
        # next run if the cache file were ever regenerated from scratch.
        _save_cache(cache)
        logger.info("%d manual category override(s) applied — these are never sent to the LLM.", len(overrides))

    uncached_mask = ~combined_df["txn_key_hash"].isin(cache.keys())
    uncached_df = combined_df[uncached_mask]

    logger.info(
        "%d transaction(s) already categorized (cache hit or manual override), %d new to send to the LLM.",
        len(combined_df) - len(uncached_df),
        len(uncached_df),
    )

    if not uncached_df.empty:
        rows = list(uncached_df[["txn_key_hash", "description", "amount"]].itertuples(index=False, name=None))
        total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(total_batches):
            batch_rows = rows[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
            batch_items = [
                TransactionToCategorize(
                    id=i,
                    description=description or "",
                    amount=float(amount) if amount is not None else 0.0,
                )
                for i, (_txn_hash, description, amount) in enumerate(batch_rows)
            ]
            hash_by_local_id = {i: txn_hash for i, (txn_hash, _desc, _amt) in enumerate(batch_rows)}

            logger.info("Categorizing batch %d/%d (%d transactions)...", batch_num + 1, total_batches, len(batch_items))
            result_map = categorize_batch(batch_items)

            for local_id, category in result_map.items():
                cache[hash_by_local_id[local_id]] = category

            _save_cache(cache)

            if batch_num < total_batches - 1:
                time.sleep(BATCH_PAUSE_SECONDS)

    combined_df["category"] = combined_df["txn_key_hash"].map(cache)

    uncategorized_count = int(combined_df["category"].isna().sum())
    if uncategorized_count > 0:
        logger.warning(
            "%d row(s) still have no category after this run — falling back to 'Uncategorized' "
            "rather than failing the whole write.",
            uncategorized_count,
        )
        combined_df["category"] = combined_df["category"].fillna("Uncategorized")

    return write_gold_transactions(combined_df)