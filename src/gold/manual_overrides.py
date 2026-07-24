"""
Manual transaction store + override resolution.

Manual entries are logged directly in the Streamlit UI (app.py) and never
pass through the Bronze/Silver Spark pipeline — there is nothing to
extract or parse; the user has already given us structured fields.

This module owns:
  1. Crash-safe persistence to data/manual/manual_transactions.json (same
     temp-file + os.replace() pattern used by every other writer in this
     project).
  2. A txn_key_hash for each manual entry that CANNOT collide with
     anything — not another manual entry, not a statement-derived
     transaction. Reusing Silver's business-key formula (date + amount +
     description + balance) is NOT safe here: manual entries have no
     running balance, so two identical-looking entries logged the same
     day (e.g. two ₹150 coffees) would hash identically and silently
     collapse. We sidestep this entirely by hashing each entry's own
     uuid4 instead of its business fields — uniqueness is true by
     construction, not by the coincidence of the input fields never
     repeating.
  3. Exposing manual entries as a small DataFrame shaped to match Gold's
     fixed output contract (see GOLD_OUTPUT_COLUMNS in gold/pipeline.py),
     so they can be concatenated onto Silver's output.
  4. Exposing {txn_key_hash: category} overrides — ONLY for entries where
     the user picked something other than the "Uncategorized" sentinel,
     i.e. entries where they actually made a choice. gold/pipeline.py
     checks this map before touching the LLM cache, so an explicit manual
     category is never sent to Groq and never gets second-guessed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, ensure_dir

logger = logging.getLogger(__name__)

MANUAL_DIR = DATA_DIR / "manual"
MANUAL_TRANSACTIONS_PATH = MANUAL_DIR / "manual_transactions.json"

# Must match CATEGORY_OPTIONS[0] in src/config.py exactly — this is the
# sentinel meaning "the user did not choose a category, let the LLM decide."
UNSET_CATEGORY_SENTINEL = "Uncategorized"


@dataclass
class ManualTransaction:
    entry_id: str
    amount: float  # signed: negative = money out, positive = money in
    merchant: str
    txn_date: date
    category: str
    logged_at: datetime = field(default_factory=datetime.now)

    @property
    def txn_key_hash(self) -> str:
        """
        Derived from entry_id alone — a uuid4 assigned once at creation
        and never recomputed from mutable/repeatable fields. Stable
        across restarts, and structurally cannot collide with another
        manual entry or with a statement-derived txn_key_hash (those are
        sha256 of business-key strings — a disjoint input domain from
        "manual::<uuid4>").
        """
        return hashlib.sha256(f"manual::{self.entry_id}".encode("utf-8")).hexdigest()

    @property
    def has_explicit_category(self) -> bool:
        return bool(self.category) and self.category != UNSET_CATEGORY_SENTINEL

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["txn_date"] = self.txn_date.isoformat()
        d["logged_at"] = self.logged_at.isoformat()
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "ManualTransaction":
        return cls(
            entry_id=d["entry_id"],
            amount=float(d["amount"]),
            merchant=d["merchant"],
            txn_date=date.fromisoformat(d["txn_date"]),
            category=d["category"],
            logged_at=datetime.fromisoformat(d["logged_at"]),
        )


def _load_all() -> list[ManualTransaction]:
    if not MANUAL_TRANSACTIONS_PATH.exists():
        return []
    with open(MANUAL_TRANSACTIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ManualTransaction.from_json_dict(d) for d in raw.get("entries", [])]


def _save_all(entries: list[ManualTransaction]) -> None:
    ensure_dir(MANUAL_DIR)
    tmp_path = MANUAL_TRANSACTIONS_PATH.with_suffix(".tmp")
    payload = {"entries": [e.to_json_dict() for e in entries]}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, MANUAL_TRANSACTIONS_PATH)


def append_manual_transaction(amount: float, merchant: str, txn_date: date, category: str) -> ManualTransaction:
    """Appends one manual entry and persists it immediately."""
    entry = ManualTransaction(
        entry_id=uuid.uuid4().hex,
        amount=amount,
        merchant=merchant.strip(),
        txn_date=txn_date,
        category=category,
    )
    entries = _load_all()
    entries.append(entry)
    _save_all(entries)
    logger.info(
        "Logged manual transaction %s: ₹%.2f at '%s' (category=%s, override=%s)",
        entry.entry_id[:8], entry.amount, entry.merchant, entry.category, entry.has_explicit_category,
    )
    return entry


def load_manual_transactions() -> list[ManualTransaction]:
    return _load_all()


def get_manual_category_overrides() -> dict[str, str]:
    return {e.txn_key_hash: e.category for e in _load_all() if e.has_explicit_category}


def manual_transactions_as_dataframe() -> pd.DataFrame:
    """
    Columns MUST match GOLD_OUTPUT_COLUMNS in src/gold/pipeline.py.
    Returns an empty (but correctly-columned) DataFrame if there are no
    manual entries yet, so pd.concat() downstream never has to special-case
    "no manual entries" as a separate code path.
    """
    entries = _load_all()
    columns = ["txn_key_hash", "txn_date", "description", "amount", "source_type", "source_file"]
    if not entries:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        [
            {
                "txn_key_hash": e.txn_key_hash,
                "txn_date": e.txn_date,
                "description": e.merchant,
                "amount": e.amount,
                "source_type": "MANUAL",
                "source_file": "manual_entry",
            }
            for e in entries
        ]
    )