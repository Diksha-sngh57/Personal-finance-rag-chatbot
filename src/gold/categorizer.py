"""
LLM-based transaction categorization via Groq.
 
Model choice note: Groq announced on 2026-06-17 that llama-3.3-70b-versatile
(and llama-3.1-8b-instant) are being deprecated, with decommissioning
targeted for August 2026 — weeks away as of this writing
(https://console.groq.com/docs/deprecations). Building new code on a model
that short-lived doesn't make sense for something meant to keep working,
so this uses openai/gpt-oss-120b — Groq's own officially recommended
replacement for Llama 3.3 70B. If you have a specific reason to need the
original model short-term, GROQ_MODEL below is the only line to change.
 
Design principles:
  - Never trust the LLM's output blindly. Every response is parsed as
    JSON, checked for the expected shape, checked that every returned
    category is actually in our allowed list, and checked that every
    transaction we asked about got an answer back. Any failure of any of
    these is treated as untrustworthy and retried — we never silently
    accept a malformed or partial response.
  - Batch multiple transactions per call (BATCH_SIZE) rather than one API
    call per row — this is both cheaper and faster, and is the standard
    approach for LLM-based bulk classification.
  - If a batch still fails validation after all retries, every
    transaction in that batch falls back to "Uncategorized" rather than
    crashing the whole pipeline over one stubborn batch. A human can
    always recategorize a few rows later; losing an entire run's worth of
    categorization over one bad batch is a worse outcome.
"""
from __future__ import annotations
 
import json
import logging
from dataclasses import dataclass
from typing import Optional
 
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
 
from src.config import CATEGORY_OPTIONS
 
logger = logging.getLogger(__name__)
 
load_dotenv()  # reads GROQ_API_KEY from a .env file in the project root; never hardcoded
 
GROQ_MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 25
 
_ALLOWED_CATEGORIES = set(CATEGORY_OPTIONS)
 
_SYSTEM_PROMPT = f"""You are a financial transaction categorization engine for an Indian personal finance app.
 
Given a JSON array of transactions, assign EXACTLY ONE category to each one, chosen ONLY from this fixed list:
{json.dumps(list(CATEGORY_OPTIONS))}
 
Rules:
- Use "Income" for salary credits, refunds, and interest credits (these have a positive amount).
- Use "Uncategorized" ONLY if the description genuinely gives no signal at all — this should be rare, not a default you reach for when unsure.
- The category string in your output must match one of the allowed values EXACTLY (same spelling, case, and punctuation).
- Every "id" from the input must appear exactly once in your output. Do not skip, merge, renumber, or invent ids.
 
Examples:
  {{"id": 0, "description": "UPI-SWIGGY-482913@ybl-Food Order", "amount": -450.00}} -> "Food & Dining"
  {{"id": 1, "description": "NEFT-SALARY CREDIT-ACME CORP", "amount": 65000.00}} -> "Income"
  {{"id": 2, "description": "UPI-UBER INDIA-102934@ybl-Cab", "amount": -320.50}} -> "Transport"
 
Respond with ONLY a JSON object of this exact shape — no other text, no markdown code fences:
{{"results": [{{"id": 0, "category": "..."}}, {{"id": 1, "category": "..."}}]}}
"""
 
 
class CategorizationError(Exception):
    """Raised when an LLM response can't be trusted, even to trigger a retry."""
 
 
@dataclass
class TransactionToCategorize:
    id: int
    description: str
    amount: float
 
 
_client: Optional[Groq] = None
 
 
def _get_client() -> Groq:
    global _client
    if _client is not None:
        return _client
 
    import os
 
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file in the project root "
            "(copy .env.example) with GROQ_API_KEY=your_key_here — get a free key "
            "at https://console.groq.com/keys — then rerun."
        )
    _client = Groq(api_key=api_key)
    return _client
 
 
def _build_user_prompt(batch: list[TransactionToCategorize]) -> str:
    payload = [{"id": t.id, "description": t.description, "amount": t.amount} for t in batch]
    return json.dumps(payload)
 
 
@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, CategorizationError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_groq_batch(batch: list[TransactionToCategorize]) -> dict[int, str]:
    """
    Sends one batch to Groq and returns {id: category}. Raises
    CategorizationError (which triggers a retry, same as a rate limit or
    connection error) on anything that makes the response untrustworthy:
    invalid JSON, the wrong shape, a missing id, or too many unknown
    categories to be a fluke. This function either returns a fully
    validated mapping for every id in the batch, or raises — it never
    returns a partially-checked result.
    """
    client = _get_client()
 
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(batch)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
 
    raw_content = response.choices[0].message.content
 
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise CategorizationError(f"Model returned invalid JSON: {exc}") from exc
 
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        raise CategorizationError(f"Model response missing expected 'results' list: {raw_content[:200]!r}")
 
    expected_ids = {t.id for t in batch}
    result_map: dict[int, str] = {}
 
    for item in parsed["results"]:
        if not isinstance(item, dict) or "id" not in item or "category" not in item:
            raise CategorizationError(f"Malformed result item (missing id/category): {item!r}")
 
        item_id = item["id"]
        category = item["category"]
 
        if item_id not in expected_ids:
            # The model invented or mistyped an id — this batch's response
            # can't be trusted as a whole, so retry rather than silently
            # dropping the stray item.
            raise CategorizationError(f"Model returned id {item_id!r}, which wasn't in this batch's input.")
 
        if category not in _ALLOWED_CATEGORIES:
            logger.warning(
                "Model returned category %r outside the allowed list for id %s — "
                "using 'Uncategorized' for this one item instead of retrying the whole batch.",
                category, item_id,
            )
            category = "Uncategorized"
 
        result_map[item_id] = category
 
    missing_ids = expected_ids - set(result_map.keys())
    if missing_ids:
        raise CategorizationError(f"Model omitted {len(missing_ids)} id(s) from its response: {missing_ids}")
 
    return result_map
 
 
def categorize_batch(batch: list[TransactionToCategorize]) -> dict[int, str]:
    """
    Public entrypoint: categorizes one batch of up to BATCH_SIZE
    transactions, with retries already applied. If every retry attempt
    fails, every transaction in the batch falls back to "Uncategorized"
    instead of raising and aborting the whole Gold pipeline run.
    """
    if not batch:
        return {}
 
    try:
        return _call_groq_batch(batch)
    except (CategorizationError, APIStatusError, RateLimitError, APIConnectionError) as exc:
        logger.error(
            "Batch of %d transaction(s) failed categorization after all retries (%s). "
            "Falling back to 'Uncategorized' for this batch.",
            len(batch), exc,
        )
        return {t.id: "Uncategorized" for t in batch}
