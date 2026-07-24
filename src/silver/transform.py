"""
Silver layer transformation logic.
 
Bronze -> Silver responsibilities:
  - Parse the raw string date into a real DateType by trying several
    known formats via Spark's native to_date() + coalesce() — no Python
    UDFs. UDFs force row-by-row Python execution and serialization
    overhead; this stays vectorized and distributed-safe, which is the
    whole point of doing this in Spark rather than pandas.
  - Clean and combine debit_raw/credit_raw into one signed `amount`
    column (negative = money out, positive = money in), stripping
    currency symbols / thousands separators defensively for real-world
    bank exports, even though our synthetic data doesn't need it.
  - Clean the description text (collapse whitespace).
  - Compute a business-level transaction key so the SAME transaction
    appearing in two overlapping statement uploads collapses to one row.
    This is different from Bronze's row_hash, which only catches
    re-ingesting the literal same file byte-for-byte.
  - Flag anything that fails validation with a quarantine_reason instead
    of silently dropping it, so bad data stays visible and debuggable
    rather than just vanishing.
 
Known, documented limitation: the business key is built from
(date, amount, cleaned description, balance). Two genuinely different
transactions sharing an identical description, amount, date, and balance
would collapse into one row. Real bank/UPI exports almost always embed a
unique reference number inside the description (our synthetic generator
does this deliberately, e.g. "UPI-SWIGGY-482913...@ybl"), which makes
this collision very unlikely in practice — but it's a real structural
limitation of statement data that has no first-class transaction ID, not
a bug we can transform our way out of.
"""
from __future__ import annotations
 
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
 
# Tried in order; first format that successfully parses a given string
# wins. Field-width differences (4-digit year first vs. last) mean most
# of these can't ambiguously double-match the same string — e.g.
# "2026-06-15" cannot also parse as dd-MM-yyyy, since "2026" isn't a
# valid day. The one genuine ambiguity is slash-separated dates like
# "05/06/2026", which validly parses as BOTH 5-June (dd/MM/yyyy) and
# 6-May (MM/dd/yyyy). We deliberately list dd/MM/yyyy first so day-first
# wins — the correct default for Indian bank/UPI statements — but this
# is a real assumption, not a proven fact about any specific ambiguous
# date string, and is worth knowing if a US-formatted CSV ever shows up.
DATE_FORMATS = (
    "dd-MM-yyyy",
    "yyyy-MM-dd",
    "dd/MM/yyyy",
    "yyyy/MM/dd",
    "MM/dd/yyyy",
    "d-M-yyyy",
    "d/M/yyyy",
)
 
 
def _parse_amount_column(col_name: str) -> Column:
    """
    Strips currency prefixes and non-numeric characters from a raw amount
    string, converts a now-blank result to NULL instead of failing the
    cast, and casts to double.
 
    Two-stage cleaning, deliberately in this order:
      1. Strip known currency-prefix TOKENS first (Rs., INR, ₹) as whole
         units. This matters because a naive single-pass character-class
         filter like [^0-9.\\-] would keep the period from an abbreviation
         like "Rs." — turning "Rs. 45,000.00" into ".45000.00", which has
         two decimal points and fails to cast. Removing the token whole
         (including its own period) avoids that.
      2. THEN strip everything that isn't a digit, decimal point, or
         minus sign — this removes thousands-separator commas and any
         remaining whitespace.
 
    Scope note: this covers ₹/Rs./INR, the common cases for Indian bank
    and UPI exports. It does not attempt to handle every currency format
    in existence (e.g. a trailing "Dr"/"Cr" suffix) — extend the prefix
    pattern below if a real statement needs it.
    """
    raw = F.col(col_name)
    no_currency_prefix = F.regexp_replace(F.trim(raw), r"(?i)(rs\.?|inr|₹)", "")
    cleaned = F.regexp_replace(F.trim(no_currency_prefix), r"[^0-9.\-]", "")
    cleaned = F.when(cleaned == "", None).otherwise(cleaned)
    return cleaned.cast(DoubleType())
 
 
def apply_silver_transform(bronze_df: DataFrame) -> DataFrame:
    """
    Takes a raw Bronze DataFrame (see pipeline.RAW_ROW_SCHEMA in
    src/bronze/pipeline.py) and returns it with Silver-layer columns
    added: txn_date, debit_amount, credit_amount, balance_amount, amount,
    description, txn_key_hash, and quarantine_reason. Does NOT filter or
    deduplicate anything — that split happens in silver/pipeline.py,
    since quarantined and valid rows need different downstream handling.
    """
    df = bronze_df
 
    date_candidates = [
        F.to_date(F.trim(F.col("transaction_date_raw")), fmt) for fmt in DATE_FORMATS
    ]
    df = df.withColumn("txn_date", F.coalesce(*date_candidates))
 
    df = df.withColumn("debit_amount", _parse_amount_column("debit_raw"))
    df = df.withColumn("credit_amount", _parse_amount_column("credit_raw"))
    df = df.withColumn("balance_amount", _parse_amount_column("balance_raw"))
 
    df = df.withColumn(
        "amount",
        F.when(F.col("credit_amount").isNotNull(), F.col("credit_amount"))
        .when(F.col("debit_amount").isNotNull(), -F.col("debit_amount"))
        .otherwise(F.lit(None).cast(DoubleType())),
    )
 
    df = df.withColumn(
        "description",
        F.when(F.col("description_raw").isNull(), None).otherwise(
            F.trim(F.regexp_replace(F.col("description_raw"), r"\s+", " "))
        ),
    )
    df = df.withColumn(
        "description",
        F.when(F.col("description") == "", None).otherwise(F.col("description")),
    )
 
    df = df.withColumn(
        "txn_key_hash",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.date_format(F.col("txn_date"), "yyyy-MM-dd"), F.lit("")),
                F.coalesce(F.format_number(F.col("amount"), 2), F.lit("")),
                F.coalesce(F.col("description"), F.lit("")),
                F.coalesce(F.format_number(F.col("balance_amount"), 2), F.lit("")),
            ),
            256,
        ),
    )
 
    # Each F.when(...) with no .otherwise() yields NULL when the condition
    # is false, and concat_ws() silently skips NULL arguments — so this
    # naturally joins only the reasons that actually apply, with no stray
    # separators for the ones that don't.
    df = df.withColumn(
        "quarantine_reason",
        F.concat_ws(
            ",",
            F.when(F.col("txn_date").isNull(), F.lit("unparseable_date")),
            F.when(F.col("amount").isNull(), F.lit("missing_amount")),
            F.when(F.col("description").isNull(), F.lit("missing_description")),
        ),
    )
    df = df.withColumn(
        "quarantine_reason",
        F.when(F.col("quarantine_reason") == "", None).otherwise(F.col("quarantine_reason")),
    )
 
    return df
