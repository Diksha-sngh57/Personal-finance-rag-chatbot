from __future__ import annotations
import torch  # noqa: F401 — must load before pandas/pyarrow to avoid WinError 1114 (see chat history)

import io
import logging
from datetime import date
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.bronze.pipeline import run_bronze_pipeline
from src.bronze.upload_handler import save_uploaded_file
from src.config import CATEGORY_OPTIONS, GOLD_TRANSACTIONS_PATH, UPLOADS_DIR
from src.gold.manual_overrides import append_manual_transaction, load_manual_transactions
from src.gold.pipeline import run_gold_pipeline
from src.rag.chat import answer_question
from src.rag.indexer import build_or_refresh_index
from src.rag.vector_store import collection_count
from src.silver.pipeline import run_silver_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

APP_TITLE = "💰 Personal Finance Intelligence"
APP_SUBTITLE = "Upload a statement for historical backfill, or log a transaction manually."

ALLOWED_STATEMENT_EXTENSIONS = ("pdf", "csv")
MAX_STATEMENT_SIZE_MB = 20
CSV_PREVIEW_ROWS = 5


def init_session_state() -> None:
    if "last_pipeline_error" not in st.session_state:
        st.session_state.last_pipeline_error: Optional[str] = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2.5rem; max-width: 1200px; }
            .card-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.15rem; }
            .card-subtitle { font-size: 0.88rem; color: #6B7280; margin-bottom: 1rem; }
            .app-subtitle { color: #6B7280; font-size: 1rem; margin-top: -0.6rem; margin-bottom: 1.8rem; }
            div[data-testid="stForm"] { border: none; padding: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _preview_csv(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, nrows=CSV_PREVIEW_ROWS)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            st.warning("The uploaded CSV appears to be empty.")
            return None
        except pd.errors.ParserError as exc:
            st.warning(f"Could not parse CSV rows for preview: {exc}")
            return None
    st.warning("Could not decode this CSV with utf-8, latin-1, or cp1252. It will still be saved as-is.")
    return None


def render_upload_card() -> None:
    with st.container(border=True):
        st.markdown('<div class="card-title">📄 Upload a Statement</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">PDF or CSV — for historical backfill of past transactions.</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            label="Choose a bank or UPI statement",
            type=list(ALLOWED_STATEMENT_EXTENSIONS),
            accept_multiple_files=False,
            key="statement_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.caption(
                f"Accepted formats: {', '.join(e.upper() for e in ALLOWED_STATEMENT_EXTENSIONS)} "
                f"· Max {MAX_STATEMENT_SIZE_MB} MB"
            )
            return

        size_kb = round(uploaded_file.size / 1024, 2)
        size_mb = size_kb / 1024

        if size_mb > MAX_STATEMENT_SIZE_MB:
            st.error(f"'{uploaded_file.name}' is {size_mb:.1f} MB, exceeding the {MAX_STATEMENT_SIZE_MB} MB limit.")
            return

        raw_bytes = uploaded_file.getvalue()
        if len(raw_bytes) == 0:
            st.error(f"'{uploaded_file.name}' is empty (0 bytes).")
            return

        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""

        if file_ext == "csv":
            preview_df = _preview_csv(raw_bytes)
            if preview_df is not None:
                st.success(f"'{uploaded_file.name}' loaded — showing first {len(preview_df)} rows:")
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
        elif file_ext == "pdf":
            st.success(f"'{uploaded_file.name}' received ({size_kb} KB).")

        if st.button("💾 Save & Stage for Processing", key="save_upload_btn", use_container_width=True):
            saved_path = save_uploaded_file(uploaded_file)
            st.toast(f"Saved to {saved_path.name}", icon="✅")
            st.info("Staged. Run **Full Pipeline** below to ingest it.", icon="ℹ️")

        staged_files = sorted(UPLOADS_DIR.glob("*"))
        staged_files = [p for p in staged_files if p.is_file() and p.suffix.lower() in (".pdf", ".csv")]
        if staged_files:
            with st.expander(f"📂 {len(staged_files)} file(s) currently in data/uploads/"):
                for p in staged_files:
                    st.caption(f"• {p.name} ({round(p.stat().st_size / 1024, 1)} KB)")


def render_manual_entry_card() -> None:
    with st.container(border=True):
        st.markdown('<div class="card-title">✍️ Log a Transaction</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">For quick, day-to-day manual entries.</div>', unsafe_allow_html=True)

        with st.form(key="manual_transaction_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                amount = st.number_input(
                    "Amount (₹)", min_value=0.0, step=10.0, format="%.2f",
                    help="Always enter a positive number — use Direction below for sign.",
                )
            with col_b:
                txn_date = st.date_input("Date", value=date.today(), max_value=date.today())

            direction = st.radio(
                "Direction", options=["Expense (money out)", "Income (money in)"],
                horizontal=True,
            )
            merchant = st.text_input("Merchant / Payee", placeholder="e.g. Swiggy, Uber, Landlord")
            category = st.selectbox(
                "Category (optional — leave as Uncategorized to let the AI decide)",
                options=CATEGORY_OPTIONS, index=0,
            )

            submitted = st.form_submit_button("Add Transaction", use_container_width=True)

        if not submitted:
            return

        errors = []
        if amount is None or amount <= 0:
            errors.append("Amount must be greater than 0.")
        if not merchant or not merchant.strip():
            errors.append("Merchant / Payee cannot be empty.")
        if txn_date > date.today():
            errors.append("Date cannot be in the future.")

        if errors:
            for err in errors:
                st.error(err)
            return

        signed_amount = -abs(float(amount)) if direction.startswith("Expense") else abs(float(amount))

        entry = append_manual_transaction(
            amount=round(signed_amount, 2), merchant=merchant.strip(), txn_date=txn_date, category=category,
        )
        override_note = (
            " (category locked — will skip the AI)" if entry.has_explicit_category
            else " (AI will categorize this)"
        )
        st.toast(f"Logged ₹{abs(entry.amount):,.2f} at {entry.merchant}{override_note}", icon="✅")


def render_manual_log_view() -> None:
    entries = load_manual_transactions()
    if not entries:
        return
    with st.expander(f"🔍 {len(entries)} manual entr{'y' if len(entries) == 1 else 'ies'} logged so far"):
        df = pd.DataFrame(
            [
                {
                    "Amount (₹)": e.amount,
                    "Merchant": e.merchant,
                    "Date": e.txn_date.isoformat(),
                    "Category": e.category,
                    "Logged At": e.logged_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for e in reversed(entries)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def _run_index_refresh(status) -> None:
    """Shared post-Gold indexing step. Never fails the pipeline run on index errors."""
    try:
        status.write("Step: syncing vector index...")
        result = build_or_refresh_index()
        status.write(f"🔍 Index synced: +{result['upserted']} upserted / -{result['removed_stale']} removed")
    except (FileNotFoundError, ValueError) as exc:
        status.write(f"⚠️ Index sync skipped: {exc}")
    except Exception as exc:  # noqa: BLE001
        status.write(f"⚠️ Index sync failed (Gold data is still valid): {exc}")
        logging.exception("Index refresh failed after Gold pipeline")


def render_pipeline_controls() -> None:
    st.divider()
    st.subheader("⚙️ Pipeline Controls")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚡ Quick Refresh**")
        st.caption("Gold only. Fast — no Spark. Use after logging manual entries.")
        if st.button("Run Quick Refresh", use_container_width=True):
            with st.status("Running Gold pipeline...", expanded=True) as status:
                try:
                    output_path = run_gold_pipeline()
                    status.write(f"✅ Gold written: {output_path.name}")
                    _run_index_refresh(status)
                    status.update(label="Done", state="complete")
                    st.session_state.last_pipeline_error = None
                except FileNotFoundError as exc:
                    status.update(label="No Silver data yet", state="error")
                    st.error(f"{exc}\n\nRun **Full Pipeline** first if you haven't uploaded any statements yet.")
                except RuntimeError as exc:
                    status.update(label="Configuration error", state="error")
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    status.update(label="Gold pipeline failed", state="error")
                    st.error(f"Unexpected error in Gold pipeline: {exc}")
                    logging.exception("Gold pipeline failed")

    with col2:
        st.markdown("**🔄 Full Pipeline**")
        st.caption("Bronze → Silver → Gold → Index. Required after uploading a new statement.")
        if st.button("Run Full Pipeline", type="primary", use_container_width=True):
            with st.status("Running full pipeline...", expanded=True) as status:
                try:
                    status.write("Step 1/4 — Bronze (ingesting staged statements)...")
                    bronze_output = run_bronze_pipeline(source_dir=UPLOADS_DIR)
                    if bronze_output:
                        status.write(f"✅ Bronze batch written: {bronze_output.name}")
                    else:
                        status.write("ℹ️ No new files to ingest (nothing staged, or already processed).")

                    status.write("Step 2/4 — Silver (validating + deduplicating)...")
                    transactions_path, quarantine_path = run_silver_pipeline()
                    status.write(f"✅ Silver written: {transactions_path.name}")
                    if quarantine_path:
                        status.write(f"⚠️ Some rows quarantined — see {quarantine_path.name}")

                    status.write("Step 3/4 — Gold (categorizing)...")
                    gold_output = run_gold_pipeline()
                    status.write(f"✅ Gold written: {gold_output.name}")

                    status.write("Step 4/4 — Vector index...")
                    _run_index_refresh(status)

                    status.update(label="Full pipeline complete", state="complete")
                    st.session_state.last_pipeline_error = None

                except FileNotFoundError as exc:
                    status.update(label="Nothing to process", state="error")
                    st.error(f"{exc}\n\nUpload and stage a statement first (Card 1 above).")
                except ValueError as exc:
                    status.update(label="Silver validation failed", state="error")
                    st.error(str(exc))
                except RuntimeError as exc:
                    status.update(label="Configuration error", state="error")
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"Unexpected error: {exc}")
                    logging.exception("Full pipeline failed")


def render_dashboard() -> None:
    st.divider()
    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.subheader("📊 Dashboard")
    with action_col:
        if st.button("🔁 Rebuild Index", use_container_width=True, help="Manually re-sync the vector index from Gold."):
            with st.spinner("Rebuilding vector index..."):
                try:
                    result = build_or_refresh_index()
                    st.toast(
                        f"Index rebuilt: +{result['upserted']} upserted / -{result['removed_stale']} removed",
                        icon="✅",
                    )
                except (FileNotFoundError, ValueError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Index rebuild failed: {exc}")
                    logging.exception("Manual index rebuild failed")

    if not GOLD_TRANSACTIONS_PATH.exists():
        st.info("No Gold data yet. Upload a statement and run the **Full Pipeline** above to populate this dashboard.")
        return

    df = pd.read_parquet(GOLD_TRANSACTIONS_PATH, engine="pyarrow")
    if df.empty:
        st.info("Gold dataset is empty.")
        return

    df["txn_date"] = pd.to_datetime(df["txn_date"])

    filt_col1, filt_col2, filt_col3 = st.columns([2, 2, 1])
    with filt_col1:
        min_date, max_date = df["txn_date"].min().date(), df["txn_date"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with filt_col2:
        categories = sorted(df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Categories", options=categories, default=categories)
    with filt_col3:
        source_filter = st.selectbox("Source", options=["All", "Statement", "Manual"])

    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["txn_date"].dt.date >= start_date) & (filtered["txn_date"].dt.date <= end_date)
        ]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if source_filter == "Statement":
        filtered = filtered[filtered["source_type"] != "MANUAL"]
    elif source_filter == "Manual":
        filtered = filtered[filtered["source_type"] == "MANUAL"]

    if filtered.empty:
        st.warning("No transactions match the current filters.")
        return

    total_income = filtered.loc[filtered["amount"] > 0, "amount"].sum()
    total_expense = -filtered.loc[filtered["amount"] < 0, "amount"].sum()
    net = total_income - total_expense

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Income", f"₹{total_income:,.0f}")
    kpi2.metric("Total Expense", f"₹{total_expense:,.0f}")
    kpi3.metric("Net", f"₹{net:,.0f}")
    kpi4.metric("Transactions", f"{len(filtered):,}")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        expense_by_category = (
            filtered[(filtered["amount"] < 0) & (filtered["category"] != "Income")]
            .assign(spend=lambda d: -d["amount"])
            .groupby("category", as_index=False)["spend"].sum()
            .sort_values("spend", ascending=False)
        )
        if not expense_by_category.empty:
            fig = px.pie(expense_by_category, names="category", values="spend", title="Spend by Category", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No expense transactions in the current filter to chart.")

    with chart_col2:
        monthly = filtered.copy()
        monthly["month"] = monthly["txn_date"].dt.to_period("M").astype(str)
        monthly_summary = monthly.groupby("month", as_index=False)["amount"].sum()
        if not monthly_summary.empty:
            fig2 = px.bar(monthly_summary, x="month", y="amount", title="Net Cash Flow by Month")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Transactions**")
    display_df = filtered.sort_values("txn_date", ascending=False)[
        ["txn_date", "description", "amount", "category", "source_type"]
    ].rename(columns={
        "txn_date": "Date", "description": "Description", "amount": "Amount (₹)",
        "category": "Category", "source_type": "Source",
    })
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_chat_section() -> None:
    st.divider()
    st.subheader("💬 Ask Your Finances")

    idx_size = collection_count()
    if idx_size == 0:
        st.info("No data indexed yet. Run **Quick Refresh** or **Full Pipeline** above at least once first.")
        return

    st.caption(
        f"{idx_size} transaction(s) indexed. Ask things like "
        f"*\"how much did I spend on food last month\"* or *\"show me my Uber transactions\"*."
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your transactions...")
    if not user_query:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer_text = answer_question(user_query)
            except RuntimeError as exc:
                answer_text = str(exc)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Chat query failed")
                answer_text = f"Sorry, I couldn't generate an answer right now ({exc}). Please try again."
        st.markdown(answer_text)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})


def main() -> None:
    st.set_page_config(page_title="Personal Finance Intelligence", page_icon="💰", layout="wide")

    init_session_state()
    inject_css()

    st.title(APP_TITLE)
    st.markdown(f'<div class="app-subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        render_upload_card()
    with right_col:
        render_manual_entry_card()

    render_manual_log_view()
    render_pipeline_controls()
    render_dashboard()
    render_chat_section()


if __name__ == "__main__":
    main()