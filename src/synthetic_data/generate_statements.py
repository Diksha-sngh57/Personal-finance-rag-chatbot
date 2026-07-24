"""
Synthetic bank/UPI statement generator.
 
Produces realistic-looking PDF and CSV statements purely for pipeline
testing — this is what lets us build and validate the Bronze ETL without
needing real bank data. Two different source *shapes* are generated on
purpose, matching two real-world export styles:
 
    - A PDF in a classic bank-statement table layout
      (Date / Description / Debit / Credit / Balance)
      -> exercises src/bronze/pdf_extractor.py
 
    - A CSV in a UPI-app export style with DIFFERENT header names
      (Txn Date / Narration / Withdrawal / Deposit / Closing Balance)
      -> exercises the column-alias mapping in src/bronze/csv_extractor.py,
         instead of trivially matching the PDF's own column names.
"""
from __future__ import annotations
 
import csv
import logging
import random
import uuid
from datetime import date, timedelta
from pathlib import Path
 
from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
 
from src.config import SYNTHETIC_DIR
 
logger = logging.getLogger(__name__)
_faker = Faker("en_IN")
 
DEBIT_DESCRIPTIONS = [
    "UPI-SWIGGY-{ref}@ybl-Food Order",
    "UPI-ZOMATO-{ref}@okaxis-Food Order",
    "UPI-AMAZON PAY-{ref}@apl-Shopping",
    "UPI-FLIPKART-{ref}@icici-Shopping",
    "UPI-BIGBASKET-{ref}@hdfcbank-Groceries",
    "UPI-{merchant}-{ref}@paytm-Payment",
    "NEFT-RENT PAYMENT-{ref}",
    "ATM WDL-{branch}",
    "UPI-UBER INDIA-{ref}@ybl-Cab",
    "UPI-IRCTC-{ref}@sbi-Travel",
    "ELECTRICITY BILL-{ref}",
    "UPI-JIO PREPAID-{ref}@icici-Recharge",
    "UPI-NETFLIX-{ref}@okhdfcbank-Entertainment",
]
 
CREDIT_DESCRIPTIONS = [
    "NEFT-SALARY CREDIT-{company}",
    "UPI-{merchant}-{ref}@ybl-Refund",
    "INTEREST CREDIT-SAVINGS AC",
    "UPI-{merchant}-{ref}@okaxis-Received",
]
 
 
def _random_ref() -> str:
    return str(random.randint(100000000000, 999999999999))
 
 
def _random_description(is_credit: bool) -> str:
    template = random.choice(CREDIT_DESCRIPTIONS if is_credit else DEBIT_DESCRIPTIONS)
    return template.format(
        ref=_random_ref(),
        merchant=_faker.company().split()[0].upper(),
        company=_faker.company().upper(),
        branch=_faker.city().upper(),
    )
 
 
def _generate_transactions(num_transactions: int, opening_balance: float, start: date) -> list[dict]:
    """
    Builds a chronological list of synthetic transactions with a running
    balance, biased toward a realistic mix (~72% debit, ~28% credit, with
    one guaranteed salary credit early in the period so downstream
    categorization logic has a clean, unambiguous signal to key off).
    """
    transactions: list[dict] = []
    balance = opening_balance
    current_date = start
 
    for i in range(num_transactions):
        current_date += timedelta(days=random.randint(0, 2))
 
        is_salary = i == 1
        is_credit = is_salary or random.random() < 0.28
 
        if is_salary:
            amount = round(random.uniform(45000, 90000), 2)
        elif is_credit:
            amount = round(random.uniform(50, 5000), 2)
        else:
            amount = round(random.uniform(30, 8000), 2)
            # Don't let one synthetic debit wipe the balance out — keeps
            # generated statements internally plausible.
            amount = round(min(amount, max(balance * 0.4, 50)), 2)
 
        if is_credit:
            balance += amount
        else:
            balance -= amount
        balance = max(round(balance, 2), 0.0)
 
        transactions.append(
            {
                "date": current_date,
                "description": _random_description(is_credit=is_credit),
                "debit": None if is_credit else amount,
                "credit": amount if is_credit else None,
                "balance": balance,
            }
        )
 
    return transactions
 
 
def generate_pdf_statement(
    output_path: Path,
    account_holder: str,
    account_number_masked: str,
    num_transactions: int = 40,
) -> Path:
    start_date = date.today() - timedelta(days=30)
    opening_balance = round(random.uniform(20000, 60000), 2)
    transactions = _generate_transactions(num_transactions, opening_balance, start_date)
 
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("StatementTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=2)
    meta_style = ParagraphStyle("StatementMeta", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
 
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )
 
    elements = [
        Paragraph("SYNTHBANK — Account Statement", title_style),
        Paragraph(f"Account Holder: {account_holder}", meta_style),
        Paragraph(f"Account Number: {account_number_masked}", meta_style),
        Paragraph(
            f"Statement Period: {start_date.strftime('%d %b %Y')} to {date.today().strftime('%d %b %Y')}",
            meta_style,
        ),
        Paragraph(
            "This is a SYNTHETIC statement generated for pipeline testing. No real financial data.",
            meta_style,
        ),
        Spacer(1, 8 * mm),
    ]
 
    table_data = [["Date", "Description", "Debit", "Credit", "Balance"]]
    for txn in transactions:
        table_data.append(
            [
                txn["date"].strftime("%d-%m-%Y"),
                txn["description"],
                f"{txn['debit']:.2f}" if txn["debit"] is not None else "",
                f"{txn['credit']:.2f}" if txn["credit"] is not None else "",
                f"{txn['balance']:.2f}",
            ]
        )
 
    table = Table(table_data, colWidths=[22 * mm, 78 * mm, 24 * mm, 24 * mm, 26 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F5")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 0), (4, -1), "RIGHT"),
            ]
        )
    )
    elements.append(table)
 
    doc.build(elements)
    logger.info("Generated synthetic PDF statement: %s (%d transactions)", output_path.name, num_transactions)
    return output_path
 
 
def generate_csv_statement(output_path: Path, num_transactions: int = 25) -> Path:
    """
    Mimics a UPI-app style CSV export with headers deliberately different
    from the PDF's — this is what exercises csv_extractor.py's column
    alias mapping instead of it trivially matching one hardcoded schema.
    """
    start_date = date.today() - timedelta(days=30)
    opening_balance = round(random.uniform(5000, 20000), 2)
    transactions = _generate_transactions(num_transactions, opening_balance, start_date)
 
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Txn Date", "Narration", "Withdrawal", "Deposit", "Closing Balance"])
        for txn in transactions:
            writer.writerow(
                [
                    txn["date"].strftime("%Y-%m-%d"),
                    txn["description"],
                    f"{txn['debit']:.2f}" if txn["debit"] is not None else "",
                    f"{txn['credit']:.2f}" if txn["credit"] is not None else "",
                    f"{txn['balance']:.2f}",
                ]
            )
 
    logger.info("Generated synthetic CSV statement: %s (%d transactions)", output_path.name, num_transactions)
    return output_path
 
 
def generate_batch(count: int = 3) -> list[Path]:
    """
    Generates `count` PDF statements (each for a distinct fake account
    holder) + 1 UPI-style CSV export into SYNTHETIC_DIR. Filenames use a
    short uuid so repeated runs never collide with or silently overwrite
    prior test data.
    """
    generated: list[Path] = []
 
    for _ in range(count):
        holder = _faker.name()
        masked_account = "XXXX XXXX " + str(random.randint(1000, 9999))
        file_id = uuid.uuid4().hex[:8]
        pdf_path = SYNTHETIC_DIR / f"statement_{file_id}.pdf"
        generate_pdf_statement(pdf_path, account_holder=holder, account_number_masked=masked_account)
        generated.append(pdf_path)
 
    csv_id = uuid.uuid4().hex[:8]
    csv_path = SYNTHETIC_DIR / f"upi_export_{csv_id}.csv"
    generate_csv_statement(csv_path)
    generated.append(csv_path)
 
    return generated
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    generate_batch(count=3)
