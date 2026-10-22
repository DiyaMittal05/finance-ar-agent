"""Generate the synthetic ERP database.

Deterministic: the same seed always produces the same rows, so eval gold
answers stay valid. A handful of hand-built customers carry the tricky cases
the eval suite targets (ambiguous names, duplicates, disputes, missing data).
Everything else is seeded random filler.

    python -m app.data.generate
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from app.config import settings

SCHEMA = """
CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    contact_name TEXT,
    contact_email TEXT,
    payment_terms TEXT NOT NULL,
    terms_days INTEGER NOT NULL,
    credit_limit_cents INTEGER NOT NULL,
    account_status TEXT NOT NULL
);
CREATE TABLE invoices (
    invoice_number TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    issue_date TEXT NOT NULL,
    due_date TEXT,
    amount_cents INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL,
    description TEXT,
    dispute_reason TEXT
);
CREATE TABLE payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT NOT NULL REFERENCES invoices(invoice_number),
    payment_date TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    method TEXT NOT NULL
);
CREATE INDEX idx_invoices_customer ON invoices(customer_id);
CREATE INDEX idx_payments_invoice ON payments(invoice_number);
"""

TERMS = {"Net 30": 30, "Net 45": 45, "Net 60": 60, "2/10 Net 30": 30}

# (id, name, contact, terms, credit limit $, account status)
SPECIAL_CUSTOMERS = [
    ("C001", "Acme Corporation", "Dana Whitfield", "Net 45", 50_000, "active"),
    ("C002", "Acme Industrial Supply", "Ravi Menon", "Net 30", 20_000, "active"),
    ("C003", "Globex Ltd", "Hank Scorpio", "Net 30", 40_000, "active"),
    ("C004", "Initech", "Bill Lumbergh", "Net 30", 15_000, "active"),
    ("C005", "Umbrella Health", "Alice Abernathy", "Net 60", 60_000, "active"),
    ("C006", "Stark Logistics", "Pepper Potts", "Net 30", 25_000, "active"),
    ("C007", "Wayne Enterprises", "Lucius Fox", "Net 30", 80_000, "active"),
    ("C008", "Hooli", "Gavin Belson", "Net 30", 10_000, "active"),
    ("C009", "Soylent Foods", "Sol Roth", "2/10 Net 30", 30_000, "on_hold"),
]

# (invoice, customer, due date or None, amount $, status, description, payments, dispute)
# payments: list of (date, amount $). Issue date = due date - terms (or set explicitly).
SPECIAL_INVOICES = [
    ("INV-2026-0101", "C001", "2026-09-10", 8_250.00, "open", "PO-77812 Packaging materials Q3", [], None),
    ("INV-2026-0102", "C001", "2026-08-16", 12_400.00, "open", "PO-77654 Warehouse shelving", [], None),
    ("INV-2026-0103", "C001", "2026-07-17", 3_975.50, "open", "PO-77420 Safety equipment", [], None),
    ("INV-2026-0104", "C001", "2026-10-20", 6_100.00, "open", "PO-77990 Pallet wrap", [], None),
    ("INV-2026-0105", "C001", "2026-06-20", 9_800.00, "paid", "PO-77101 Forklift parts", [("2026-06-18", 9_800.00)], None),
    ("INV-2026-0201", "C002", "2026-09-05", 4_320.00, "open", "Industrial fasteners", [], None),
    ("INV-2026-0202", "C002", "2026-07-30", 2_150.00, "paid", "Hydraulic fittings", [("2026-07-28", 2_150.00)], None),
    ("INV-2026-0301", "C003", "2026-07-31", 15_000.00, "open", "Consulting retainer July", [], None),
    ("INV-2026-0302", "C003", "2026-07-31", 15_000.00, "open", "Consulting retainer July", [], None),
    ("INV-2026-0303", "C003", "2026-09-20", 2_480.00, "open", "Travel expenses", [], None),
    ("INV-2026-0401", "C004", "2026-08-20", 7_600.00, "disputed", "Software licences x40", [],
     "Customer says the contract covers 32 licences, not 40"),
    ("INV-2026-0402", "C004", "2026-09-15", 1_890.00, "open", "Support hours August", [], None),
    ("INV-2026-0501", "C005", "2026-08-01", 20_000.00, "partially_paid", "Medical supplies batch 14",
     [("2026-08-20", 12_500.00)], None),
    ("INV-2026-0502", "C005", "2026-11-15", 5_400.00, "open", "Medical supplies batch 15", [], None),
    ("INV-2026-0601", "C006", None, 6_750.00, "open", "Freight services July", [], None),
    ("INV-2026-0602", "C006", "2026-09-12", 2_300.00, "open", "Freight services August", [], None),
    ("INV-2026-0701", "C007", "2026-08-15", 11_000.00, "paid", "Security consulting", [("2026-08-10", 11_000.00)], None),
    ("INV-2026-0702", "C007", "2026-09-14", 4_500.00, "paid", "Equipment rental", [("2026-09-12", 4_500.00)], None),
    ("INV-2026-0901", "C009", "2026-05-25", 18_900.00, "open", "Bulk ingredients April", [], None),
    ("INV-2026-0902", "C009", "2026-06-30", 4_200.00, "open", "Bulk ingredients May", [], None),
]
SPECIAL_ISSUE_DATES = {"INV-2026-0601": "2026-07-15"}

PREFIXES = ["Northwind", "Bluepeak", "Cobalt", "Evergreen", "Harbor", "Ironwood", "Juniper",
            "Keystone", "Lumen", "Meridian", "Nimbus", "Orion", "Pinecrest", "Redwood",
            "Summit", "Tidewater", "Vantage", "Willow", "Zephyr", "Granite", "Falcon"]
SUFFIXES = ["Analytics", "Foods", "Logistics", "Labs", "Manufacturing", "Retail", "Systems",
            "Partners", "Energy", "Textiles"]
FIRST = ["Jordan", "Priya", "Marcus", "Elena", "Kenji", "Amara", "Lucas", "Sofia", "Omar", "Grace",
         "Mateo", "Hana", "Noah", "Zara", "Ethan", "Leila", "Samuel", "Ines", "Tomas", "Yara"]
LAST = ["Okafor", "Lindqvist", "Patel", "Moreau", "Tanaka", "Kowalski", "Haddad", "Romero",
        "Nguyen", "Fischer", "Mensah", "Costa", "Ibrahim", "Larsen", "Silva", "Novak"]
ITEMS = ["Office supplies", "Cloud hosting", "Maintenance contract", "Raw materials",
         "Logistics services", "Marketing services", "Equipment lease", "Spare parts",
         "Training workshop", "Data subscription"]


def cents(dollars: float) -> int:
    return int(round(dollars * 100))


def slug(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def build(db_path: Path | None = None, seed: int = 7) -> Path:
    db_path = db_path or settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    rng = random.Random(seed)
    as_of = settings.as_of_date
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    for cid, name, contact, terms, limit, status in SPECIAL_CUSTOMERS:
        email = f"{contact.split()[0].lower()}.{contact.split()[-1].lower()}@{slug(name)}.example"
        conn.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)",
            (cid, name, contact, email, terms, TERMS[terms], cents(limit), status),
        )
    terms_by_customer = {c[0]: TERMS[c[3]] for c in SPECIAL_CUSTOMERS}

    for number, cid, due, amount, status, desc, payments, dispute in SPECIAL_INVOICES:
        if number in SPECIAL_ISSUE_DATES:
            issue = SPECIAL_ISSUE_DATES[number]
        else:
            issue = (date.fromisoformat(due) - timedelta(days=terms_by_customer[cid])).isoformat()
        conn.execute(
            "INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?)",
            (number, cid, issue, due, cents(amount), "USD", status, desc, dispute),
        )
        for pay_date, pay_amount in payments:
            conn.execute(
                "INSERT INTO payments (invoice_number, payment_date, amount_cents, method) VALUES (?,?,?,?)",
                (number, pay_date, cents(pay_amount), "ACH"),
            )

    # Seeded random filler customers, C010 onwards.
    used_names: set[str] = set()
    invoice_seq = 1000
    for i in range(10, 31):
        while True:
            name = f"{rng.choice(PREFIXES)} {rng.choice(SUFFIXES)}"
            if name not in used_names:
                used_names.add(name)
                break
        cid = f"C{i:03d}"
        contact = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        terms = rng.choice(["Net 30", "Net 30", "Net 45", "Net 60"])
        email = f"{contact.split()[0].lower()}.{contact.split()[1].lower()}@{slug(name)}.example"
        conn.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)",
            (cid, name, contact, email, terms, TERMS[terms], cents(rng.choice([10, 20, 30, 50]) * 1000), "active"),
        )
        for _ in range(rng.randint(2, 6)):
            invoice_seq += 1
            number = f"INV-2026-{invoice_seq}"
            issue = as_of - timedelta(days=rng.randint(5, 150))
            due = issue + timedelta(days=TERMS[terms])
            amount = round(rng.uniform(400, 18_000), 2)
            roll = rng.random()
            pays: list[tuple[str, float]] = []
            if due > as_of or roll < 0.45:
                status = "open"
                if due <= as_of and roll < 0.3:
                    # Paid in full, on or after the due date.
                    status = "paid"
                    pay_date = min(due + timedelta(days=rng.randint(-5, 10)), as_of)
                    pays.append((pay_date.isoformat(), amount))
            elif roll < 0.6:
                status = "partially_paid"
                pays.append(((due - timedelta(days=2)).isoformat(), round(amount * rng.uniform(0.2, 0.7), 2)))
            else:
                status = "open"
            conn.execute(
                "INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?)",
                (number, cid, issue.isoformat(), due.isoformat(), cents(amount), "USD", status,
                 rng.choice(ITEMS), None),
            )
            for pay_date, pay_amount in pays:
                conn.execute(
                    "INSERT INTO payments (invoice_number, payment_date, amount_cents, method) VALUES (?,?,?,?)",
                    (number, pay_date, cents(pay_amount), rng.choice(["ACH", "wire", "card"])),
                )
    conn.commit()
    conn.close()
    return db_path


def ensure_db() -> Path:
    if not settings.db_path.exists():
        build()
    return settings.db_path


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
