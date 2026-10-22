"""Read-only data access over the synthetic ERP SQLite database."""
from __future__ import annotations

import difflib
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import settings
from app.data.generate import ensure_db

NAME_STOPWORDS = {"inc", "ltd", "llc", "corp", "corporation", "co", "the", "company"}
# Industry words shared by many account names. Matching on one of these alone
# ("Cyberdyne Systems" vs "Harbor Systems") is not evidence of the same customer.
GENERIC_TOKENS = {"systems", "foods", "logistics", "labs", "manufacturing", "retail", "partners",
                  "energy", "textiles", "analytics", "health", "supply", "industrial", "enterprises",
                  "group", "holdings", "services", "solutions", "international", "global"}


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_db()
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _customer(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["credit_limit"] = d.pop("credit_limit_cents") / 100
    return d


def _invoice(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["amount"] = d.pop("amount_cents") / 100
    d["paid_amount"] = (d.pop("paid_cents", 0) or 0) / 100
    return d


def _tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in NAME_STOPWORDS}


def search_customers(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Match customers by exact name, token overlap, then fuzzy similarity (typos)."""
    q = query.strip().lower()
    q_tokens = _tokens(q)
    with connect() as conn:
        rows = [_customer(r) for r in conn.execute("SELECT * FROM customers")]
    scored = []
    for c in rows:
        name = c["name"].lower()
        if q == name:
            score = 1.0
        elif len(q.split()) > 1 and name.startswith(q):
            score = 0.95
        elif q_tokens and q_tokens <= _tokens(name):
            score = 0.9
        elif (q_tokens & _tokens(name)) - GENERIC_TOKENS:
            score = 0.6
        else:
            score = max(
                difflib.SequenceMatcher(None, q, name).ratio(),
                max((difflib.SequenceMatcher(None, qt, nt).ratio()
                     for qt in q_tokens - GENERIC_TOKENS for nt in _tokens(name) - GENERIC_TOKENS),
                    default=0) * 0.85,
            )
            if score < 0.7:
                continue
        scored.append({"id": c["id"], "name": c["name"], "match_score": round(score, 2)})
    scored.sort(key=lambda m: -m["match_score"])
    # An exact name match is unambiguous even if it also shares tokens with others.
    if scored and scored[0]["match_score"] == 1.0:
        return scored[:1]
    return scored[:limit]


def get_customer(customer_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return _customer(row) if row else None


INVOICE_SELECT = """
SELECT i.*, COALESCE((SELECT SUM(p.amount_cents) FROM payments p
                      WHERE p.invoice_number = i.invoice_number), 0) AS paid_cents
FROM invoices i
"""


def list_invoices(customer_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    sql, args = INVOICE_SELECT + " WHERE 1=1", []
    if customer_id:
        sql += " AND i.customer_id = ?"
        args.append(customer_id)
    if status:
        sql += " AND i.status = ?"
        args.append(status)
    sql += " ORDER BY i.due_date IS NULL, i.due_date"
    with connect() as conn:
        return [_invoice(r) for r in conn.execute(sql, args)]


def get_invoice(invoice_number: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(INVOICE_SELECT + " WHERE i.invoice_number = ?",
                           (invoice_number.upper(),)).fetchone()
        if not row:
            return None
        inv = _invoice(row)
        inv["payments"] = [
            {"payment_date": p["payment_date"], "amount": p["amount_cents"] / 100, "method": p["method"]}
            for p in conn.execute(
                "SELECT * FROM payments WHERE invoice_number = ? ORDER BY payment_date",
                (inv["invoice_number"],))
        ]
    return inv


def customer_names() -> dict[str, str]:
    with connect() as conn:
        return {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM customers")}
