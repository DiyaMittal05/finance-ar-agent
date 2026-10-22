"""FastAPI service standing in for an ERP / financial system API (read-only)."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from app.config import settings
from app.erp import db
from app.finance.analytics import aging_summary, enrich_invoice

app = FastAPI(title="Synthetic ERP API", version="1.0",
              description="Read-only AR data over a synthetic SQLite database.")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "as_of_date": settings.as_of_date.isoformat()}


@app.get("/customers")
def search_customers(q: str = Query(..., min_length=1), limit: int = 5) -> dict:
    matches = db.search_customers(q, limit)
    top = matches[0]["match_score"] if matches else 0
    close = [m for m in matches if m["match_score"] >= top - 0.05]
    return {"query": q, "matches": matches, "ambiguous": len(close) > 1}


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> dict:
    customer = db.get_customer(customer_id.upper())
    if not customer:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return customer


@app.get("/customers/{customer_id}/invoices")
def customer_invoices(customer_id: str, status: str | None = None) -> dict:
    if not db.get_customer(customer_id.upper()):
        raise HTTPException(404, f"Customer {customer_id} not found")
    invoices = db.list_invoices(customer_id.upper(), status)
    return {"customer_id": customer_id.upper(), "invoices": invoices}


@app.get("/customers/{customer_id}/aging")
def customer_aging(customer_id: str) -> dict:
    customer = db.get_customer(customer_id.upper())
    if not customer:
        raise HTTPException(404, f"Customer {customer_id} not found")
    summary = aging_summary(db.list_invoices(customer["id"]), settings.as_of_date)
    return {"customer_id": customer["id"], "customer_name": customer["name"], **summary}


@app.get("/invoices/overdue")
def overdue_invoices(min_days: int = 1, max_days: int | None = None) -> dict:
    names = db.customer_names()
    rows = []
    for inv in db.list_invoices():
        e = enrich_invoice(inv, settings.as_of_date)
        if not e["is_overdue"] or e["days_overdue"] < min_days:
            continue
        if max_days is not None and e["days_overdue"] > max_days:
            continue
        rows.append({k: e[k] for k in ("invoice_number", "customer_id", "due_date", "amount",
                                        "outstanding", "days_overdue", "aging_bucket", "status")}
                    | {"customer_name": names[e["customer_id"]]})
    rows.sort(key=lambda r: -r["days_overdue"])
    return {"as_of_date": settings.as_of_date.isoformat(), "min_days": min_days,
            "max_days": max_days, "count": len(rows), "invoices": rows}


@app.get("/invoices/{invoice_number}")
def get_invoice(invoice_number: str) -> dict:
    inv = db.get_invoice(invoice_number)
    if not inv:
        raise HTTPException(404, f"Invoice {invoice_number} not found")
    return enrich_invoice(inv, settings.as_of_date)
