"""Deterministic AR analytics: outstanding balances, aging buckets, anomalies.

Kept out of the LLM on purpose. Prompt v1 asked the model to do this
arithmetic itself; v2 onward hands it these computed facts instead.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

BUCKETS = ["current", "1-30", "31-60", "61-90", "90+"]


def bucket_for(days_overdue: int | None) -> str:
    if days_overdue is None or days_overdue <= 0:
        return "current"
    if days_overdue <= 30:
        return "1-30"
    if days_overdue <= 60:
        return "31-60"
    if days_overdue <= 90:
        return "61-90"
    return "90+"


def dunning_stage(days_overdue: int | None) -> int | None:
    """Collections policy stage (1-4) for a number of days overdue."""
    if days_overdue is None or days_overdue <= 0:
        return None
    if days_overdue <= 30:
        return 1
    if days_overdue <= 60:
        return 2
    if days_overdue <= 90:
        return 3
    return 4


def enrich_invoice(inv: dict[str, Any], as_of: date) -> dict[str, Any]:
    """Add outstanding, days_overdue, bucket and flags to an invoice dict.

    Expects amount and paid_amount in dollars and due_date as ISO string or None.
    """
    amount = float(inv["amount"])
    paid = float(inv.get("paid_amount", 0.0))
    outstanding = round(max(amount - paid, 0.0), 2)
    if inv.get("status") == "void":
        outstanding = 0.0
    due = inv.get("due_date")
    days_overdue: int | None = None
    if due and outstanding > 0:
        days_overdue = (as_of - date.fromisoformat(due)).days
    flags = []
    if not due and outstanding > 0:
        flags.append("missing_due_date")
    if inv.get("status") == "disputed":
        flags.append("disputed")
    if 0 < paid < amount:
        flags.append("partial_payment")
    return {
        **inv,
        "outstanding": outstanding,
        "days_overdue": days_overdue if days_overdue and days_overdue > 0 else 0 if due else None,
        "aging_bucket": bucket_for(days_overdue) if due else "unknown",
        "is_overdue": bool(days_overdue and days_overdue > 0 and outstanding > 0),
        "dunning_stage": None if inv.get("status") == "disputed" else dunning_stage(days_overdue),
        "flags": flags,
    }


def find_duplicates(invoices: list[dict[str, Any]]) -> list[list[str]]:
    """Group invoices for the same customer with identical amount, issue date and description."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for inv in invoices:
        key = (inv["customer_id"], inv["amount"], inv["issue_date"], inv.get("description"))
        groups[key].append(inv["invoice_number"])
    return [sorted(numbers) for numbers in groups.values() if len(numbers) > 1]


def aging_summary(invoices: list[dict[str, Any]], as_of: date) -> dict[str, Any]:
    """Summarise a set of raw invoices. Amounts in dollars, rounded to cents."""
    enriched = [enrich_invoice(inv, as_of) for inv in invoices]
    by_bucket = {b: 0.0 for b in BUCKETS + ["unknown"]}
    for inv in enriched:
        if inv["outstanding"] > 0:
            by_bucket[inv["aging_bucket"]] = round(by_bucket[inv["aging_bucket"]] + inv["outstanding"], 2)
    overdue = [i for i in enriched if i["is_overdue"]]
    actionable = [i for i in overdue if "disputed" not in i["flags"]]
    max_days = max((i["days_overdue"] for i in actionable), default=0)
    duplicates = find_duplicates(invoices)
    for group in duplicates:
        for inv in enriched:
            if inv["invoice_number"] in group:
                inv["flags"].append("possible_duplicate")
    anomalies = []
    for group in duplicates:
        anomalies.append({"type": "possible_duplicate", "invoices": group,
                          "detail": "Same customer, amount, issue date and description"})
    for inv in enriched:
        for flag in inv["flags"]:
            if flag in ("missing_due_date", "disputed", "partial_payment"):
                anomalies.append({"type": flag, "invoices": [inv["invoice_number"]],
                                  "detail": inv.get("dispute_reason") or ""})
        if inv["aging_bucket"] == "90+" and inv["outstanding"] > 0:
            anomalies.append({"type": "over_90_days", "invoices": [inv["invoice_number"]], "detail": ""})
    return {
        "as_of_date": as_of.isoformat(),
        "invoice_count": len(enriched),
        "total_outstanding": round(sum(i["outstanding"] for i in enriched), 2),
        "total_overdue": round(sum(i["outstanding"] for i in overdue), 2),
        "overdue_invoice_count": len(overdue),
        "outstanding_by_bucket": by_bucket,
        "max_days_overdue_undisputed": max_days,
        "recommended_dunning_stage": dunning_stage(max_days),
        "anomalies": anomalies,
        "invoices": enriched,
    }
