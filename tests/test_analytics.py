from datetime import date

from app.erp import db
from app.finance.analytics import aging_summary, bucket_for, dunning_stage

AS_OF = date(2026, 9, 30)


def summary(cid):
    return aging_summary(db.list_invoices(cid), AS_OF)


def test_buckets_and_stages():
    assert [bucket_for(d) for d in (None, 0, 1, 30, 31, 60, 61, 90, 91)] == [
        "current", "current", "1-30", "1-30", "31-60", "31-60", "61-90", "61-90", "90+"]
    assert [dunning_stage(d) for d in (0, 20, 45, 75, 128)] == [None, 1, 2, 3, 4]


def test_acme_corporation_aging():
    s = summary("C001")
    assert s["total_outstanding"] == 30725.50
    assert s["total_overdue"] == 24625.50
    assert s["outstanding_by_bucket"]["61-90"] == 3975.50
    assert s["recommended_dunning_stage"] == 3


def test_partial_payment_outstanding():
    inv = {i["invoice_number"]: i for i in summary("C005")["invoices"]}["INV-2026-0501"]
    assert inv["outstanding"] == 7500.00 and "partial_payment" in inv["flags"]


def test_duplicates_disputes_missing_due_date():
    assert {"type": "possible_duplicate", "invoices": ["INV-2026-0301", "INV-2026-0302"],
            "detail": "Same customer, amount, issue date and description"} in summary("C003")["anomalies"]
    initech = summary("C004")
    disputed = {i["invoice_number"]: i for i in initech["invoices"]}["INV-2026-0401"]
    assert disputed["dunning_stage"] is None
    assert initech["max_days_overdue_undisputed"] == 15
    stark = {i["invoice_number"]: i for i in summary("C006")["invoices"]}["INV-2026-0601"]
    assert stark["days_overdue"] is None and "missing_due_date" in stark["flags"]


def test_fully_paid_customer_has_nothing_overdue():
    s = summary("C007")
    assert s["total_outstanding"] == 0 and s["recommended_dunning_stage"] is None
