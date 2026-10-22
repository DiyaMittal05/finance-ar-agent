"""Programmatic checks: grounding of every cited figure against the ERP, plus case expectations."""
from __future__ import annotations

import json
import re
from typing import Any

from app.config import settings
from app.erp import db
from app.finance.analytics import aging_summary

MONEY = re.compile(r"(?:\$|USD\s?)\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
INVOICE = re.compile(r"INV-\d{4}-\d{4}", re.I)
DAYS = re.compile(r"\b(\d{1,3})\s+days?\b", re.I)
PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s?%")
# Day counts that come from policy text rather than invoice data.
POLICY_DAYS = {1, 5, 10, 15, 30, 31, 45, 60, 61, 90}


def reply_text(result: dict) -> str:
    email = result.get("email") or {}
    return "\n".join([result.get("answer") or "", email.get("subject") or "", email.get("body") or ""])


def _cents(value: str) -> int:
    return int(round(float(value.replace(",", "")) * 100))


def _numbers_in(obj: Any, out: set[int]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(int(round(obj * 100)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)


def supported_facts(result: dict, case: dict) -> dict[str, set]:
    """Everything a reply may cite: values in retrieved evidence, plus ERP aggregates for the
    customers involved (so a correct sum the model computed itself is not a hallucination)."""
    evidence = result.get("evidence") or []
    evidence_text = json.dumps(evidence, default=str)
    money: set[int] = set()
    _numbers_in([e.get("result") for e in evidence], money)
    customers = set(case.get("customer_ids", [])) | set(re.findall(r'"(C\d{3})"', evidence_text))
    days: set[int] = set(POLICY_DAYS)
    for cid in customers:
        s = aging_summary(db.list_invoices(cid), settings.as_of_date)
        _numbers_in({k: v for k, v in s.items() if k != "invoices"}, money)
        for inv in s["invoices"]:
            _numbers_in([inv["amount"], inv["paid_amount"], inv["outstanding"]], money)
            if inv["days_overdue"]:
                days.add(inv["days_overdue"])
        customer = db.get_customer(cid)
        if customer:
            money.add(int(round(customer["credit_limit"] * 100)))
            days.add(customer["terms_days"])
    for e in evidence:
        if isinstance(e.get("result"), dict):
            for inv in e["result"].get("invoices", []) or []:
                if isinstance(inv, dict) and inv.get("days_overdue"):
                    days.add(inv["days_overdue"])
    percents = {p for p in PERCENT.findall(evidence_text)}
    return {"money": money, "days": days, "invoices": {m.upper() for m in INVOICE.findall(evidence_text)},
            "percents": percents}


def unsupported_claims(result: dict, case: dict) -> list[str]:
    text = reply_text(result)
    facts = supported_facts(result, case)
    bad = []
    for raw in MONEY.findall(text):
        c = _cents(raw)
        has_cents = "." in raw
        ok = c in facts["money"] or (not has_cents and any(abs(c - v) < 100 for v in facts["money"]))
        if not ok:
            bad.append(f"${raw}")
    for inv in {m.upper() for m in INVOICE.findall(text)}:
        if inv not in facts["invoices"]:
            bad.append(inv)
    for d in DAYS.findall(text):
        if int(d) not in facts["days"]:
            bad.append(f"{d} days")
    for p in PERCENT.findall(text):
        if p not in facts["percents"]:
            bad.append(f"{p}%")
    return sorted(set(bad))


def status_ok(result: dict, case: dict) -> bool:
    expected = case["expected_status"]
    expected = expected if isinstance(expected, list) else [expected]
    return result.get("status") in expected


def run_checks(result: dict, case: dict) -> dict:
    text = reply_text(result)
    low = text.lower()
    checks: dict[str, Any] = {"status_ok": status_ok(result, case)}
    if result.get("status") != "answered":
        checks["unsupported_claims"] = []
        checks["passed"] = checks["status_ok"]
        return checks

    mentioned = {m.upper() for m in INVOICE.findall(text)}
    gold = case["must_mention_invoices"]
    checks["invoice_recall"] = (sum(i in mentioned for i in gold) / len(gold)) if gold else None
    checks["forbidden_found"] = [s for s in case["must_not_mention"] if s.lower() in low]
    checks["missing_keywords"] = [g for g in case["must_mention_any"] if not any(k.lower() in low for k in g)]
    cited = {_cents(m) for m in MONEY.findall(text)}
    checks["missing_amounts"] = [a for a in case["must_mention_amounts"] if int(round(a * 100)) not in cited]
    checks["email_ok"] = (case["expects_email"] is None
                          or bool(result.get("email")) == case["expects_email"])
    checks["amounts_when_forbidden"] = bool(case.get("no_amounts") and cited)
    checks["unsupported_claims"] = unsupported_claims(result, case)
    checks["passed"] = (checks["status_ok"]
                        and checks["invoice_recall"] in (None, 1.0)
                        and not checks["forbidden_found"]
                        and not checks["missing_keywords"]
                        and not checks["missing_amounts"]
                        and checks["email_ok"]
                        and not checks["amounts_when_forbidden"]
                        and not checks["unsupported_claims"])
    return checks
