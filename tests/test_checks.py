import json
from pathlib import Path

from app.agents.tools import ToolBox
from evals.checks import run_checks

CASES = {c["id"]: c for c in map(json.loads, Path("evals/cases.jsonl").read_text().splitlines())}


def evidence_for(cid):
    tools = ToolBox(include_aging=True)
    return [{"tool": t, "args": {"customer_id": cid}, "result": tools.call(t, {"customer_id": cid})}
            for t in ("get_customer", "get_aging_summary")]


def result(answer, email=None, status="answered", cid="C001"):
    return {"status": status, "answer": answer, "email": email, "evidence": evidence_for(cid)}


GOOD_ACME_EMAIL = {
    "to": "dana.whitfield@acmecorporation.example",
    "subject": "Final notice: overdue invoices",
    "body": ("Dear Dana Whitfield,\n\nThis is a final notice regarding INV-2026-0103 (PO-77420, $3,975.50, "
             "75 days overdue), INV-2026-0102 (PO-77654, $12,400.00) and INV-2026-0101 (PO-77812, $8,250.00). "
             "The total overdue is $24,625.50. Please arrange payment within 10 business days to avoid a "
             "credit hold on the account.\n\nKind regards,\nAR Team"),
}


def test_correct_reply_passes():
    checks = run_checks(result("Drafted a stage 3 final notice.", GOOD_ACME_EMAIL), CASES["normal-03"])
    assert checks["passed"], checks


def test_invented_amount_is_flagged():
    email = {**GOOD_ACME_EMAIL, "body": GOOD_ACME_EMAIL["body"] + " A $500.00 penalty now applies."}
    checks = run_checks(result("Drafted.", email), CASES["normal-03"])
    assert "$500.00" in checks["unsupported_claims"] and not checks["passed"]


def test_rounded_total_is_accepted_but_wrong_total_is_not():
    ok = run_checks(result("Acme Corporation owes about $24,626 overdue."), CASES["normal-01"])
    bad = run_checks(result("Acme Corporation owes $26,000.00 overdue."), CASES["normal-01"])
    assert ok["unsupported_claims"] == [] and bad["unsupported_claims"] == ["$26,000.00"]


def test_missing_invoice_lowers_recall():
    checks = run_checks(result("Overdue: INV-2026-0101 and INV-2026-0102."), CASES["normal-01"])
    assert round(checks["invoice_recall"], 2) == 0.67 and not checks["passed"]


def test_refusal_and_clarify_cases():
    assert run_checks({"status": "refused", "answer": "No."}, CASES["oos-01"])["passed"]
    assert not run_checks({"status": "answered", "answer": "Here is a poem"}, CASES["oos-01"])["passed"]
    assert run_checks({"status": "clarify", "answer": "Which Acme?"}, CASES["ambiguous-01"])["passed"]
