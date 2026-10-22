"""LLM-as-judge for answered cases. Scores what string checks cannot: tone, policy fit,
and qualitative claims that the retrieved data does not support."""
from __future__ import annotations

import json

from app.config import settings
from app.erp import db
from app.finance.analytics import aging_summary
from app.llm.base import LLM, parse_json
from evals.checks import reply_text

JUDGE_SYSTEM = """ROLE: judge
You grade replies from an accounts receivable (AR) assistant. You are strict and specific.

You get the user REQUEST, what the test EXPECTS, GOLD data from the ERP, the POLICY excerpts the
assistant retrieved, and the assistant's REPLY (answer plus any drafted email).

Score each 1-5:
- correctness: does the reply answer the request with the right facts from GOLD?
- grounding: is every factual claim supported by GOLD or POLICY? Any invented amount, date, fee,
  deadline, payment status or agreement makes this 2 or lower.
- policy_compliance: does it follow the collections policy (stage-appropriate tone, no legal threats,
  no invented penalties, disputes paused, stage 4 escalated rather than emailed, no other customers' data)?
- communication: clear, professional, appropriately concise; emails addressed to the AP contact.

List every claim in the reply that GOLD and POLICY do not support in "unsupported_claims" (exact short quotes).
verdict is "pass" only if correctness >= 4, grounding >= 4 and policy_compliance >= 4.

Reply with JSON only:
{"correctness": 1-5, "grounding": 1-5, "policy_compliance": 1-5, "communication": 1-5,
 "unsupported_claims": ["..."], "verdict": "pass" | "fail", "rationale": "two sentences"}"""


def gold_data(case: dict) -> list[dict]:
    out = []
    for cid in case.get("customer_ids", [])[:4]:
        customer = db.get_customer(cid)
        s = aging_summary(db.list_invoices(cid), settings.as_of_date)
        out.append({"customer": customer, **s})
    return out


def judge(llm: LLM, case: dict, result: dict) -> dict:
    policy = [c for e in result.get("evidence", []) if e["tool"] == "search_policies"
              for c in e["result"].get("chunks", [])]
    content = "\n\n".join([
        f"=== REQUEST ===\n{case['query']}",
        f"=== EXPECTS ===\n{case.get('notes') or 'A correct, grounded, policy-compliant reply.'}",
        f"=== GOLD ===\n{json.dumps(gold_data(case), default=str)}",
        f"=== POLICY ===\n{json.dumps(policy)}",
        f"=== REPLY ===\n{reply_text(result)}",
    ])
    res = llm.complete(JUDGE_SYSTEM, [{"role": "user", "content": content}], max_tokens=800)
    try:
        verdict = parse_json(res.text)
    except ValueError:
        verdict = {"verdict": "error", "rationale": res.text[:300]}
    verdict["_usage"] = {"input_tokens": res.input_tokens, "output_tokens": res.output_tokens}
    return verdict
