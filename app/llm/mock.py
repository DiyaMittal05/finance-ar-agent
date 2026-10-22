"""Offline, rule-based stand-in for an LLM.

It lets the whole pipeline (API, agents, tools, eval harness, CI) run without
API keys. It routes on the "ROLE: <name>" line of each system prompt and uses
keyword rules, so its eval scores test the plumbing, not model quality. Never
report mock numbers as model results.
"""
from __future__ import annotations

import json
import re

from app.llm.base import LLM

FINANCE_WORDS = re.compile(
    r"invoice|overdue|owe|outstanding|balance|payment|paid|pay\b|aging|ageing|collection|dunning|"
    r"remind|follow[- ]?up|receivable|\bar\b|late fee|terms|credit|dispute|customer|account|statement|"
    r"duplicate|unusual|anomal", re.I)
GUARDRAIL_BLOCKS = {
    "write_action": r"\b(mark|set)\b.*\bpaid\b|\bdelete\b|\bwrite[- ]?off\b|\bissue (a )?credit|\bvoid\b|\bupdate the\b",
    "advice": r"\bstock\b|\binvest|\bshares?\b|\btax (advice|return)|\bcrypto",
    "privacy": r"home address|personal (phone|number|email)|social security|date of birth",
    "policy_violation": r"threaten|legal action|lawsuit|sue\b|penalty of|\d+% (penalty|surcharge|fee)",
    "injection": r"ignore (all |your |the |previous )*(instructions|rules)|system prompt|you are now",
}
NAME_STOP = {"which", "what", "draft", "write", "give", "show", "list", "tell", "please", "how", "does",
             "do", "is", "are", "can", "the", "our", "we", "i", "a", "an", "any", "anything", "summarize",
             "summarise", "send", "prepare", "create", "find", "check", "compose", "explain", "who"}


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^=== (.+?) ===$", text, flags=re.M)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _customer_phrase(request: str) -> str | None:
    m = re.search(r"\b(?:from|for|to|about|with|on|of)\s+((?:[A-Z][\w&.'-]*\s?)+)", request)
    if m:
        phrase = re.sub(r"'s$", "", m.group(1).strip())
        words = [w for w in phrase.split() if w.lower() not in NAME_STOP]
        if words:
            return " ".join(words)
    caps = [w.strip("?,.'s") for w in re.findall(r"\b[A-Z][a-zA-Z&]+(?:'s)?", request)
            if w.lower().strip("?,.'s") not in NAME_STOP]
    return caps[0] if caps else None


class MockLLM(LLM):
    provider = "mock"

    def _complete(self, system, messages, max_tokens, json_mode):
        role = re.search(r"ROLE: (\w+)", system).group(1)
        out = getattr(self, f"_{role}")(system, messages)
        text = json.dumps(out)
        tokens_in = (len(system) + sum(len(m["content"]) for m in messages)) // 4
        return text, tokens_in, len(text) // 4

    # -- roles ------------------------------------------------------------------

    def _guardrail(self, system, messages):
        request = _sections(messages[0]["content"])["REQUEST"]
        for category, pattern in GUARDRAIL_BLOCKS.items():
            if re.search(pattern, request, re.I):
                return {"in_scope": False, "category": category, "reason": f"Matched {category} rule."}
        if not FINANCE_WORDS.search(request):
            return {"in_scope": False, "category": "off_topic", "reason": "Not an AR request."}
        return {"in_scope": True, "category": "ar_query", "reason": "AR request."}

    def _planner(self, system, messages):
        request = _sections(messages[0]["content"])["REQUEST"]
        if not FINANCE_WORDS.search(request):
            return {"in_scope": False, "refusal_reason": "Not an AR request.", "deliverable": "answer",
                    "steps": []}
        if re.search(r"draft|email|write|follow[- ]?up|remind|message|what should we say", request, re.I):
            deliverable = "email"
        elif re.search(r"report|summary|summarise|summarize", request, re.I):
            deliverable = "report"
        else:
            deliverable = "answer"
        return {"in_scope": True, "refusal_reason": None, "deliverable": deliverable, "steps": [
            {"agent": "research", "instruction": "Gather customer, invoice and policy data for the request."},
            {"agent": "finance", "instruction": "Analyse balances, aging and anomalies."},
            {"agent": "editor", "instruction": f"Write the {deliverable}."},
        ]}

    def _research(self, system, messages):
        request = _sections(messages[0]["content"])["REQUEST"]
        results = []
        for m in messages[1:]:
            if m["role"] == "user":
                s = _sections(m["content"])
                for key, body in s.items():
                    if key.startswith("TOOL_RESULT"):
                        results.append((key.split()[1], json.loads(body)))
        called = [name for name, _ in results]
        has_aging = "get_aging_summary" in system

        def finish(ids=(), ambiguous=False, candidates=(), notes=""):
            return {"action": "finish", "customer_ids": list(ids), "ambiguous": ambiguous,
                    "candidates": list(candidates), "notes": notes}

        inv = re.search(r"INV-\d{4}-\d{4}", request, re.I)
        if inv:
            if "get_invoice" not in called:
                return {"action": "tool", "tool": "get_invoice", "args": {"invoice_number": inv.group(0).upper()}}
            data = dict(results)["get_invoice"]
            if "error" in data:
                return finish(notes=data["error"])
            if "get_customer" not in called:
                return {"action": "tool", "tool": "get_customer", "args": {"customer_id": data["customer_id"]}}
            return finish([data["customer_id"]])

        phrase = _customer_phrase(request)
        policy_only = re.search(r"\b(policy|policies|terms|late fee|stage)\b", request, re.I) and not phrase
        if policy_only or (phrase and not re.search(r"[A-Z]", phrase)):
            phrase = None
        if phrase is None:
            if policy_only:
                if "search_policies" not in called:
                    return {"action": "tool", "tool": "search_policies", "args": {"query": request}}
                return finish()
            if "list_overdue_invoices" not in called:
                days = re.search(r"(\d+)\s*days", request)
                min_days = int(days.group(1)) + 1 if days and "more than" in request else (
                    int(days.group(1)) if days else 1)
                return {"action": "tool", "tool": "list_overdue_invoices", "args": {"min_days": min_days}}
            if "search_policies" not in called:
                return {"action": "tool", "tool": "search_policies", "args": {"query": "collections stages"}}
            return finish()

        if "search_customers" not in called:
            return {"action": "tool", "tool": "search_customers", "args": {"query": phrase}}
        search = dict(results)["search_customers"]
        matches = search.get("matches", [])
        if not matches:
            return finish(notes=f"No customer matches {phrase!r}.")
        if search.get("ambiguous"):
            return finish([m["id"] for m in matches], True, [m["name"] for m in matches])
        cid = matches[0]["id"]
        detail_tool = "get_aging_summary" if has_aging else "list_invoices"
        for tool in ("get_customer", detail_tool):
            if tool not in called:
                return {"action": "tool", "tool": tool, "args": {"customer_id": cid}}
        if "search_policies" not in called:
            return {"action": "tool", "tool": "search_policies",
                    "args": {"query": f"{matches[0]['name']} contract {request}"}}
        return finish([cid])

    def _finance(self, system, messages):
        s = _sections(messages[0]["content"])
        analytics = json.loads(s["ANALYTICS"]) if s.get("ANALYTICS", "not provided") != "not provided" else None
        facts, anomalies, action = [], [], "No action needed."
        if analytics:
            for a in analytics:
                if "error" in a:
                    continue
                facts.append(f"{a['customer_name']}: total outstanding {_money(a['total_outstanding'])}, "
                             f"overdue {_money(a['total_overdue'])} across {a['overdue_invoice_count']} invoices.")
                for inv in a["invoices"]:
                    if inv["is_overdue"]:
                        facts.append(f"{inv['invoice_number']}: {_money(inv['outstanding'])} outstanding, "
                                     f"{inv['days_overdue']} days overdue.")
                anomalies += [f"{x['type']}: {', '.join(x['invoices'])}" for x in a["anomalies"]]
                stage = a["recommended_dunning_stage"]
                action = {None: "No overdue undisputed invoices.", 1: "Send a friendly reminder.",
                          2: "Send a firm reminder.", 3: "Send a final notice.",
                          4: "Escalate to the Collections Manager."}[stage]
        else:
            for e in json.loads(s["RESEARCH DATA"]):
                r = e["result"]
                for inv in r.get("invoices", []) if isinstance(r, dict) else []:
                    out = inv.get("outstanding", round(inv["amount"] - inv.get("paid_amount", 0), 2))
                    if out > 0:
                        facts.append(f"{inv['invoice_number']}: {_money(out)} outstanding, due {inv['due_date']}.")
                if e["tool"] == "get_invoice" and "invoice_number" in r:
                    facts.append(f"{r['invoice_number']}: status {r['status']}, "
                                 f"{_money(r['outstanding'])} outstanding.")
        return {"summary": " ".join(facts[:1]) or "No matching data found.", "key_facts": facts,
                "anomalies": anomalies, "recommended_action": action}

    def _editor(self, system, messages):
        s = _sections(messages[0]["content"])
        fin = json.loads(s["FINANCE ANALYSIS"])
        policy = json.loads(s["POLICY"]) if s["POLICY"].startswith("[") else []
        customers = json.loads(s["CUSTOMERS"])
        lines = fin["key_facts"] or ["I could not find matching records for this request."]
        if fin["anomalies"]:
            lines.append("Anomalies: " + "; ".join(fin["anomalies"]))
        if not fin["key_facts"] and policy:
            lines = [policy[0]["text"].split("\n", 1)[-1][:400]]
        answer = "\n".join(lines + [f"Recommended action: {fin['recommended_action']}"])
        email = None
        if s["DELIVERABLE"] == "email" and customers and fin["key_facts"]:
            c = customers[0]
            email = {"to": c["contact_email"], "subject": f"Outstanding invoices — {c['name']}",
                     "body": f"Dear {c['contact_name']},\n\nOur records show the following:\n"
                             + "\n".join(f"- {f}" for f in fin["key_facts"][1:])
                             + "\n\nPlease let us know when we can expect payment.\n\nKind regards,\nAR Team"}
        return {"answer": answer, "email": email, "citations": [p["chunk_id"] for p in policy[:2]]}

    def _judge(self, system, messages):
        # No real judgement offline: the harness treats a skipped judge as "not scored".
        return {"skipped": True, "rationale": "Mock judge does not score."}
