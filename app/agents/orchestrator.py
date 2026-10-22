"""Plan-and-execute orchestration of the guardrail, planner, research, finance and editor agents."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.agents.tools import ToolBox
from app.config import settings
from app.llm.base import LLM, cost_usd, parse_json

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_RESEARCH_STEPS = 8
AUTO_ANALYTICS = "aging_summary (computed by orchestrator)"


@lru_cache(maxsize=None)
def load_prompts(version: str) -> dict:
    return yaml.safe_load((PROMPTS_DIR / f"{version}.yaml").read_text())


def prompt_versions() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("v*.yaml"))


def section(name: str, body: Any) -> str:
    if not isinstance(body, str):
        body = json.dumps(body, indent=1, default=str)
    return f"=== {name} ===\n{body}"


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0


class Orchestrator:
    def __init__(self, llm: LLM, version: str | None = None):
        self.llm = llm
        self.version = version or settings.prompt_version
        spec = load_prompts(self.version)
        self.features = spec["features"]
        self.prompts = spec["prompts"]

    # -- LLM plumbing ---------------------------------------------------------

    def _system(self, role: str, **extra: str) -> str:
        text = self.prompts[role].replace("{as_of_date}", settings.as_of_date.isoformat())
        for key, value in extra.items():
            text = text.replace("{" + key + "}", value)
        return text

    def _ask(self, trace: Trace, agent: str, system: str, messages: list[dict]) -> dict:
        """Call the model and parse JSON, with one repair retry on malformed output."""
        for attempt in range(2):
            res = self.llm.complete(system, messages)
            trace.llm_calls += 1
            trace.input_tokens += res.input_tokens
            trace.output_tokens += res.output_tokens
            try:
                out = parse_json(res.text)
            except ValueError:
                out = None
            trace.steps.append({"agent": agent, "latency_s": round(res.latency_s, 3),
                                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                                "output": out if out is not None else res.text[:500]})
            if out is not None:
                return out
            messages = [*messages, {"role": "assistant", "content": res.text},
                        {"role": "user", "content": "That was not valid JSON. Reply with one JSON object only."}]
        raise ValueError(f"{agent} agent did not return JSON")

    # -- Agents -----------------------------------------------------------------

    def guardrail(self, trace: Trace, query: str) -> dict:
        return self._ask(trace, "guardrail", self._system("guardrail"),
                         [{"role": "user", "content": section("REQUEST", query)}])

    def plan(self, trace: Trace, query: str) -> dict:
        return self._ask(trace, "planner", self._system("planner"),
                         [{"role": "user", "content": section("REQUEST", query)}])

    def research(self, trace: Trace, query: str, instruction: str, tools: ToolBox,
                 evidence: list[dict]) -> dict:
        system = self._system("research", tools=tools.describe())
        messages = [{"role": "user", "content": section("REQUEST", query) + "\n\n"
                     + section("INSTRUCTION", instruction)}]
        for _ in range(MAX_RESEARCH_STEPS):
            step = self._ask(trace, "research", system, messages)
            messages.append({"role": "assistant", "content": json.dumps(step)})
            if step.get("action") == "finish":
                return step
            name, args = step.get("tool", ""), step.get("args") or {}
            result = tools.call(name, args)
            evidence.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "user", "content": section(f"TOOL_RESULT {name}", result)})
        return {"action": "finish", "customer_ids": [], "ambiguous": False, "candidates": [],
                "notes": "Stopped after the maximum number of research steps."}

    def finance(self, trace: Trace, query: str, instruction: str, evidence: list[dict],
                analytics: list[dict] | None) -> dict:
        content = "\n\n".join([
            section("REQUEST", query), section("INSTRUCTION", instruction),
            section("RESEARCH DATA", [e for e in evidence if e["tool"] != AUTO_ANALYTICS]),
            section("ANALYTICS", analytics if analytics is not None else "not provided"),
        ])
        return self._ask(trace, "finance", self._system("finance"), [{"role": "user", "content": content}])

    def editor(self, trace: Trace, query: str, deliverable: str, instruction: str, finance: dict,
               evidence: list[dict], analytics: list[dict] | None) -> dict:
        policy = [c for e in evidence if e["tool"] == "search_policies"
                  for c in e["result"].get("chunks", [])]
        facts = [e for e in evidence if e["tool"] not in ("search_policies", AUTO_ANALYTICS)]
        customers = [e["result"] for e in evidence
                     if e["tool"] == "get_customer" and "error" not in e["result"]]
        content = "\n\n".join([
            section("REQUEST", query), section("DELIVERABLE", deliverable),
            section("INSTRUCTION", instruction), section("FINANCE ANALYSIS", finance),
            section("FACTS", facts),
            section("ANALYTICS", analytics if analytics is not None else "not provided"),
            section("POLICY", policy or "no policy excerpts retrieved"),
            section("CUSTOMERS", customers),
        ])
        return self._ask(trace, "editor", self._system("editor"), [{"role": "user", "content": content}])

    # -- Orchestration ----------------------------------------------------------

    def run(self, query: str) -> dict:
        start = time.perf_counter()
        trace = Trace()
        evidence: list[dict] = []
        result: dict[str, Any] = {"status": "answered", "answer": "", "email": None, "citations": [],
                                  "customer_ids": [], "plan": None}
        try:
            self._run(query, trace, evidence, result)
        except Exception as exc:  # surfaced in the eval report, not swallowed
            result.update(status="error", answer=f"{type(exc).__name__}: {exc}")
        cost = cost_usd(self.llm.model, trace.input_tokens, trace.output_tokens)
        result.update(
            evidence=evidence, trace=trace.steps, prompt_version=self.version,
            provider=self.llm.provider, model=self.llm.model,
            latency_s=round(time.perf_counter() - start, 3),
            usage={"input_tokens": trace.input_tokens, "output_tokens": trace.output_tokens,
                   "llm_calls": trace.llm_calls, "cost_usd": cost},
        )
        return result

    def _run(self, query: str, trace: Trace, evidence: list[dict], result: dict) -> None:
        if self.features.get("guardrail_classifier"):
            verdict = self.guardrail(trace, query)
            if not verdict.get("in_scope", False):
                result.update(status="refused", answer=refusal(verdict.get("reason")))
                return

        plan = self.plan(trace, query)
        result["plan"] = plan
        if not plan.get("in_scope", True):
            result.update(status="refused", answer=refusal(plan.get("refusal_reason")))
            return
        steps = plan.get("steps") or []
        instructions = {s.get("agent"): s.get("instruction", "") for s in steps}
        deliverable = plan.get("deliverable", "answer")

        tools = ToolBox(include_aging=self.features.get("deterministic_analytics", False))
        found = self.research(trace, query, instructions.get("research", query), tools, evidence)
        customer_ids = valid_customer_ids(found.get("customer_ids") or [], evidence)
        result["customer_ids"] = customer_ids

        if self.features.get("clarify_on_ambiguity"):
            candidates = ambiguous_candidates(found, evidence)
            if candidates:
                names = ", ".join(f"{c['name']} ({c['id']})" for c in candidates)
                result.update(status="clarify", customer_ids=[],
                              answer=f"More than one customer matches your request: {names}. "
                                     "Which account do you mean?")
                return

        analytics = None
        if self.features.get("deterministic_analytics"):
            analytics = [tools.call("get_aging_summary", {"customer_id": cid}) for cid in customer_ids[:3]]
            # Recorded as evidence so grounding checks see everything the agents were shown.
            evidence.extend({"tool": AUTO_ANALYTICS, "args": {"customer_id": cid}, "result": a}
                            for cid, a in zip(customer_ids, analytics))

        fin = self.finance(trace, query, instructions.get("finance", ""), evidence, analytics)
        out = self.editor(trace, query, deliverable, instructions.get("editor", ""), fin, evidence, analytics)
        email = out.get("email") if isinstance(out.get("email"), dict) else None
        result.update(answer=out.get("answer", ""), email=email, citations=out.get("citations") or [],
                      finance=fin)


def refusal(reason: str | None) -> str:
    base = ("I can only help with accounts receivable work: invoices, payments, balances, aging, "
            "collections policy and AR follow-up emails, and I can't change records.")
    return f"{base} {reason}" if reason else base


def valid_customer_ids(ids: list[str], evidence: list[dict]) -> list[str]:
    """Keep only customer IDs that actually appeared in tool results (no invented IDs)."""
    seen = json.dumps([e["result"] for e in evidence], default=str)
    return [cid for cid in dict.fromkeys(str(i).upper() for i in ids) if f'"{cid}"' in seen]


def ambiguous_candidates(found: dict, evidence: list[dict]) -> list[dict]:
    """Return candidate customers when the customer reference is ambiguous.

    Trusts the research agent's flag, and also checks the evidence directly: if every
    customer search came back ambiguous, the agent cannot have resolved it from data.
    """
    searches = [e["result"] for e in evidence if e["tool"] == "search_customers"
                and isinstance(e["result"], dict) and e["result"].get("matches")]
    if not searches:
        return []
    if found.get("ambiguous") or all(s.get("ambiguous") for s in searches):
        last = next((s for s in reversed(searches) if s.get("ambiguous")), None)
        if last:
            top = last["matches"][0]["match_score"]
            return [m for m in last["matches"] if m["match_score"] >= top - 0.05]
        return [{"id": "?", "name": str(n)} for n in found.get("candidates") or []]
    return []
