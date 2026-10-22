"""Run the eval suite against one or more (provider, model, prompt version) configurations.

    python -m evals.run --provider mock --versions v1 v2 v3
    python -m evals.run --provider anthropic --versions v3 --judge-provider gemini
    python -m evals.run --provider openai --model gpt-4.1-mini --limit 10 --workers 4

Each configuration writes evals/results/<provider>_<model>_<version>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.agents.orchestrator import Orchestrator, prompt_versions
from app.config import settings
from app.llm.base import cost_usd
from app.llm.registry import get_llm
from evals.checks import run_checks
from evals.judge import judge

CASES = Path(__file__).parent / "cases.jsonl"
RESULTS = Path(__file__).parent / "results"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)


def evaluate_case(orch: Orchestrator, judge_llm, case: dict) -> dict:
    result = orch.run(case["query"])
    checks = run_checks(result, case)
    verdict = None
    if judge_llm is not None and result["status"] == "answered":
        verdict = judge(judge_llm, case, result)
    judged = verdict is not None and not verdict.get("skipped") and verdict.get("verdict") in ("pass", "fail")
    judge_unsupported = (verdict or {}).get("unsupported_claims") or [] if judged else []
    success = checks["passed"] and (not judged or verdict["verdict"] == "pass")
    return {
        "id": case["id"], "category": case["category"], "query": case["query"],
        "expected_status": case["expected_status"], "status": result["status"],
        "success": success, "checks": checks, "judge": verdict,
        "hallucinated": bool(checks["unsupported_claims"] or judge_unsupported),
        "latency_s": result["latency_s"], "usage": result["usage"],
        "answer": result["answer"], "email": result["email"], "citations": result["citations"],
        "trace": result["trace"], "evidence_tools": [e["tool"] for e in result["evidence"]],
    }


def summarize(rows: list[dict], cases: list[dict]) -> dict:
    by_id = {c["id"]: c for c in cases}
    n = len(rows)
    answered = [r for r in rows if r["status"] == "answered"]
    must_refuse = [r for r in rows if by_id[r["id"]]["expected_status"] == "refused"]
    must_not_refuse = [r for r in rows if "refused" not in (
        by_id[r["id"]]["expected_status"] if isinstance(by_id[r["id"]]["expected_status"], list)
        else [by_id[r["id"]]["expected_status"]])]
    refusal_correct = sum(r["status"] == "refused" for r in must_refuse) + sum(
        r["status"] != "refused" for r in must_not_refuse)
    refusal_total = len(must_refuse) + len(must_not_refuse)
    costs = [r["usage"]["cost_usd"] for r in rows]
    judged = [r["judge"] for r in rows if r["judge"] and r["judge"].get("verdict") in ("pass", "fail")]
    categories = sorted({r["category"] for r in rows})

    def mean(xs):
        return round(statistics.mean(xs), 3) if xs else None

    return {
        "cases": n,
        "task_success": mean([float(r["success"]) for r in rows]),
        "hallucination_rate": mean([float(r["hallucinated"]) for r in answered]),
        "unsupported_claims_total": sum(len(r["checks"]["unsupported_claims"]) for r in rows),
        "refusal_accuracy": round(refusal_correct / refusal_total, 3) if refusal_total else None,
        "refusal_recall": mean([float(r["status"] == "refused") for r in must_refuse]),
        "false_refusal_rate": mean([float(r["status"] == "refused") for r in must_not_refuse]),
        "clarify_accuracy": mean([float(r["status"] == "clarify") for r in rows
                                  if by_id[r["id"]]["expected_status"] == "clarify"]),
        "errors": sum(r["status"] == "error" for r in rows),
        "latency_p50_s": percentile([r["latency_s"] for r in rows], 0.5),
        "latency_p95_s": percentile([r["latency_s"] for r in rows], 0.95),
        "tokens_per_task": mean([r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in rows]),
        "cost_per_task_usd": round(statistics.mean(costs), 5) if costs and None not in costs else None,
        "judge_scores": {k: mean([j[k] for j in judged if isinstance(j.get(k), (int, float))])
                         for k in ("correctness", "grounding", "policy_compliance", "communication")}
        if judged else None,
        "by_category": {c: mean([float(r["success"]) for r in rows if r["category"] == c]) for c in categories},
    }


def run_config(provider: str, model: str | None, version: str, judge_provider: str | None,
               judge_model: str | None, cases: list[dict], workers: int) -> dict:
    llm = get_llm(provider, model)
    judge_llm = get_llm(judge_provider, judge_model) if judge_provider else None
    orch = Orchestrator(llm, version)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda c: evaluate_case(orch, judge_llm, c), cases))
    summary = summarize(rows, cases)
    judge_in = sum((r["judge"] or {}).get("_usage", {}).get("input_tokens", 0) for r in rows)
    judge_out = sum((r["judge"] or {}).get("_usage", {}).get("output_tokens", 0) for r in rows)
    report = {
        "provider": provider, "model": llm.model, "prompt_version": version,
        "judge": f"{judge_llm.provider}/{judge_llm.model}" if judge_llm else None,
        "judge_cost_usd": cost_usd(judge_llm.model, judge_in, judge_out) if judge_llm else None,
        "as_of_date": settings.as_of_date.isoformat(), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_s": round(time.time() - started, 1),
        "summary": summary, "results": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{provider}_{llm.model}_{version}.json".replace("/", "-")
    out.write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default=settings.llm_provider)
    p.add_argument("--model", default=settings.llm_model or None)
    p.add_argument("--versions", nargs="+", default=prompt_versions())
    p.add_argument("--judge-provider", default=settings.judge_provider,
                   help="Provider for LLM-as-judge; 'none' to skip judging")
    p.add_argument("--judge-model", default=settings.judge_model or None)
    p.add_argument("--limit", type=int, default=None, help="Run only the first N cases")
    p.add_argument("--only", nargs="+", help="Run only these case ids")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    if args.limit:
        cases = cases[: args.limit]
    judge_provider = None if args.judge_provider == "none" else args.judge_provider
    for version in args.versions:
        report = run_config(args.provider, args.model, version, judge_provider, args.judge_model,
                            cases, args.workers)
        s = report["summary"]
        print(f"\n{report['provider']}/{report['model']} {version}: {s['cases']} cases")
        for key in ("task_success", "hallucination_rate", "refusal_accuracy", "clarify_accuracy",
                    "errors", "latency_p50_s", "latency_p95_s", "tokens_per_task", "cost_per_task_usd"):
            print(f"  {key:22} {s[key]}")
        print(f"  {'by_category':22} {s['by_category']}")
        failed = [r["id"] for r in report["results"] if not r["success"]]
        print(f"  failed: {', '.join(failed) or 'none'}")
    if report["provider"] == "mock":
        print("\nNote: mock runs test the pipeline only. They are not model results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
