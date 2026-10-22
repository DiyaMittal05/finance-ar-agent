from fastapi.testclient import TestClient

from app.agents.orchestrator import Orchestrator
from app.llm.mock import MockLLM
from app.main import app


def run(query, version="v3"):
    return Orchestrator(MockLLM("mock"), version).run(query)


def test_v3_asks_for_clarification_on_ambiguous_customer():
    r = run("Which invoices from Acme are overdue and what should we say to them?")
    assert r["status"] == "clarify"
    assert "Acme Corporation" in r["answer"] and "Acme Industrial Supply" in r["answer"]


def test_v1_does_not_clarify():
    assert run("Which invoices from Acme are overdue?", "v1")["status"] == "answered"


def test_out_of_scope_and_write_actions_are_refused():
    assert run("Write me a poem about the ocean.")["status"] == "refused"
    assert run("Mark INV-2026-0101 as paid.")["status"] == "refused"


def test_answer_is_grounded_in_tool_evidence():
    r = run("Draft a follow-up email to Acme Corporation about their overdue invoices.")
    assert r["status"] == "answered" and r["customer_ids"] == ["C001"]
    tools = [e["tool"] for e in r["evidence"]]
    assert "search_customers" in tools and "search_policies" in tools
    assert r["email"]["to"].startswith("dana.whitfield")
    assert r["usage"]["llm_calls"] >= 4


def test_api_endpoints():
    client = TestClient(app)
    assert client.get("/erp/health").json()["ok"]
    assert client.get("/api/config").json()["provider"] in {"mock", "anthropic", "openai", "gemini"}
    r = client.post("/api/ask", json={"query": "What is the status of invoice INV-2026-0501?",
                                      "prompt_version": "v3"})
    assert r.status_code == 200 and r.json()["status"] == "answered"
    assert client.post("/api/ask", json={"query": "hello", "prompt_version": "v9"}).status_code == 400
