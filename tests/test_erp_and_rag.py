from fastapi.testclient import TestClient

from app.erp.service import app
from app.rag.retriever import get_retriever

client = TestClient(app)


def search(q):
    return client.get("/customers", params={"q": q}).json()


def test_ambiguous_name_is_flagged():
    r = search("Acme")
    assert r["ambiguous"] and {m["id"] for m in r["matches"]} == {"C001", "C002"}


def test_exact_and_typo_matches():
    assert search("acme corporation")["matches"] == [{"id": "C001", "name": "Acme Corporation", "match_score": 1.0}]
    assert search("Globx")["matches"][0]["id"] == "C003"


def test_generic_industry_word_alone_does_not_match():
    assert search("Cyberdyne Systems")["matches"] == []


def test_not_found_is_404():
    assert client.get("/invoices/INV-2026-9999").status_code == 404
    assert client.get("/customers/C999").status_code == 404


def test_overdue_endpoint_filters_by_days():
    rows = client.get("/invoices/overdue", params={"min_days": 91}).json()["invoices"]
    assert rows and all(r["days_overdue"] >= 91 for r in rows)
    assert "INV-2026-0901" in {r["invoice_number"] for r in rows}


def test_retrieval_finds_relevant_policy():
    top = lambda q: [c["chunk_id"] for c in get_retriever().search(q, k=3)]
    assert "payment_terms.md#late-fees" in top("late fee terms")
    assert "collections_policy.md#disputed-invoices" in top("disputed invoice")
    assert "contracts/acme_corporation_msa.md#payment-terms" in top("Acme Corporation contract payment terms")
