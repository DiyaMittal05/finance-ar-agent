"""Tools the research agent can call. ERP tools go through the FastAPI service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx

from app.config import settings
from app.rag.retriever import get_retriever


def erp_client() -> httpx.Client:
    """HTTP client for the ERP API: remote if ERP_BASE_URL is set, else the in-process app."""
    if settings.erp_base_url:
        return httpx.Client(base_url=settings.erp_base_url, timeout=20)
    from fastapi.testclient import TestClient  # an httpx.Client subclass

    from app.erp.service import app as erp_app
    return TestClient(erp_app)


@dataclass
class Tool:
    name: str
    signature: str
    description: str
    run: Callable[..., Any]


class ToolBox:
    def __init__(self, include_aging: bool):
        self.client = erp_client()
        tools = [
            Tool("search_customers", "search_customers(query: str)",
                 "Find customers by name. Returns matches with match_score and an 'ambiguous' flag.",
                 lambda query: self._get("/customers", q=query)),
            Tool("get_customer", "get_customer(customer_id: str)",
                 "Customer record: payment terms, credit limit, account status, AP contact.",
                 lambda customer_id: self._get(f"/customers/{customer_id}")),
            Tool("list_invoices", "list_invoices(customer_id: str, status: str | None = None)",
                 "Raw invoices for a customer with amount and paid_amount. Status: open, paid, partially_paid, disputed.",
                 lambda customer_id, status=None: self._get(f"/customers/{customer_id}/invoices", status=status)),
            Tool("get_invoice", "get_invoice(invoice_number: str)",
                 "One invoice with its payments, outstanding amount and days overdue.",
                 lambda invoice_number: self._get(f"/invoices/{invoice_number}")),
            Tool("list_overdue_invoices", "list_overdue_invoices(min_days: int = 1, max_days: int | None = None)",
                 "Overdue invoices across all customers, filtered by days overdue.",
                 lambda min_days=1, max_days=None: self._get("/invoices/overdue", min_days=min_days,
                                                             max_days=max_days)),
            Tool("search_policies", "search_policies(query: str)",
                 "Search payment terms, the collections policy and customer contracts. Returns cited text chunks.",
                 lambda query: {"query": query, "chunks": get_retriever().search(query, k=4)}),
        ]
        if include_aging:
            tools.insert(3, Tool("get_aging_summary", "get_aging_summary(customer_id: str)",
                                 "ERP-computed aging: outstanding and overdue totals, buckets, days overdue per "
                                 "invoice, dunning stage and anomalies (duplicates, disputes, missing due dates).",
                                 lambda customer_id: self._get(f"/customers/{customer_id}/aging")))
        self.tools = {t.name: t for t in tools}

    def _get(self, path: str, **params: Any) -> Any:
        r = self.client.get(path, params={k: v for k, v in params.items() if v is not None})
        if r.status_code == 404:
            return {"error": r.json().get("detail", "not found")}
        r.raise_for_status()
        return r.json()

    def describe(self) -> str:
        return "\n".join(f"- {t.signature}: {t.description}" for t in self.tools.values())

    def call(self, name: str, args: dict) -> Any:
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"Unknown tool {name!r}. Available: {sorted(self.tools)}"}
        try:
            return tool.run(**(args or {}))
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
