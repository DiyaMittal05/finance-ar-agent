"""Public app: the AR assistant API, a demo UI, and the ERP API mounted at /erp.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agents.orchestrator import AUTO_ANALYTICS, Orchestrator, prompt_versions
from app.config import ROOT, settings
from app.data.generate import ensure_db
from app.erp.service import app as erp_app
from app.llm.registry import get_llm

STATIC = Path(__file__).parent / "static"
RESULTS = ROOT / "evals" / "results"
# Protects API spend on a public URL. In-memory, so it is per instance.
MAX_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "30"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_db()
    yield


app = FastAPI(title="AR Assistant", version="1.0", lifespan=lifespan,
              description="Multi-agent accounts receivable assistant over a synthetic ERP.")
app.mount("/erp", erp_app)
_hits: dict[str, deque] = defaultdict(deque)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    prompt_version: str | None = None


def _rate_limit(request: Request) -> None:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    now, window = time.time(), _hits[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= MAX_PER_HOUR:
        raise HTTPException(429, "Hourly request limit reached for this demo.")
    window.append(now)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    llm = get_llm(settings.llm_provider, settings.llm_model or None)
    return {"provider": llm.provider, "model": llm.model, "prompt_versions": prompt_versions(),
            "default_prompt_version": settings.prompt_version, "as_of_date": settings.as_of_date.isoformat()}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> dict:
    _rate_limit(request)
    version = body.prompt_version or settings.prompt_version
    if version not in prompt_versions():
        raise HTTPException(400, f"Unknown prompt version {version!r}")
    orch = Orchestrator(get_llm(settings.llm_provider, settings.llm_model or None), version)
    result = orch.run(body.query.strip())
    result["evidence"] = [{"tool": e["tool"], "args": e["args"]} for e in result["evidence"]
                          if e["tool"] != AUTO_ANALYTICS]
    return result


@app.get("/api/evals")
def evals() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("*.json")):
        r = json.loads(p.read_text())
        out.append({k: r[k] for k in ("provider", "model", "prompt_version", "judge", "run_at", "summary")})
    return out


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}
