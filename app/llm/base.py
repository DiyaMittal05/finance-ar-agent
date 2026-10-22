"""Provider-neutral LLM interface.

Agents talk to every model through `LLM.complete` with plain text and ask for
JSON in the prompt. That keeps the same agent code running on Claude, Gemini,
GPT and the offline mock, which is what makes cross-model evals fair.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import yaml

from app.config import ROOT


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


class LLMError(RuntimeError):
    pass


class LLM:
    provider = "base"

    def __init__(self, model: str, timeout: float = 90.0, max_retries: int = 4):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, system: str, messages: list[dict], max_tokens: int = 2000,
                 json_mode: bool = True) -> LLMResult:
        start = time.perf_counter()
        text, tin, tout = self._complete(system, messages, max_tokens, json_mode)
        return LLMResult(text, tin, tout, time.perf_counter() - start, self.model)

    def _complete(self, system: str, messages: list[dict], max_tokens: int,
                  json_mode: bool) -> tuple[str, int, int]:
        raise NotImplementedError

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        """POST with exponential backoff on rate limits and server errors."""
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(url, headers=headers, json=body, timeout=self.timeout)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.provider} request failed: {exc}") from exc
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code not in (408, 429, 500, 502, 503, 504, 529) or attempt == self.max_retries:
                    raise LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:500]}")
            time.sleep(delay)
            delay *= 2
        raise LLMError("unreachable")


def parse_json(text: str) -> dict:
    """Parse the first JSON object in a model reply, tolerating code fences and prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = ch == "\\" and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"No JSON object in model output: {text[:200]!r}")


@lru_cache(maxsize=1)
def _pricing() -> dict:
    return yaml.safe_load(Path(ROOT / "config" / "pricing.yaml").read_text()) or {}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = _pricing().get(model)
    if not price or price.get("input_per_mtok") is None or price.get("output_per_mtok") is None:
        return None
    return (input_tokens * price["input_per_mtok"] + output_tokens * price["output_per_mtok"]) / 1e6
