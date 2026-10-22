"""Build an LLM client from a provider name and optional model id."""
from __future__ import annotations

from app.llm.base import LLM
from app.llm.mock import MockLLM
from app.llm.providers import AnthropicLLM, GeminiLLM, OpenAILLM

# Defaults only. Model names move quickly; override with LLM_MODEL / --model.
DEFAULT_MODELS = {
    "mock": "mock",
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
}
PROVIDERS = {"mock": MockLLM, "anthropic": AnthropicLLM, "openai": OpenAILLM, "gemini": GeminiLLM}


def get_llm(provider: str, model: str | None = None) -> LLM:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[provider](model or DEFAULT_MODELS[provider])
