"""HTTP adapters for Anthropic, OpenAI and Google Gemini (no vendor SDKs needed)."""
from __future__ import annotations

import os

from app.llm.base import LLM, LLMError


def _key(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise LLMError(f"{name} is not set")
    return value


class AnthropicLLM(LLM):
    provider = "anthropic"

    def _complete(self, system, messages, max_tokens, json_mode):
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            {"model": self.model, "max_tokens": max_tokens, "system": system,
             "messages": messages, "temperature": 0},
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class OpenAILLM(LLM):
    provider = "openai"

    def _complete(self, system, messages, max_tokens, json_mode):
        body = {"model": self.model, "max_completion_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post("https://api.openai.com/v1/chat/completions",
                          {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"}, body)
        usage = data.get("usage", {})
        return (data["choices"][0]["message"].get("content") or "",
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


class GeminiLLM(LLM):
    provider = "gemini"

    def _complete(self, system, messages, max_tokens, json_mode):
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]
        config = {"temperature": 0, "maxOutputTokens": max(max_tokens, 4096)}
        if json_mode:
            config["responseMimeType"] = "application/json"
        data = self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": _key("GEMINI_API_KEY")},
            {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
             "generationConfig": config},
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"gemini returned no candidates: {str(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        usage = data.get("usageMetadata", {})
        return ("".join(p.get("text", "") for p in parts),
                usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0))
