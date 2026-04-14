"""Ollama chat client used for the simulated user AND the text judge.

Same model, different system prompts. Supports a fallback-tag chain so the
bench degrades gracefully when the preferred tag (e.g. `nemotron:70b`) isn't
pulled on the Ollama host.
"""

from __future__ import annotations

import json
from typing import Iterable


class OllamaUnavailable(RuntimeError):
    pass


class OllamaChatClient:
    def __init__(self, base_url: str, model: str,
                 fallback_tags: Iterable[str] | None = None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.requested_model = model
        self.fallback_tags = list(fallback_tags) if fallback_tags else []
        self.timeout = timeout
        self._resolved_model: str | None = None

    def _list_models(self) -> list[str]:
        import httpx
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            r.raise_for_status()
        except Exception as e:
            raise OllamaUnavailable(f"{self.base_url}/api/tags: {e}") from e
        return [m["name"] for m in r.json().get("models", [])]

    def resolve_model(self) -> str:
        if self._resolved_model is not None:
            return self._resolved_model
        available = self._list_models()
        chain: list[str] = []
        if self.requested_model:
            chain.append(self.requested_model)
        for tag in self.fallback_tags:
            if tag not in chain:
                chain.append(tag)
        for tag in chain:
            if tag in available:
                self._resolved_model = tag
                return tag
        raise OllamaUnavailable(
            f"None of {chain} available on {self.base_url}. Pulled: {available}"
        )

    def chat(self, messages: list[dict], *, format_json: bool = False,
             temperature: float = 0.7) -> str:
        import httpx
        model = self.resolve_model()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except Exception as e:
            raise OllamaUnavailable(f"{self.base_url}/api/chat: {e}") from e
        body = r.json()
        return body.get("message", {}).get("content", "")

    def chat_json(self, messages: list[dict], *, temperature: float = 0.2) -> dict:
        content = self.chat(messages, format_json=True, temperature=temperature)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start:end + 1])
            raise


class SimulatedUser:
    """Wraps OllamaChatClient with a persona/scenario system prompt."""

    def __init__(self, client: OllamaChatClient, persona: str,
                 target_role: str, must_do: list[str]):
        self.client = client
        self.persona = persona
        self.target_role = target_role
        self.must_do = list(must_do)

    def _system_prompt(self) -> str:
        return (
            "You are role-playing as a conversational partner in a spoken-audio benchmark. "
            f"Your persona: {self.persona}\n"
            f"The model you are talking to is supposed to play: {self.target_role}. "
            "Stay fully in character. Speak naturally. "
            "CRITICAL: exactly ONE short spoken sentence per turn, at most 20 words, "
            "never more than 180 characters. No lists, no narration, no actions, no "
            "meta-commentary. Do NOT reveal you are an AI. "
            "Gently steer the conversation so the other party has opportunities to: "
            + "; ".join(self.must_do) + "."
        )

    def next_utterance(self, history: list[dict]) -> str:
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(history)
        if not history or history[-1]["role"] != "user":
            messages.append({"role": "user", "content": "(your turn to speak)"})
        return self.client.chat(messages, temperature=0.8).strip()
