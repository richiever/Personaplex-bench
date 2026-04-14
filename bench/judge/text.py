"""Nemotron-via-Ollama text judge. Structured JSON output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..simulator.llm import OllamaChatClient
from ..utils.logging import get_logger

log = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "text_judge.md"


def _format_transcript(conv) -> str:
    lines: list[str] = []
    for t in conv.turns:
        u = (t.user_transcript or t.user_text or "").strip()
        a = (t.agent_transcript or t.inline_agent_text or "").strip()
        lines.append(f"[turn {t.index}] USER ({t.user_sr} Hz): {u}")
        lines.append(f"[turn {t.index}] TARGET: {a}")
    return "\n".join(lines)


class TextJudge:
    def __init__(self, ollama_url: str, model: str,
                 fallback_tags: list[str] | None = None):
        self.client = OllamaChatClient(ollama_url, model, fallback_tags=fallback_tags)
        self._prompt_tmpl: str | None = None

    def _prompt(self) -> str:
        if self._prompt_tmpl is None:
            self._prompt_tmpl = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_tmpl

    def judge(self, scenario, conv) -> dict[str, Any]:
        tmpl = self._prompt()
        rubric = scenario.rubric
        filled = tmpl.format(
            user_persona=scenario.user_persona,
            target_role=rubric.target_role,
            system_prompt_for_target=scenario.system_prompt_for_target,
            must_do="; ".join(rubric.must_do) or "(none specified)",
            red_flags="; ".join(rubric.red_flags) or "(none specified)",
            transcript=_format_transcript(conv),
        )
        messages = [
            {"role": "system", "content": "You are a precise, strict JSON-only evaluator."},
            {"role": "user", "content": filled},
        ]
        try:
            result = self.client.chat_json(messages, temperature=0.1)
        except Exception as e:
            log.exception("text judge failed: %s", e)
            return {
                "error": str(e),
                "role_inversion": None,
                "persona_adherence": None,
                "task_progress": None,
                "coherence": None,
                "per_turn": [],
                "verdict": "error",
                "reason": f"judge error: {e}",
            }
        result.setdefault("verdict", "keep")
        result.setdefault("reason", "")
        result.setdefault("per_turn", [])
        result["resolved_judge_model"] = self.client.resolve_model()
        return result
