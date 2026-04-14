from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Rubric:
    target_role: str
    must_do: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    system_prompt_for_target: str
    user_persona: str
    user_voice_ref_wav: str
    rubric: Rubric
    target_model_hint: str | None = None
    opening_utterance: str | None = None
    turns: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        rubric = Rubric(
            target_role=d["rubric"]["target_role"],
            must_do=list(d["rubric"].get("must_do", [])),
            red_flags=list(d["rubric"].get("red_flags", [])),
        )
        return cls(
            id=d["id"],
            system_prompt_for_target=d["system_prompt_for_target"],
            user_persona=d["user_persona"],
            user_voice_ref_wav=d["user_voice_ref_wav"],
            rubric=rubric,
            target_model_hint=d.get("target_model_hint"),
            opening_utterance=d.get("opening_utterance"),
            turns=int(d.get("turns", 5)),
        )


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                scenarios.append(Scenario.from_dict(json.loads(line)))
            except Exception as e:
                raise ValueError(f"{path}:{line_num}: bad scenario JSONL row: {e}") from e
    return scenarios
