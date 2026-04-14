"""Per-run artifact tree + report/rollup writers.

Layout:
  <output_dir>/<scenario_id>/run_<n>/
    user_turn_<k>.wav
    agent_turn_<k>.wav
    transcript.json   <- conversation log + whisper text
    report.json       <- + judges + scoring
  <output_dir>/rollup.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def run_dir(output_dir: Path, scenario_id: str, run_index: int) -> Path:
    d = Path(output_dir) / scenario_id / f"run_{run_index:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_transcript(conv, path: Path | None = None) -> Path:
    if path is None:
        path = Path(conv.turns[0].user_wav_path).parent / "transcript.json" if conv.turns else None
        assert path is not None, "cannot infer transcript path from empty conversation"
    from .orchestrator import conversation_to_dict
    path.write_text(json.dumps(conversation_to_dict(conv), indent=2), encoding="utf-8")
    return path


def write_report(run_dir_path: Path, report: dict[str, Any]) -> Path:
    p = run_dir_path / "report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


def write_rollup(output_dir: Path, rollup_obj: dict[str, Any]) -> Path:
    p = Path(output_dir) / "rollup.json"
    p.write_text(json.dumps(rollup_obj, indent=2), encoding="utf-8")
    return p


def load_transcript(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
