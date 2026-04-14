"""Composite 0-100 scoring combining text + audio judges."""

from __future__ import annotations

from typing import Any


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _turn_taking_score(label: str | None) -> float:
    return {"ok": 1.0, "lag": 0.5, "cutoff": 0.5, "overlap": 0.3}.get(label or "", 0.5)


def compute_composite(text_judge: dict[str, Any],
                      audio_judge: list[dict[str, Any]] | None) -> dict[str, Any]:
    role_inversion = bool(text_judge.get("role_inversion") or False)
    persona = float(text_judge.get("persona_adherence") or 0.0)
    task = float(text_judge.get("task_progress") or 0.0)
    coherence = float(text_judge.get("coherence") or 0.0)

    persona_component = 0.0 if role_inversion else persona
    persona_pts = 35.0 * persona_component
    task_pts = 20.0 * task
    coherence_pts = 15.0 * coherence

    if audio_judge:
        sq = _avg([t.get("speech_quality") for t in audio_judge]) / 5.0
        tt = _avg([_turn_taking_score(t.get("turn_taking")) for t in audio_judge])
    else:
        sq = 0.0
        tt = 0.0
    speech_pts = 20.0 * sq
    turn_pts = 10.0 * tt

    total = persona_pts + task_pts + coherence_pts + speech_pts + turn_pts
    return {
        "composite_0_100": round(total, 2),
        "components": {
            "persona_adherence": round(persona_pts, 2),
            "task_progress": round(task_pts, 2),
            "coherence": round(coherence_pts, 2),
            "speech_quality": round(speech_pts, 2),
            "turn_taking": round(turn_pts, 2),
        },
        "role_inversion": role_inversion,
        "audio_judged": bool(audio_judge),
    }


def aggregate_run(conv, text_judge: dict[str, Any],
                  audio_judge: list[dict[str, Any]] | None) -> dict[str, Any]:
    scoring = compute_composite(text_judge, audio_judge)
    return {
        "scenario_id": conv.scenario_id,
        "run_index": conv.run_index,
        "seed": conv.seed,
        "resolved_user_llm": conv.resolved_user_llm,
        "target_ws_url": conv.target_ws_url,
        "target_protocol": conv.target_protocol,
        "text_judge": text_judge,
        "audio_judge": audio_judge if audio_judge is not None else [],
        "audio_judge_status": "complete" if audio_judge else "pending",
        "scoring": scoring,
        "num_turns": len(conv.turns),
    }


def rollup(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"n": 0, "scenarios": []}
    by_scenario: dict[str, list[dict]] = {}
    for r in reports:
        by_scenario.setdefault(r["scenario_id"], []).append(r)
    scenarios_out = []
    all_scores = []
    for sid, runs in by_scenario.items():
        scores = [r["scoring"]["composite_0_100"] for r in runs]
        role_inv = sum(1 for r in runs if r["scoring"]["role_inversion"])
        verdicts = [r["text_judge"].get("verdict") for r in runs]
        scenarios_out.append({
            "scenario_id": sid,
            "n_runs": len(runs),
            "mean_composite": round(sum(scores) / len(scores), 2),
            "min_composite": min(scores),
            "max_composite": max(scores),
            "role_inversion_count": role_inv,
            "verdicts": verdicts,
        })
        all_scores.extend(scores)
    return {
        "n": len(reports),
        "mean_composite": round(sum(all_scores) / len(all_scores), 2),
        "scenarios": scenarios_out,
    }
