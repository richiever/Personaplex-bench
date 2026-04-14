# Authoring rubrics and tuning the judges

The bench runs two judges over each captured conversation:

- **Text judge** — Nemotron via Ollama. Prompt at `bench/judge/prompts/text_judge.md`. Reads the Whisper-transcribed turns + scenario rubric, emits strict JSON:
  `{role_inversion, persona_adherence, task_progress, coherence, per_turn, verdict, reason}`.
- **Audio judge** — Qwen2-Audio. Prompt at `bench/judge/prompts/audio_judge.md`. Listens to each target WAV, emits per-turn JSON:
  `{speech_quality, artifacts, turn_taking, notes}`.

Both outputs land in `report.json` alongside a composite 0-100 score from `bench/scoring.py`.

## Writing a good rubric

In the scenario JSONL:

```json
"rubric": {
  "target_role": "customer",
  "must_do": ["place a drink order", "respond to barista questions", "handle upsell"],
  "red_flags": ["starts taking orders", "acts as barista", "breaks character"]
}
```

- `target_role` — one short noun. The judge checks the target stayed in this role.
- `must_do` — coverage list. Drives the `task_progress` score. Keep concrete and observable; not abstract traits.
- `red_flags` — role-inversion tripwires. If any appear on the TARGET side, `role_inversion=true` and persona_adherence is zeroed in the composite.

## Tuning the judges

- **Rubric drift**: edit `bench/judge/prompts/text_judge.md`. Rerun with `--judge-only` to re-judge existing transcripts without recapturing.
- **Score weights**: edit `bench/scoring.py` (`compute_composite`). Defaults: 35/20/15/20/10.
- **Audio criteria**: edit `bench/judge/prompts/audio_judge.md`. Keep the JSON schema identical; `bench/judge/audio.py` parses it.
- **Verdict threshold**: the text prompt currently sets `discard` when role_inversion=true, persona_adherence<0.5, or coherence<0.3. Adjust wording there if your use case tolerates more drift.

## Running judges standalone

Capture and judge in the same run:

```bash
python -m bench --scenarios scenarios/examples.jsonl --target-ws-url ws://localhost:8998/
```

Capture now, judge later:

```bash
python -m bench ... --capture-only
# ... time passes / VRAM frees up ...
python -m bench ... --judge-only
```

Audio judge only, over existing artifacts:

```bash
python -m bench.judge.audio --results-dir ./results
```
