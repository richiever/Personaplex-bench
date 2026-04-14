# Personaplex-bench

Audio-in / audio-out LLM-as-judge benchmark for full-duplex conversational speech models (Personaplex, Moshi, or any WebSocket-audio server).

## What it does

1. **Simulates a user** — an LLM (default Nemotron via Ollama) writes the next utterance for a worker/co-worker persona; F5-TTS (voice-cloned from a reference WAV) synthesizes it.
2. **Talks to the target model** — streams audio frames over WebSocket to the speech-to-speech model being benchmarked (plays e.g. a customer).
3. **Records everything** — per-turn user and agent WAVs, Whisper-transcribed both sides, turn timings.
4. **Judges** — Nemotron-via-Ollama scores persona adherence, task progress, coherence, role inversion (text judge). Qwen2-Audio rates speech quality, turn-taking, audio artifacts (audio judge).
5. **Reports** — per-run JSON + cross-scenario rollup + composite 0-100 score.

Focus is role/task validation and audio quality, not latency.

## Prerequisites

- GPU with ~20 GB VRAM (phase 1 talks to the target model; phase 2 loads Qwen2-Audio).
- A full-duplex audio WS server for the target (e.g. Personaplex/Moshi running on the same Pod at `ws://localhost:8998/ws`).
- Ollama running with Nemotron pulled. `ollama pull nemotron:70b` (falls back to `nemotron:22b` or `nemotron:mini` if unavailable).
- System `libopus` if the target uses Opus. `apt-get install libopus0 libopus-dev`.

## Install

```bash
git clone <this repo>
cd Personaplex-bench
pip install -r requirements.txt
python scripts/fetch_voices.py   # pulls a few permissive reference voices
```

## Quickstart

Local smoke (no real target, no GPU) with the mock echo server:

```bash
# terminal 1
python -m bench.target.mock_server --port 8999

# terminal 2
python -m bench \
  --scenarios scenarios/examples.jsonl \
  --target-ws-url ws://localhost:8999/ \
  --target-protocol moshi --target-codec pcm \
  --tts mock --skip-audio-judge --skip-whisper \
  --num-runs 1
```

Real run against a Personaplex WS on the same host:

```bash
python -m bench \
  --scenarios scenarios/examples.jsonl \
  --target-ws-url ws://localhost:8998/ \
  --target-protocol personaplex \
  --user-llm-model nemotron:70b \
  --text-judge-model nemotron:70b \
  --num-runs 5
```

## Scenario authoring

Scenarios live in JSONL files. One line per scenario:

```json
{"id": "coffee_rush_0001",
 "target_model_hint": "richiever/Personaplex-fine-coffee",
 "system_prompt_for_target": "You are a regular customer at a busy morning cafe, in a hurry.",
 "user_persona": "Friendly barista taking orders during a rush.",
 "user_voice_ref_wav": "voices/barista_1.wav",
 "opening_utterance": "Hey! What can I get started for you today?",
 "turns": 5,
 "rubric": {
   "target_role": "customer",
   "must_do": ["place a drink order", "respond to barista questions"],
   "red_flags": ["starts taking orders", "acts as barista", "breaks character"]
 }}
```

- **Simulated user** = the worker side (the one the benchmark plays).
- **Target model** = the one under test (e.g. a customer persona).
- Roles are fully swappable: nothing in the schema hardcodes who's customer vs worker.

See `bench/judge/prompts/text_judge.md` to tune the judge rubric.

## Two-phase execution

The bench runs in phases so you can judge later or on a different host:

- **Phase 1 (capture)** streams to the WS server and saves per-turn WAVs + `transcript.json`.
  `python -m bench ... --capture-only` exits here.
- **Phase 2a (transcription)** runs Whisper over every saved WAV, updates `transcript.json`.
- **Phase 2b (text judge)** calls Ollama and writes `report.json` with composite score.
- **Phase 2c (audio judge)** loads Qwen2-Audio. If OOM / unavailable, it's skipped and `audio_judge_status: pending` stays in `report.json`.

To run only the judges over existing artifacts:

```bash
python -m bench ... --judge-only
python -m bench.judge.audio --results-dir ./results
```

## Output layout

```
results/
  <scenario_id>/
    run_00/
      user_turn_00.wav, agent_turn_00.wav, ... (per turn)
      transcript.json   # conv log + whisper text
      report.json       # + judges + composite 0-100
    run_01/ ...
  rollup.json           # aggregate across scenarios
```

## Composite score (0-100)

- 35 pts persona / role adherence (text_judge.persona_adherence, zeroed if role_inversion)
- 20 pts task progress
- 15 pts coherence
- 20 pts speech quality (audio_judge.speech_quality / 5)
- 10 pts turn-taking

Weights are editable in `bench/scoring.py`.
