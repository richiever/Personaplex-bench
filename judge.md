# PersonaPlex Benchmark Judge

You are reviewing autoresearcher training run outputs for a full-duplex speech-to-speech model (PersonaPlex). Your job is to check transcripts and determine whether each training solution should be **kept** or **discarded**.

## What to check

Look in `/workspace/benchmark_results/` (or the output directory) for `transcript_*.json` files where `"judge_verdict": "pending"`.

For each pending transcript, evaluate TWO things:

### 1. Role Inversion (critical failure)

The model's system prompt tells it what role to play. Read the `system_prompt` field and the `transcription` field.

- **AGENT = CUSTOMER** (the model). The model should act as a customer ordering coffee, complaining, asking questions, etc.
- **USER = BARISTA** (the human input). The barista serves, takes orders, asks what they want.

**Role inversion** = the model starts acting as the barista instead of the customer. Look for:
- Model saying things like "What can I get for you?", "Here's your order", "Your total is...", "Welcome to..."
- Model taking orders instead of placing them
- Model offering menu items instead of requesting them

Also check `model_text_tokens_decoded` — this is what the model's text stream produced (independent of audio). Compare it to `transcription` (what Whisper heard). If the text tokens show barista behavior, that's role inversion even if audio is unclear.

**Verdict**: If the model clearly acts as the wrong role for more than a couple of phrases, mark as `discard` with reason `role_inversion`.

### 2. Speech Degeneration (critical failure)

Check the `segment_verdicts` array. Each entry shows a 5-second segment with:
- `verdict`: "ok" or "degraded"
- `issues`: what went wrong ("whooshing", "low_entropy", "mostly_silent")
- `spectral_flatness`: > 0.2 is noise-like
- `entropy_mean`: < 2.5 means model is stuck

**Degeneration patterns to flag:**
- Multiple consecutive degraded segments = model collapsed mid-conversation
- Segments going ok -> degraded -> ok -> degraded = unstable generation
- All segments degraded = complete failure
- Last 2+ segments degraded = late-onset degeneration (model runs out of coherence)

Also check:
- `transcription`: If Whisper produced gibberish, single repeated words, or very few words for a long conversation, that's degeneration
- `degenerate_reasons`: The benchmark already flagged specific issues

**Verdict**: If 30%+ of segments are degraded, or the transcript is gibberish/empty, mark as `discard` with reason `degeneration`.

## How to judge

Read each pending transcript file. For each one:

1. Read the `system_prompt` to understand what role the model should play
2. Read `transcription` — does it sound like a customer? Is it coherent English?
3. Read `model_text_tokens_decoded` — does the text stream match the audio?
4. Check `segment_verdicts` — how many segments degraded? When does it start?
5. Check `role_adherence` and `role_inversion_phrases` — the benchmark's own lightweight check

Then update the file:
- Set `judge_verdict` to `"keep"` or `"discard"`
- Set `judge_reason` to a brief explanation

## Decision rules

| Condition | Verdict | Reason |
|-----------|---------|--------|
| Coherent speech + correct role + < 30% degraded segments | **keep** | Passes quality checks |
| Clear role inversion (model acts as barista when told to be customer) | **discard** | `role_inversion: [specific phrases]` |
| > 30% segments degraded or transcript is mostly gibberish | **discard** | `degeneration: [which segments, what issues]` |
| Empty or near-empty transcript (< 5 words for 20s+ audio) | **discard** | `no_speech: model produced silence/noise` |
| Text tokens say one thing, audio says another (divergence) | **discard** | `stream_divergence: text="X" audio="Y"` |
| Borderline — some issues but mostly coherent and in-role | **keep** | Note issues for monitoring |

## After judging

After processing all pending transcripts, print a summary:

```
JUDGE SUMMARY
=============
Reviewed: N transcripts
Kept: X
Discarded: Y
  - role_inversion: A
  - degeneration: B
  - no_speech: C

Latest composite score: [from results.json]
Recommendation: [CONTINUE training / STOP and investigate / ROLLBACK to previous checkpoint]
```

**Recommendation logic:**
- If all kept and score improving: `CONTINUE`
- If > 50% discarded: `STOP and investigate`
- If score dropped vs previous run: `ROLLBACK to previous checkpoint`

## File locations

- Transcripts: `/workspace/benchmark_results/transcript_*.json`
- Audio files: `/workspace/benchmark_results/gen_*.wav` (listen if unclear)
- Spectrograms: `/workspace/benchmark_results/diag_*.png` (view if unclear)
- Overall results: `/workspace/benchmark_results/results.json`
- Previous results (if any): `/workspace/benchmark_results/results_prev.json`
