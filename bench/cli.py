from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .artifacts import run_dir, write_rollup, write_report, write_transcript
from .config import load_scenarios
from .orchestrator import conversation_to_dict, run_session
from .scoring import aggregate_run, rollup
from .utils.logging import get_logger

log = get_logger("bench.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench",
        description="Audio-in/audio-out LLM-as-judge benchmark for full-duplex speech models.",
    )
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./results"))
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--target-backend", default="ws", choices=["ws", "offline"],
                   help="ws = full-duplex WebSocket; offline = invoke moshi.offline subprocess per turn.")
    p.add_argument("--target-ws-url", default="ws://localhost:8998/ws")
    p.add_argument("--target-protocol", default="moshi",
                   choices=["moshi", "personaplex", "custom"])
    p.add_argument("--target-codec", default=None, choices=["opus", "pcm"],
                   help="Override dialect's default codec (opus|pcm).")
    p.add_argument("--target-voice-prompt", default=None,
                   help="Voice prompt file name (e.g. NATM1.pt) passed to the target.")
    p.add_argument("--moshi-venv-python", default=None,
                   help="Path to a python interpreter where `moshi.offline` is importable "
                        "(for --target-backend offline).")
    p.add_argument("--end-of-turn-silence-ms", type=int, default=1200)
    p.add_argument("--inter-conversation-silence-s", type=float, default=3.0,
                   help="Silence pumped between back-to-back conversations on the "
                        "same persistent WS connection.")

    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--user-llm-model", default="nemotron:70b")
    p.add_argument("--text-judge-model", default="nemotron:70b")
    p.add_argument("--ollama-fallback-tags", nargs="*",
                   default=["nemotron:70b", "nemotron:22b", "nemotron:mini"])

    p.add_argument("--audio-judge-model", default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument("--skip-audio-judge", action="store_true")

    p.add_argument("--tts", default="f5-tts", choices=["f5-tts", "mock"])
    p.add_argument("--tts-voices-dir", type=Path, default=Path("./voices"))

    p.add_argument("--whisper-model", default="base")
    p.add_argument("--whisper-device", default="cuda")
    p.add_argument("--skip-whisper", action="store_true")

    p.add_argument("--capture-only", action="store_true",
                   help="Run only phase 1 (capture audio); skip all judges.")
    p.add_argument("--judge-only", action="store_true",
                   help="Skip phase 1; run judges over an existing output-dir.")
    return p


async def _phase1_capture(args, scenarios) -> list:
    conversations = []
    for scenario in scenarios:
        log.info("PHASE 1 capture: %s (persistent WS, %d runs)",
                 scenario.id, args.num_runs)
        try:
            session_results = await run_session(
                scenario,
                num_runs=args.num_runs,
                base_seed=args.seed,
                output_dir=args.output_dir,
                target_ws_url=args.target_ws_url,
                target_protocol=args.target_protocol,
                target_codec=args.target_codec,
                ollama_url=args.ollama_url,
                user_llm_model=args.user_llm_model,
                fallback_tags=args.ollama_fallback_tags,
                tts_name=args.tts,
                voices_dir=args.tts_voices_dir,
                eot_silence_ms=args.end_of_turn_silence_ms,
                target_backend=args.target_backend,
                target_voice_prompt=args.target_voice_prompt,
                moshi_venv_python=args.moshi_venv_python,
                inter_conversation_silence_s=args.inter_conversation_silence_s,
            )
        except Exception as e:
            log.exception("scenario %s session failed: %s", scenario.id, e)
            continue
        for conv in session_results:
            d = run_dir(args.output_dir, scenario.id, conv.run_index)
            write_transcript(conv, d / "transcript.json")
            conversations.append(conv)
    return conversations


def _phase2_transcribe(args, conversations) -> None:
    if args.skip_whisper:
        return
    try:
        from .transcription.whisper import WhisperTranscriber
    except Exception as e:
        log.warning("whisper unavailable: %s", e)
        return
    w = WhisperTranscriber(model=args.whisper_model, device=args.whisper_device)
    for conv in conversations:
        try:
            w.transcribe_conversation(conv)
        except Exception as e:
            log.warning("whisper failed for %s run %d: %s", conv.scenario_id, conv.run_index, e)
            continue
        d = run_dir(args.output_dir, conv.scenario_id, conv.run_index)
        write_transcript(conv, d / "transcript.json")


def _phase2_text_judge(args, scenarios_by_id, conversations) -> list[dict]:
    from .judge.text import TextJudge
    judge = TextJudge(args.ollama_url, args.text_judge_model,
                      fallback_tags=args.ollama_fallback_tags)
    reports: list[dict] = []
    for conv in conversations:
        scenario = scenarios_by_id[conv.scenario_id]
        try:
            text_verdict = judge.judge(scenario, conv)
        except Exception as e:
            log.exception("text judge failed: %s", e)
            text_verdict = {"error": str(e), "verdict": "error", "reason": str(e)}
        report = aggregate_run(conv, text_verdict, audio_judge=None)
        d = run_dir(args.output_dir, conv.scenario_id, conv.run_index)
        write_report(d, report)
        reports.append(report)
    return reports


def _phase2_audio_judge(args, conversations, reports) -> None:
    if args.skip_audio_judge:
        log.info("audio judge skipped by flag")
        return
    try:
        from .judge.audio import AudioJudge, AudioJudgeUnavailable
    except Exception as e:
        log.warning("audio judge import failed: %s", e)
        return
    try:
        judge = AudioJudge(model=args.audio_judge_model)
    except AudioJudgeUnavailable as e:
        log.warning("audio judge unavailable: %s", e)
        return
    for conv, report in zip(conversations, reports):
        try:
            audio_verdicts = judge.judge_conversation(conv)
        except Exception as e:
            log.warning("audio judge failed for %s run %d: %s (report left pending)",
                        conv.scenario_id, conv.run_index, e)
            continue
        # rewrite report with audio judge included
        updated = aggregate_run(conv, report["text_judge"], audio_verdicts)
        d = run_dir(args.output_dir, conv.scenario_id, conv.run_index)
        write_report(d, updated)
        report.clear()
        report.update(updated)


def _load_conversations_from_disk(output_dir: Path) -> list:
    from .orchestrator import ConversationLog, TurnLog
    convs = []
    for tr in sorted(output_dir.glob("*/run_*/transcript.json")):
        data = json.loads(tr.read_text(encoding="utf-8"))
        turns = [TurnLog(**t) for t in data.get("turns", [])]
        conv = ConversationLog(
            scenario_id=data["scenario_id"],
            run_index=data["run_index"],
            seed=data.get("seed", 0),
            resolved_user_llm=data.get("resolved_user_llm", ""),
            target_ws_url=data.get("target_ws_url", ""),
            target_protocol=data.get("target_protocol", ""),
        )
        conv.turns = turns
        conv.started_at = data.get("started_at", 0.0)
        conv.ended_at = data.get("ended_at", 0.0)
        convs.append(conv)
    return convs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        print(f"no scenarios loaded from {args.scenarios}", file=sys.stderr)
        return 2
    scenarios_by_id = {s.id: s for s in scenarios}

    if args.judge_only:
        conversations = _load_conversations_from_disk(args.output_dir)
        if not conversations:
            print(f"no transcripts under {args.output_dir} for judge-only mode", file=sys.stderr)
            return 2
    else:
        conversations = asyncio.run(_phase1_capture(args, scenarios))
        if not conversations:
            print("no conversations captured", file=sys.stderr)
            return 1
        if args.capture_only:
            log.info("capture-only mode; skipping judges")
            return 0

    _phase2_transcribe(args, conversations)

    reports = _phase2_text_judge(args, scenarios_by_id, conversations)

    _phase2_audio_judge(args, conversations, reports)

    rollup_obj = rollup(reports)
    write_rollup(args.output_dir, rollup_obj)
    log.info("rollup: %s", rollup_obj)
    print(json.dumps(rollup_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
