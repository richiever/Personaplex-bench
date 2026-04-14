"""Per-scenario x per-turn conversation loop.

Phase-1 of the bench: stream audio back and forth, save per-turn WAVs,
produce `transcript.json` (without Whisper text yet; transcription fills in).
Phase-2 (judges) runs afterwards over the saved artifacts.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Scenario
from .simulator.llm import OllamaChatClient, SimulatedUser
from .simulator.tts import build_tts
from .target.offline_client import OfflineMoshiClient
from .target.ws_client import MoshiWSClient
from .utils.audio import concat, save_wav
from .utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TurnLog:
    index: int
    user_text: str
    user_wav_path: str
    user_sr: int
    agent_wav_path: str
    agent_sr: int
    agent_duration_s: float
    inline_agent_text: str = ""
    user_transcript: str = ""
    agent_transcript: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0


@dataclass
class ConversationLog:
    scenario_id: str
    run_index: int
    seed: int
    resolved_user_llm: str
    target_ws_url: str
    target_protocol: str
    turns: list[TurnLog] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0


async def _silence_pump(client: MoshiWSClient, sample_rate: int,
                        stop_event: asyncio.Event, chunk_s: float = 0.48) -> None:
    """Continuously pump silence frames so a full-duplex server keeps processing.

    Moshi-style servers only advance state when receiving audio; otherwise the
    model never produces a response. We feed zeros at real-time pace.
    """
    n = int(sample_rate * chunk_s)
    zeros = np.zeros(n, dtype=np.float32)
    try:
        while not stop_event.is_set():
            await client.send_audio(zeros, sample_rate)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=chunk_s)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("silence pump error: %s", e)


async def _collect_agent_turn(client: MoshiWSClient, sample_rate: int,
                              eot_silence_s: float,
                              max_turn_s: float = 40.0,
                              max_wait_for_first_audio_s: float = 15.0,
                              send_task: asyncio.Task | None = None
                              ) -> tuple[list[np.ndarray], list[str]]:
    """Drain frames until `eot_silence_s` of silence after first audio, or `max_turn_s`.

    A silence pump runs after user audio finishes sending, so the full-duplex
    server keeps advancing its decoder while we wait for the agent's reply.
    """
    audio_chunks: list[np.ndarray] = []
    text_chunks: list[str] = []
    t_start = time.monotonic()
    last_audio_t = None
    got_audio = False

    stop_pump = asyncio.Event()
    pump_task: asyncio.Task | None = None

    async def _maybe_start_pump():
        nonlocal pump_task
        if pump_task is not None:
            return
        if send_task is not None and not send_task.done():
            return
        pump_task = asyncio.create_task(_silence_pump(client, sample_rate, stop_pump))

    try:
        while True:
            now = time.monotonic()
            if now - t_start > max_turn_s:
                break
            if not got_audio and now - t_start > max_wait_for_first_audio_s:
                break
            await _maybe_start_pump()
            item = await client.recv_audio(timeout=0.25)
            if item is None:
                if got_audio and last_audio_t is not None \
                        and (time.monotonic() - last_audio_t) > eot_silence_s:
                    break
                continue
            kind = item[0]
            if kind == "audio":
                arr = item[1]
                if arr.size > 0:
                    audio_chunks.append(arr)
                    got_audio = True
                    last_audio_t = time.monotonic()
            elif kind == "text":
                text_chunks.append(item[1])
            elif kind == "error":
                log.warning("ws reader reported: %r", item[1])
                break
    finally:
        stop_pump.set()
        if pump_task is not None:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
    return audio_chunks, text_chunks


def _persist_transcript(conv, output_dir: Path) -> None:
    import json
    d = output_dir / conv.scenario_id / f"run_{conv.run_index:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(
        json.dumps(conversation_to_dict(conv), indent=2), encoding="utf-8"
    )


async def _pump_silence_between_conversations(target, duration_s: float) -> None:
    sr = target.sample_rate
    n = int(sr * duration_s)
    zeros = np.zeros(n, dtype=np.float32)
    try:
        await target.send_audio(zeros, sr)
    except Exception as e:
        log.warning("inter-conversation silence pump failed: %s", e)


async def _run_one_conversation(scenario: Scenario, target, sim: SimulatedUser,
                                 tts, *, run_index: int, seed: int,
                                 output_dir: Path, target_ws_url: str,
                                 target_protocol: str, eot_silence_ms: int,
                                 resolved_llm: str) -> ConversationLog:
    """One N-turn conversation on an already-open target connection."""
    random.seed(seed)
    np.random.seed(seed)

    run_dir = output_dir / scenario.id / f"run_{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_obj = ConversationLog(
        scenario_id=scenario.id,
        run_index=run_index,
        seed=seed,
        resolved_user_llm=resolved_llm,
        target_ws_url=target_ws_url,
        target_protocol=target_protocol,
        started_at=time.time(),
    )

    history: list[dict] = []
    for k in range(scenario.turns):
        turn_start = time.monotonic()

        if k == 0 and scenario.opening_utterance:
            user_text = scenario.opening_utterance
        else:
            user_text = sim.next_utterance(history)
        log.info("[%s run%d turn%d] user: %s", scenario.id, run_index, k, user_text)

        user_pcm, user_sr = tts.synthesize(user_text, scenario.user_voice_ref_wav)
        user_wav = run_dir / f"user_turn_{k:02d}.wav"
        save_wav(user_wav, user_pcm, user_sr)

        send_task = asyncio.create_task(target.send_audio(user_pcm, user_sr))
        audio_chunks, text_chunks = await _collect_agent_turn(
            target, sample_rate=user_sr,
            eot_silence_s=eot_silence_ms / 1000.0,
            send_task=send_task,
        )
        if not send_task.done():
            send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, Exception):
                pass

        agent_pcm = concat(audio_chunks)
        agent_sr = target.sample_rate
        agent_wav = run_dir / f"agent_turn_{k:02d}.wav"
        save_wav(agent_wav, agent_pcm, agent_sr)

        inline_text = " ".join(text_chunks).strip()
        agent_dur = len(agent_pcm) / agent_sr if agent_sr else 0.0
        log.info("[%s run%d turn%d] agent audio %.2fs text=%r",
                 scenario.id, run_index, k, agent_dur, inline_text[:80])

        turn = TurnLog(
            index=k, user_text=user_text,
            user_wav_path=str(user_wav), user_sr=user_sr,
            agent_wav_path=str(agent_wav), agent_sr=agent_sr,
            agent_duration_s=agent_dur, inline_agent_text=inline_text,
            started_at=turn_start, ended_at=time.monotonic(),
        )
        log_obj.turns.append(turn)

        history.append({"role": "assistant", "content": user_text})
        history.append({"role": "user", "content": inline_text or "(audio response)"})

    log_obj.ended_at = time.time()
    return log_obj


async def run_session(scenario: Scenario, *, num_runs: int, base_seed: int,
                      output_dir: Path, target_ws_url: str, target_protocol: str,
                      target_codec: str | None,
                      ollama_url: str, user_llm_model: str,
                      fallback_tags: list[str], tts_name: str,
                      voices_dir: Path, eot_silence_ms: int,
                      target_backend: str = "ws",
                      target_voice_prompt: str | None = None,
                      moshi_venv_python: str | None = None,
                      inter_conversation_silence_s: float = 3.0,
                      ) -> list[ConversationLog]:
    """Open the target WS ONCE, run `num_runs` back-to-back conversations with
    the same system prompt, close once at the end. The persistent connection
    avoids the ~30-60s system-prompt reload the Moshi server runs on every new
    connection, and matches the user's intended semantics: prompt the model
    once, end the conversation, start a new one on the same socket.
    """
    user_client = OllamaChatClient(ollama_url, user_llm_model, fallback_tags=fallback_tags)
    sim = SimulatedUser(
        client=user_client,
        persona=scenario.user_persona,
        target_role=scenario.rubric.target_role,
        must_do=scenario.rubric.must_do,
    )
    tts = build_tts(tts_name, voices_dir)

    try:
        resolved_llm = user_client.resolve_model()
    except Exception as e:
        log.warning("user LLM unavailable: %s", e)
        resolved_llm = "(unavailable)"

    if target_backend == "offline":
        target = OfflineMoshiClient(
            ws_url=target_ws_url, protocol=target_protocol,
            codec_name=target_codec,
            moshi_venv_python=moshi_venv_python,
        )
    else:
        target = MoshiWSClient(target_ws_url, protocol=target_protocol,
                               codec_name=target_codec)
    await target.open(system_prompt=scenario.system_prompt_for_target,
                      voice_prompt_path=target_voice_prompt)

    results: list[ConversationLog] = []
    try:
        for run_index in range(num_runs):
            seed = base_seed + run_index
            log.info("=== [%s] starting run %d/%d on persistent WS ===",
                     scenario.id, run_index + 1, num_runs)
            try:
                conv = await _run_one_conversation(
                    scenario, target, sim, tts,
                    run_index=run_index, seed=seed,
                    output_dir=output_dir,
                    target_ws_url=target_ws_url,
                    target_protocol=target_protocol,
                    eot_silence_ms=eot_silence_ms,
                    resolved_llm=resolved_llm,
                )
                # Persist transcript immediately so an interrupted session still
                # leaves valid per-run artifacts that phase-2 can consume.
                _persist_transcript(conv, output_dir)
                results.append(conv)
            except Exception as e:
                log.exception("[%s] run %d failed: %s", scenario.id, run_index, e)
                continue
            if run_index < num_runs - 1 and inter_conversation_silence_s > 0:
                await _pump_silence_between_conversations(
                    target, inter_conversation_silence_s
                )
    finally:
        await target.close()
    return results


async def run_scenario(scenario: Scenario, *, run_index: int, seed: int,
                       output_dir: Path, target_ws_url: str, target_protocol: str,
                       target_codec: str | None,
                       ollama_url: str, user_llm_model: str,
                       fallback_tags: list[str], tts_name: str,
                       voices_dir: Path, eot_silence_ms: int,
                       target_backend: str = "ws",
                       target_voice_prompt: str | None = None,
                       moshi_venv_python: str | None = None) -> ConversationLog:
    """Back-compat: run ONE conversation with its own open/close. Prefer
    `run_session` for multi-run scenarios to avoid server-side prompt reload.
    """
    results = await run_session(
        scenario, num_runs=1, base_seed=seed,
        output_dir=output_dir,
        target_ws_url=target_ws_url, target_protocol=target_protocol,
        target_codec=target_codec, ollama_url=ollama_url,
        user_llm_model=user_llm_model, fallback_tags=fallback_tags,
        tts_name=tts_name, voices_dir=voices_dir,
        eot_silence_ms=eot_silence_ms, target_backend=target_backend,
        target_voice_prompt=target_voice_prompt,
        moshi_venv_python=moshi_venv_python,
    )
    if not results:
        raise RuntimeError(f"run_scenario: no conversation produced for {scenario.id}")
    conv = results[0]
    conv.run_index = run_index
    conv.seed = seed
    return conv


def conversation_to_dict(c: ConversationLog) -> dict[str, Any]:
    return {
        **{k: v for k, v in asdict(c).items() if k != "turns"},
        "turns": [asdict(t) for t in c.turns],
    }


async def run_all(scenarios: list[Scenario], *, num_runs: int, base_seed: int,
                  **kwargs) -> list[ConversationLog]:
    results: list[ConversationLog] = []
    for scenario in scenarios:
        for i in range(num_runs):
            seed = base_seed + i
            try:
                conv = await run_scenario(scenario, run_index=i, seed=seed, **kwargs)
                results.append(conv)
            except Exception as e:
                log.exception("scenario %s run %d failed: %s", scenario.id, i, e)
    return results
