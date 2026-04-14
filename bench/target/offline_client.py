"""Offline `moshi.offline` subprocess target client.

Fallback for when the live WS path has issues. Same Personaplex weights;
instead of a duplex WS stream, each turn:
  1. Save user PCM to a temp WAV.
  2. Invoke `python -m moshi.offline` with the scenario's text-prompt and
     voice-prompt -> writes an output WAV + text JSON.
  3. Load the WAV/text and surface them via the same API as MoshiWSClient.

The orchestrator sees the same `recv_audio` / `send_audio` surface, so
switching backends is a CLI flag.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..utils.audio import load_wav, save_wav
from ..utils.logging import get_logger

log = get_logger(__name__)


class OfflineMoshiClient:
    def __init__(self, ws_url: str = "", protocol: str = "personaplex",
                 codec_name: str | None = None, seed: int | None = None,
                 *, moshi_venv_python: str | None = None,
                 voice_prompt_dir: str | None = None,
                 sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.seed = seed
        self.moshi_python = (
            moshi_venv_python
            or os.environ.get("MOSHI_PYTHON")
            or "/home/ubuntu/personaplex/moshi/.venv/bin/python"
        )
        self.voice_prompt_dir = voice_prompt_dir
        self._system_prompt: str | None = None
        self._voice_prompt_path: str | None = None
        self._out_queue: asyncio.Queue = asyncio.Queue()
        self._tmpdir: Path | None = None

    async def open(self, *, system_prompt: str | None = None,
                   voice_prompt_path: str | None = None) -> None:
        self._system_prompt = system_prompt
        self._voice_prompt_path = voice_prompt_path
        self._tmpdir = Path(tempfile.mkdtemp(prefix="bench_offline_"))
        log.info("offline target ready; tmpdir=%s", self._tmpdir)

    async def send_audio(self, pcm: np.ndarray, sr: int,
                         realtime_pace: bool = True) -> None:
        assert self._tmpdir is not None
        in_wav = self._tmpdir / "in.wav"
        out_wav = self._tmpdir / "out.wav"
        out_json = self._tmpdir / "out.json"
        save_wav(in_wav, pcm, sr)

        cmd = [
            self.moshi_python, "-m", "moshi.offline",
            "--input-wav", str(in_wav),
            "--output-wav", str(out_wav),
            "--output-text", str(out_json),
        ]
        if self._voice_prompt_path:
            cmd += ["--voice-prompt", self._voice_prompt_path]
        if self._system_prompt:
            cmd += ["--text-prompt", self._system_prompt]
        if self.seed is not None and self.seed >= 0:
            cmd += ["--seed", str(self.seed)]

        log.info("offline target: launching moshi.offline (input=%.2fs)",
                 len(pcm) / sr)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "NO_TORCH_COMPILE": "1"},
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            log.warning("moshi.offline exit=%d stderr=%s",
                        proc.returncode, stderr.decode(errors="replace")[-400:])
            await self._out_queue.put(("error",
                f"moshi.offline exit {proc.returncode}: {stderr.decode(errors='replace')[-200:]}"))
            return
        if not out_wav.exists():
            log.warning("moshi.offline produced no output wav; stderr=%s",
                        stderr.decode(errors="replace")[-400:])
            await self._out_queue.put(("error", "no output wav"))
            return
        out_pcm, out_sr = load_wav(out_wav)
        await self._out_queue.put(("audio", out_pcm.astype(np.float32), out_sr))
        if out_json.exists():
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                text = data.get("text") if isinstance(data, dict) else None
                if isinstance(data, list):
                    text = "".join(str(x.get("text", "")) for x in data if isinstance(x, dict))
                if text:
                    await self._out_queue.put(("text", text))
            except Exception as e:
                log.warning("offline out_json parse: %s", e)

    async def recv_audio(self, timeout: float = 0.1):
        try:
            return await asyncio.wait_for(self._out_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        if self._tmpdir is not None and self._tmpdir.exists():
            for p in self._tmpdir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                self._tmpdir.rmdir()
            except Exception:
                pass
            self._tmpdir = None
