"""Async full-duplex WS audio client.

Connects to a speech-to-speech server (Personaplex/Moshi/etc.), streams
outgoing audio as encoded frames, and pushes incoming frames into a queue
for the orchestrator to drain.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np

from .protocols import Dialect, get_dialect


class TargetClient(Protocol):
    async def open(self, *, system_prompt: str | None = None,
                   voice_prompt_path: str | None = None) -> None: ...
    async def send_audio(self, pcm: np.ndarray, sr: int) -> None: ...
    async def recv_audio(self, timeout: float): ...
    async def close(self) -> None: ...


class MoshiWSClient:
    """Full-duplex WS client for Moshi/PersonaPlex-style servers.

    Outgoing: `send_audio(pcm, sr)` -> dialect encodes + frames PCM into WS
    messages and ships them in order. Caller is responsible for pacing.

    Incoming: `recv_audio(timeout)` returns the next parsed message
    (`("audio", np.ndarray, sr)`, `("text", str)`, or `None` on timeout).
    An internal task drains the WS socket into an asyncio.Queue; the
    orchestrator's silence-timeout heuristic feeds off `None` returns.
    """

    def __init__(self, ws_url: str, protocol: str = "moshi",
                 codec_name: str | None = None, seed: int | None = None):
        self.ws_url = ws_url
        self.dialect: Dialect = get_dialect(protocol, codec_name)
        self.seed = seed
        self._ws = None
        self._queue: asyncio.Queue | None = None
        self._reader_task: asyncio.Task | None = None
        self._ready_event: asyncio.Event | None = None
        self._sample_rate = getattr(self.dialect, "SAMPLE_RATE", None) or self.dialect.codec.sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def open(self, *, system_prompt: str | None = None,
                   voice_prompt_path: str | None = None,
                   handshake_timeout_s: float = 180.0,
                   handshake_retries: int = 1) -> None:
        import ssl
        import websockets
        final_url = self.dialect.prepare_url(
            self.ws_url,
            system_prompt=system_prompt,
            voice_prompt=voice_prompt_path,
            seed=self.seed,
        )
        connect_kwargs = {"max_size": None}
        if final_url.startswith("wss://"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            connect_kwargs["ssl"] = ctx

        last_err: Exception | None = None
        for attempt in range(handshake_retries + 1):
            try:
                self._ws = await websockets.connect(final_url, **connect_kwargs)
                self._queue = asyncio.Queue()
                self._ready_event = asyncio.Event()
                self._reader_task = asyncio.create_task(self._reader())
                await self.dialect.handshake(
                    self._ws,
                    system_prompt=system_prompt,
                    voice_prompt=voice_prompt_path,
                )
                await self._wait_for_ready(timeout=handshake_timeout_s)
                return
            except TimeoutError as e:
                last_err = e
                await self._teardown_silently()
                if attempt < handshake_retries:
                    continue
                raise
        assert last_err is None  # unreachable

    async def _teardown_silently(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._queue = None
        self._reader_task = None
        self._ready_event = None

    async def _reader(self) -> None:
        assert self._ws is not None and self._queue is not None
        try:
            async for msg in self._ws:
                parsed = self.dialect.parse_message(msg)
                if self._ready_event is not None and not self._ready_event.is_set():
                    self._ready_event.set()
                if parsed is None:
                    continue
                if parsed[0] in ("keepalive", "buffered"):
                    continue
                await self._queue.put(parsed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._queue.put(("error", exc))

    async def _wait_for_ready(self, timeout: float) -> None:
        assert self._ready_event is not None
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"target server never sent a handshake byte within {timeout}s; "
                f"likely still loading system prompts or failed to start"
            )

    async def send_audio(self, pcm: np.ndarray, sr: int,
                         realtime_pace: bool = True) -> None:
        assert self._ws is not None
        frame_ms = getattr(self.dialect, "FRAME_SAMPLES", None)
        if frame_ms is not None:
            frame_ms = 1000.0 * frame_ms / getattr(self.dialect, "SAMPLE_RATE", 24000)
        else:
            frame_ms = getattr(self.dialect.codec, "frame_ms", 80.0)
        pace = (frame_ms / 1000.0) if realtime_pace else 0.0
        for frame_msg in self.dialect.encode_pcm(pcm, sr):
            await self._ws.send(frame_msg)
            if pace > 0:
                await asyncio.sleep(pace)

    async def send_text(self, text: str) -> None:
        assert self._ws is not None
        await self._ws.send(self.dialect.encode_text(text))

    async def recv_audio(self, timeout: float = 0.1):
        assert self._queue is not None
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            await self._ws.close()
        try:
            await self.dialect.close()
        except Exception:
            pass
        self._ws = None
        self._queue = None
        self._reader_task = None
