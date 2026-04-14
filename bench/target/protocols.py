"""Codec + dialect abstractions for full-duplex audio WS servers.

The Moshi / PersonaPlex on-wire protocol (verified against
NVIDIA/personaplex @ moshi/server.py):

- WS endpoint: `/api/chat?text_prompt=...&voice_prompt=...&seed=...`
- Client -> Server binary messages:
    `\\x01 + <opus_ogg_stream_bytes>`   -- audio chunk
- Server -> Client binary messages:
    `\\x00`                              -- keepalive
    `\\x01 + <opus_ogg_stream_bytes>`   -- audio chunk
    `\\x02 + <utf-8>`                    -- streamed text piece

Audio is a continuous Ogg-Opus stream (sphn.OpusStreamWriter/Reader), not
individual Opus packets. Chunk boundaries in WS messages are arbitrary;
the receiver's reader handles packet reassembly.

For model-agnostic use, a RawPcmCodec + per-frame dialect is also provided
for tests / servers that accept int16 PCM frames directly.
"""

from __future__ import annotations

import urllib.parse
from typing import Iterable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Per-frame codecs (tests, trivial servers)
# ---------------------------------------------------------------------------


class Codec(Protocol):
    name: str
    frame_ms: float
    sample_rate: int
    frame_samples: int

    def encode_frame(self, pcm_frame: np.ndarray) -> bytes: ...
    def decode_frame(self, encoded: bytes) -> np.ndarray: ...


class RawPcmCodec:
    name = "pcm"

    def __init__(self, sample_rate: int = 24000, frame_ms: float = 80.0):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)

    def encode_frame(self, pcm_frame: np.ndarray) -> bytes:
        if pcm_frame.dtype != np.int16:
            pcm_frame = np.clip(pcm_frame * 32767.0, -32768, 32767).astype(np.int16)
        return pcm_frame.tobytes()

    def decode_frame(self, encoded: bytes) -> np.ndarray:
        return np.frombuffer(encoded, dtype=np.int16).astype(np.float32) / 32767.0


class OpusCodec:
    """Per-packet Opus codec (opuslib). Not used by Moshi servers."""

    name = "opus"

    def __init__(self, sample_rate: int = 24000, frame_ms: float = 20.0,
                 bitrate: int = 48000):
        import opuslib
        valid = {2.5, 5.0, 10.0, 20.0, 40.0, 60.0}
        if frame_ms not in valid:
            raise ValueError(f"Opus frame_ms must be in {valid}, got {frame_ms}")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._enc = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_VOIP)
        self._enc.bitrate = bitrate
        self._dec = opuslib.Decoder(sample_rate, 1)

    def encode_frame(self, pcm_frame: np.ndarray) -> bytes:
        if pcm_frame.dtype != np.int16:
            pcm_frame = np.clip(pcm_frame * 32767.0, -32768, 32767).astype(np.int16)
        return self._enc.encode(pcm_frame.tobytes(), self.frame_samples)

    def decode_frame(self, encoded: bytes) -> np.ndarray:
        raw = self._dec.decode(encoded, self.frame_samples)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def make_codec(name: str, sample_rate: int = 24000, frame_ms: float | None = None) -> Codec:
    if name == "pcm":
        return RawPcmCodec(sample_rate, frame_ms if frame_ms is not None else 80.0)
    if name == "opus":
        return OpusCodec(sample_rate, frame_ms if frame_ms is not None else 20.0)
    raise ValueError(f"unknown codec: {name}")


def _resample(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return pcm
    try:
        import librosa
        return librosa.resample(pcm.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)
    except ImportError:
        ratio = dst_sr / src_sr
        idx = (np.arange(int(len(pcm) * ratio)) / ratio).astype(np.int64)
        idx = np.clip(idx, 0, len(pcm) - 1)
        return pcm[idx]


# ---------------------------------------------------------------------------
# Dialects
# ---------------------------------------------------------------------------


class Dialect:
    """Per-frame dialect used by tests / simple servers.

    Client -> Server and Server -> Client both use `tag + payload`:
        \\x01 = one encoded audio frame
        \\x02 = utf-8 text
    """

    AUDIO_TAG = 0x01
    TEXT_TAG = 0x02

    def __init__(self, codec: Codec):
        self.codec = codec

    def prepare_url(self, base_url: str, *, system_prompt: str | None = None,
                    voice_prompt: str | None = None, seed: int | None = None) -> str:
        return base_url

    async def handshake(self, ws, *, system_prompt: str | None = None,
                        voice_prompt: str | None = None) -> None:
        pass

    async def close(self) -> None:
        pass

    def encode_pcm(self, pcm: np.ndarray, sr: int) -> Iterable[bytes]:
        if sr != self.codec.sample_rate:
            pcm = _resample(pcm, sr, self.codec.sample_rate)
        fs = self.codec.frame_samples
        n = (len(pcm) // fs) * fs
        tag = bytes([self.AUDIO_TAG])
        for i in range(0, n, fs):
            yield tag + self.codec.encode_frame(pcm[i:i + fs])

    def encode_text(self, text: str) -> bytes:
        return bytes([self.TEXT_TAG]) + text.encode("utf-8")

    def parse_message(self, msg) -> tuple | None:
        if not isinstance(msg, (bytes, bytearray)) or len(msg) < 1:
            return None
        tag = msg[0]
        payload = bytes(msg[1:])
        if tag == self.AUDIO_TAG:
            return ("audio", self.codec.decode_frame(payload), self.codec.sample_rate)
        if tag == self.TEXT_TAG:
            return ("text", payload.decode("utf-8", errors="replace"))
        if tag == 0x00:
            return ("keepalive",)
        return ("other", tag, payload)


class MoshiDialect(Dialect):
    """Moshi / PersonaPlex dialect.

    Uses sphn.OpusStreamWriter/Reader for a continuous Ogg-Opus stream.
    URL query params carry system prompt, voice prompt, seed.
    """

    SAMPLE_RATE = 24000
    FRAME_SAMPLES = 1920  # 80 ms at 24 kHz; allowed by sphn
    WS_PATH = "/api/chat"

    def __init__(self, codec_name: str | None = None):
        # codec_name is informational; Moshi always uses sphn opus streams.
        self.codec = RawPcmCodec(sample_rate=self.SAMPLE_RATE,
                                 frame_ms=1000.0 * self.FRAME_SAMPLES / self.SAMPLE_RATE)
        self._writer = None
        self._reader = None

    def _lazy_streams(self):
        if self._writer is not None:
            return
        import sphn
        self._writer = sphn.OpusStreamWriter(self.SAMPLE_RATE)
        self._reader = sphn.OpusStreamReader(self.SAMPLE_RATE)

    def prepare_url(self, base_url: str, *, system_prompt: str | None = None,
                    voice_prompt: str | None = None, seed: int | None = None) -> str:
        parsed = urllib.parse.urlparse(base_url)
        path = parsed.path or "/"
        if not path.endswith(self.WS_PATH):
            path = self.WS_PATH if path == "/" else path.rstrip("/") + self.WS_PATH
        # NOTE: personaplex server has a bug on handle_chat line 171 where
        # `"seed" in request.query` is tested but `request["seed"]` is read
        # (the query attribute is dropped). Supplying `seed=` therefore raises
        # KeyError server-side. We omit it from the query entirely; the server
        # uses a random seed when absent, which is fine for multi-run variance.
        q = {
            "text_prompt": system_prompt or "",
            "voice_prompt": voice_prompt or "",
        }
        qs = urllib.parse.urlencode(q)
        return urllib.parse.urlunparse(parsed._replace(path=path, query=qs))

    async def handshake(self, ws, *, system_prompt=None, voice_prompt=None) -> None:
        # All conditioning is in the URL query string; nothing to send here.
        self._lazy_streams()

    async def close(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        self._writer = None
        self._reader = None

    def encode_pcm(self, pcm: np.ndarray, sr: int) -> Iterable[bytes]:
        self._lazy_streams()
        if sr != self.SAMPLE_RATE:
            pcm = _resample(pcm, sr, self.SAMPLE_RATE)
        pcm = pcm.astype(np.float32)
        fs = self.FRAME_SAMPLES
        n = (len(pcm) // fs) * fs
        tag = bytes([self.AUDIO_TAG])
        for i in range(0, n, fs):
            self._writer.append_pcm(pcm[i:i + fs])
            data = self._writer.read_bytes()
            if data:
                yield tag + bytes(data)

    def parse_message(self, msg) -> tuple | None:
        if not isinstance(msg, (bytes, bytearray)) or len(msg) < 1:
            return None
        self._lazy_streams()
        tag = msg[0]
        payload = bytes(msg[1:])
        if tag == self.AUDIO_TAG:
            self._reader.append_bytes(payload)
            pcm = self._reader.read_pcm()
            arr = np.asarray(pcm, dtype=np.float32)
            if arr.size == 0:
                return ("buffered",)
            return ("audio", arr, self.SAMPLE_RATE)
        if tag == self.TEXT_TAG:
            return ("text", payload.decode("utf-8", errors="replace"))
        if tag == 0x00:
            return ("keepalive",)
        return ("other", tag, payload)


class PersonaPlexDialect(MoshiDialect):
    """Same wire protocol as Moshi."""


def get_dialect(name: str, codec_name: str | None = None) -> Dialect:
    if name == "moshi":
        return MoshiDialect(codec_name)
    if name == "personaplex":
        return PersonaPlexDialect(codec_name)
    if name == "custom":
        return Dialect(make_codec(codec_name or "pcm"))
    raise ValueError(f"unknown dialect: {name}")
