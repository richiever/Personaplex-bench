"""F5-TTS adapter. Voice-cloning from a reference WAV.

F5-TTS is imported lazily so the rest of the bench works in environments
where F5-TTS isn't installed (e.g. judge-only or mock-WS smoke tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class TTSUnavailable(RuntimeError):
    pass


class F5TTSClient:
    def __init__(self, voices_dir: Path):
        self.voices_dir = Path(voices_dir)
        self._tts = None

    def _lazy(self):
        if self._tts is not None:
            return self._tts
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise TTSUnavailable(
                "F5-TTS not installed. pip install f5-tts"
            ) from e
        self._tts = F5TTS()
        return self._tts

    def _resolve_ref(self, ref_voice_wav: str | Path) -> Path:
        p = Path(ref_voice_wav)
        if not p.is_absolute():
            candidate = self.voices_dir / p.name if p.parent == Path(".") else Path(ref_voice_wav)
            if candidate.exists():
                p = candidate
        if not p.exists():
            raise FileNotFoundError(f"voice reference not found: {ref_voice_wav}")
        return p

    def synthesize(self, text: str, ref_voice_wav: str | Path,
                   ref_text: str = "") -> Tuple[np.ndarray, int]:
        tts = self._lazy()
        ref_path = self._resolve_ref(ref_voice_wav)
        wav, sr, _ = tts.infer(
            ref_file=str(ref_path),
            ref_text=ref_text,
            gen_text=text,
            remove_silence=True,
            speed=1.0,
        )
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr, int(sr)


class MockTTSClient:
    """Generates a short sine burst per call. For tests where F5-TTS is absent."""

    def __init__(self, voices_dir: Path, sample_rate: int = 24000):
        self.voices_dir = Path(voices_dir)
        self.sample_rate = sample_rate

    def synthesize(self, text: str, ref_voice_wav: str | Path,
                   ref_text: str = "") -> Tuple[np.ndarray, int]:
        dur = max(0.3, min(3.0, len(text) * 0.06))
        n = int(self.sample_rate * dur)
        freq = 220.0 + 40.0 * (abs(hash(text)) % 8)
        t = np.arange(n) / self.sample_rate
        pcm = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        env = np.minimum(np.linspace(0, 1, min(240, n)), 1.0)
        pcm[:len(env)] *= env
        pcm[-len(env):] *= env[::-1]
        return pcm, self.sample_rate


def build_tts(name: str, voices_dir: Path):
    if name == "f5-tts":
        return F5TTSClient(voices_dir)
    if name == "mock":
        return MockTTSClient(voices_dir)
    raise ValueError(f"unknown TTS: {name}")
