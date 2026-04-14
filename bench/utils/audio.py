"""PCM helpers: RMS/silence detection, WAV I/O, concatenation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def rms(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))


def is_silent(pcm: np.ndarray, threshold: float = 1e-3) -> bool:
    return rms(pcm) < threshold


def concat(chunks: Iterable[np.ndarray]) -> np.ndarray:
    arrs = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
    if not arrs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(arrs)


def save_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
    try:
        import soundfile as sf
        sf.write(str(path), arr, sr, subtype="PCM_16")
        return
    except ImportError:
        pass
    try:
        import sphn
        sphn.write_wav(str(path), arr, sr)
        return
    except ImportError:
        pass
    import wave
    pcm16 = np.clip(arr * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        arr, sr = sf.read(str(path), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), int(sr)
    except ImportError:
        pass
    import wave
    with wave.open(str(path), "rb") as r:
        sr = r.getframerate()
        frames = r.readframes(r.getnframes())
    arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    return arr, sr
