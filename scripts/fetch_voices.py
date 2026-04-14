"""Populate `voices/` with reference WAVs usable by F5-TTS.

Strategy, in order:
1. Copy sample ref WAVs shipped with F5-TTS (if installed).
2. Fall back to `librosa.ex()` samples (CC-licensed speech fragments).
3. Last-resort synthetic buzz so the pipeline has something to point at
   (F5-TTS quality will be poor -- replace with real voice ASAP).

Replace any placeholder WAV in `voices/` with a real 5-15s single-speaker clip
of the worker/co-worker persona you want to simulate. F5-TTS voice-clones from
the provided reference at inference time.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
TARGET_NAMES = ["barista_1.wav", "worker_1.wav", "customer_1.wav"]


def _from_f5_tts() -> Path | None:
    try:
        import f5_tts
    except ImportError:
        return None
    file_attr = getattr(f5_tts, "__file__", None)
    paths: list[Path] = []
    if file_attr:
        paths.append(Path(file_attr).parent)
    for p in getattr(f5_tts, "__path__", []) or []:
        paths.append(Path(p))
    for pkg_dir in paths:
        for pat in ("*_ref_en*.wav", "*ref_en*.wav", "*ref*.wav"):
            hits = list(pkg_dir.rglob(pat))
            if hits:
                return hits[0]
    return None


def _from_librosa() -> Path | None:
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return None
    for name in ("libri1", "libri2", "libri3", "trumpet", "brahms"):
        try:
            path = librosa.example(name)
            y, sr = librosa.load(path, sr=24000, mono=True)
            if len(y) >= 24000 * 3:
                out = Path("_tmp_ref.wav")
                sf.write(str(out), y[: 24000 * 10], 24000, subtype="PCM_16")
                return out
        except Exception:
            continue
    return None


def _synthetic(out: Path, sr: int = 24000, dur: float = 6.0) -> None:
    t = np.arange(int(sr * dur)) / sr
    f0 = 140.0 + 20.0 * np.sin(2 * np.pi * 0.7 * t)
    sig = 0.3 * np.sin(2 * np.pi * f0 * t)
    for h, a in [(2, 0.15), (3, 0.1), (4, 0.07)]:
        sig += a * np.sin(2 * np.pi * f0 * h * t)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    sig *= env * 0.3
    sig = sig.astype(np.float32)
    try:
        import soundfile as sf
        sf.write(str(out), sig, sr, subtype="PCM_16")
    except ImportError:
        import wave
        pcm16 = np.clip(sig * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16.tobytes())


def main() -> int:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    src = _from_f5_tts() or _from_librosa()
    if src is None:
        print("[fetch_voices] no F5-TTS/librosa samples available; writing synthetic placeholders.",
              file=sys.stderr)
    for name in TARGET_NAMES:
        dst = VOICES_DIR / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[fetch_voices] {dst} already exists; skipping.")
            continue
        if src is not None:
            shutil.copy(src, dst)
            print(f"[fetch_voices] copied {src} -> {dst}")
        else:
            _synthetic(dst)
            print(f"[fetch_voices] synthesized placeholder -> {dst}")
    if src is not None and Path("_tmp_ref.wav").exists():
        Path("_tmp_ref.wav").unlink()
    print(f"\nVoice refs in {VOICES_DIR}. Replace placeholders with real 5-15s "
          f"single-speaker clips for best TTS quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
