"""Qwen2-Audio-based per-turn audio judge.

Phase-2 of the bench. VRAM-heavy; lazy-loaded. Runnable standalone via
`python -m bench.judge.audio --results-dir ./results` so the judge pass can
be deferred to another host or another run when the capture-phase GPU is
needed for the target model.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..utils.audio import load_wav
from ..utils.logging import get_logger

log = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "audio_judge.md"


def _coerce_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


class AudioJudgeUnavailable(RuntimeError):
    pass


class AudioJudge:
    def __init__(self, model: str = "Qwen/Qwen2-Audio-7B-Instruct",
                 device_map: str = "auto", dtype: str = "bfloat16",
                 max_new_tokens: int = 256):
        self.model_name = model
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._prompt_tmpl: str | None = None

    def _prompt(self) -> str:
        if self._prompt_tmpl is None:
            self._prompt_tmpl = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_tmpl

    def _lazy(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ImportError as e:
            raise AudioJudgeUnavailable(
                "transformers with Qwen2-Audio support required. "
                "pip install 'transformers>=4.45' torch librosa"
            ) from e
        log.info("loading audio judge model=%s", self.model_name)
        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, device_map=self.device_map, torch_dtype=torch_dtype
        )
        self._torch = torch

    def judge_wav(self, wav_path: Path) -> dict[str, Any]:
        p = Path(wav_path)
        if not p.exists() or p.stat().st_size == 0:
            return {"speech_quality": 0, "artifacts": ["silence"],
                    "turn_taking": "cutoff", "notes": "empty/missing file"}
        self._lazy()
        target_sr = self._processor.feature_extractor.sampling_rate
        audio, sr = load_wav(p)
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                ratio = target_sr / sr
                import numpy as np
                idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(np.int64)
                idx = idx.clip(0, len(audio) - 1)
                audio = audio[idx]

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": self._prompt()},
            ]},
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=text, audios=[audio], sampling_rate=target_sr,
                                 return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        out_text = self._processor.batch_decode(gen, skip_special_tokens=True)[0]
        parsed = _coerce_json(out_text)
        if not parsed:
            log.warning("audio judge non-JSON output: %r", out_text[:200])
            return {"speech_quality": None, "artifacts": [], "turn_taking": "ok",
                    "notes": out_text.strip()[:160], "_raw": out_text}
        parsed.setdefault("speech_quality", None)
        parsed.setdefault("artifacts", [])
        parsed.setdefault("turn_taking", "ok")
        parsed.setdefault("notes", "")
        return parsed

    def judge_conversation(self, conv) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for t in conv.turns:
            res = self.judge_wav(Path(t.agent_wav_path))
            res["turn"] = t.index
            results.append(res)
        return results


def _load_runs(results_dir: Path) -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    for tr in results_dir.glob("*/run_*/transcript.json"):
        try:
            data = json.loads(tr.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("skip %s: %s", tr, e)
            continue
        found.append((tr, data))
    return found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.judge.audio")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    args = p.parse_args(argv)

    judge = AudioJudge(model=args.model)
    runs = _load_runs(args.results_dir)
    if not runs:
        print(f"no runs under {args.results_dir}")
        return 1
    for tr_path, data in runs:
        log.info("judging %s", tr_path)
        per_turn: list[dict] = []
        for t in data.get("turns", []):
            wav = t.get("agent_wav_path")
            if not wav:
                continue
            res = judge.judge_wav(Path(wav))
            res["turn"] = t.get("index", len(per_turn))
            per_turn.append(res)
        data["audio_judge"] = per_turn
        tr_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
