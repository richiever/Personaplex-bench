"""Lazy-loaded Whisper transcriber. Dual-channel (user + agent) helper.

One shared model instance across all turns of all scenarios in a run.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)


class WhisperTranscriber:
    def __init__(self, model: str = "base", device: str = "cuda"):
        self.model_name = model
        self.device = device
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as e:
            raise RuntimeError("openai-whisper not installed. pip install openai-whisper") from e
        log.info("loading whisper model=%s device=%s", self.model_name, self.device)
        try:
            self._model = whisper.load_model(self.model_name, device=self.device)
        except Exception:
            log.warning("whisper load on %s failed, falling back to cpu", self.device)
            self._model = whisper.load_model(self.model_name, device="cpu")
            self.device = "cpu"
        return self._model

    def transcribe_wav(self, wav_path: Path) -> str:
        p = Path(wav_path)
        if not p.exists() or p.stat().st_size == 0:
            return ""
        m = self._lazy()
        try:
            out = m.transcribe(str(p), fp16=(self.device != "cpu"))
            return out.get("text", "").strip()
        except Exception as e:
            log.warning("whisper transcribe failed for %s: %s", p, e)
            return ""

    def transcribe_conversation(self, conv) -> None:
        """Mutates each TurnLog: fills user_transcript + agent_transcript."""
        for turn in conv.turns:
            turn.user_transcript = self.transcribe_wav(Path(turn.user_wav_path))
            turn.agent_transcript = self.transcribe_wav(Path(turn.agent_wav_path))
