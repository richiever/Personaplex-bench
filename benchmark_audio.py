#!/usr/bin/env python3
"""
Personaplex-bench: Autoresearcher benchmark for PersonaPlex speech-to-speech models.

Two evaluation modes:
  Mode A (token-eval):      Teacher-forcing accuracy/loss on held-out .pt files
  Mode B (generation-eval): Free streaming generation → audio quality + coherence checks

Usage:
  python benchmark_audio.py --mode both --eval-files test_001.pt --device cuda
  python benchmark_audio.py --compare baseline.json current.json
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lazy imports for heavy libraries (only loaded when needed)
# ---------------------------------------------------------------------------

def _import_librosa():
    import librosa
    return librosa

def _import_whisper():
    import whisper
    return whisper

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class TokenEvalResult:
    file: str = ""
    text_accuracy: float = 0.0
    text_loss: float = 0.0
    text_perplexity: float = 0.0
    semantic_accuracy: float = 0.0
    semantic_loss: float = 0.0
    acoustic_loss: float = 0.0
    total_weighted_loss: float = 0.0


@dataclass
class FrameRecord:
    step_idx: int = 0
    text_token: int = 0
    audio_tokens: list = field(default_factory=list)
    is_pad: bool = True
    consecutive_pad_count: int = 0
    wall_time_ms: float = 0.0
    text_entropy: float = 0.0
    audio_entropy_cb0: float = 0.0


@dataclass
class GenerationSampleResult:
    file: str = ""
    system_prompt: str = ""
    seed: int = 0
    transcription: str = ""
    word_count: int = 0
    domain_keywords_found: list = field(default_factory=list)
    text_audio_word_overlap: float = 0.0
    utmos: float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_flatness_max_window: float = 0.0
    zcr_mean: float = 0.0
    energy_cv: float = 0.0
    token_entropy_mean: float = 0.0
    token_entropy_min_window: float = 0.0
    token_repeat_max_frames: int = 0
    token_unique_ratio: float = 0.0
    response_latency_s: float = 0.0
    silence_ratio: float = 0.0
    speech_segments: int = 0
    longest_speech_s: float = 0.0
    rtf: float = 0.0
    degenerate: bool = False
    degenerate_reasons: list = field(default_factory=list)
    # Role adherence (lightweight phrase-based check)
    expected_role: str = ""
    role_adherence: bool = True
    role_inversion_phrases: list = field(default_factory=list)
    # Artifact paths (for cron judge to review)
    output_wav_path: str = ""
    output_transcript_path: str = ""
    diagnostic_png_path: str = ""
    # Per-segment degeneration tracking
    segment_verdicts: list = field(default_factory=list)
    # LLM judge verdict (filled by judge_cron.py, not by benchmark)
    judge_verdict: str = ""  # "pending", "keep", "discard"
    judge_reason: str = ""
    pass_all: bool = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "coffee", "latte", "americano", "espresso", "milk", "oat", "order", "drink",
    "menu", "cup", "size", "sugar", "cream", "brew", "morning", "shop", "barista",
    "customer", "cafe", "cappuccino", "mocha", "decaf", "iced", "hot", "grande",
    "venti", "tall", "black", "whip", "syrup", "shot", "beans",
    # Retail/service keywords
    "appliance", "repair", "dishwasher", "part", "stock", "labor", "cost",
    "replacement", "warranty", "service", "technician", "schedule",
}

SILENCE_TOKEN_CB0 = 948
PAD_TOKEN = 3
FRAME_RATE = 12.5  # Hz
SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Mode A: Token Eval
# ---------------------------------------------------------------------------

def token_eval(lm, eval_files: list[Path], device: str) -> list[TokenEvalResult]:
    """Run teacher-forcing evaluation on .pt files using forward_train()."""
    results = []
    lm.eval()

    for pt_path in eval_files:
        codes = torch.load(pt_path, weights_only=True).unsqueeze(0).to(device)
        result = TokenEvalResult(file=pt_path.name)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = lm.forward_train(codes)

        # Text accuracy and loss
        text_target = codes[:, 0, :]
        valid_text = output.text_mask[:, 0].bool() & (text_target != -1)
        if valid_text.any():
            text_preds = output.text_logits[:, 0][valid_text].argmax(dim=-1)
            text_targets = text_target[valid_text]
            result.text_accuracy = (text_preds == text_targets).float().mean().item()
            result.text_loss = F.cross_entropy(
                output.text_logits[:, 0][valid_text], text_targets
            ).item()
            result.text_perplexity = math.exp(min(result.text_loss, 20.0))

        # Semantic codebook (cb0) accuracy and loss
        audio_target = codes[:, 1:9, :]
        valid_sem = output.mask[:, 0].bool()
        if valid_sem.any():
            sem_preds = output.logits[:, 0][valid_sem].argmax(dim=-1)
            sem_targets = audio_target[:, 0][valid_sem]
            result.semantic_accuracy = (sem_preds == sem_targets).float().mean().item()
            result.semantic_loss = F.cross_entropy(
                output.logits[:, 0][valid_sem], sem_targets
            ).item()

        # Acoustic codebooks (cb1-7) loss
        acoustic_losses = []
        for k in range(1, 8):
            valid = output.mask[:, k].bool()
            if valid.any():
                loss_k = F.cross_entropy(
                    output.logits[:, k][valid], audio_target[:, k][valid]
                ).item()
                acoustic_losses.append(loss_k)
        if acoustic_losses:
            result.acoustic_loss = sum(acoustic_losses) / len(acoustic_losses)

        # Total weighted loss (matches training formula)
        result.total_weighted_loss = (
            result.text_loss + 100.0 * result.semantic_loss + result.acoustic_loss
        )

        results.append(result)
        print(f"  {pt_path.name}: text_acc={result.text_accuracy:.3f} "
              f"text_loss={result.text_loss:.3f} sem_loss={result.semantic_loss:.3f} "
              f"total={result.total_weighted_loss:.1f}")

    return results


# ---------------------------------------------------------------------------
# Audio analysis utilities
# ---------------------------------------------------------------------------

def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy in bits from logits tensor."""
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
    return float(entropy.mean())


def spectral_flatness_analysis(audio: np.ndarray, sr: int) -> dict:
    """Compute spectral flatness globally and in 2-second windows."""
    librosa = _import_librosa()
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512)[0]
    mean_flatness = float(np.mean(flatness))

    # Windowed analysis: 2-second windows (sr*2 / hop_length frames per window)
    window_frames = int(2.0 * sr / 512)
    max_window = 0.0
    if len(flatness) >= window_frames:
        step = max(1, window_frames // 2)  # 50% overlap
        for i in range(0, len(flatness) - window_frames + 1, step):
            w = float(np.mean(flatness[i:i + window_frames]))
            max_window = max(max_window, w)
    else:
        max_window = mean_flatness

    return {"mean": mean_flatness, "max_window": max_window}


def zcr_analysis(audio: np.ndarray, sr: int) -> float:
    librosa = _import_librosa()
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)
    return float(np.mean(zcr))


def energy_analysis(audio: np.ndarray, sr: int) -> float:
    """Return coefficient of variation of RMS energy."""
    librosa = _import_librosa()
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    if mean_rms < 1e-8:
        return 0.0
    return std_rms / mean_rms


def token_repetition_analysis(semantic_tokens: list[int]) -> dict:
    """Detect repetitive patterns in semantic codebook tokens."""
    if len(semantic_tokens) < 10:
        return {"max_repeat_frames": 0, "unique_ratio": 1.0}

    max_repeat_len = 0
    for n in range(2, min(11, len(semantic_tokens) // 2)):
        for start in range(len(semantic_tokens) - 2 * n):
            subseq = semantic_tokens[start:start + n]
            repeats = 1
            pos = start + n
            while (pos + n <= len(semantic_tokens) and
                   semantic_tokens[pos:pos + n] == subseq):
                repeats += 1
                pos += n
            if repeats >= 3:
                max_repeat_len = max(max_repeat_len, n * repeats)

    unique_ratio = len(set(semantic_tokens)) / max(len(semantic_tokens), 1)
    return {"max_repeat_frames": max_repeat_len, "unique_ratio": unique_ratio}


def windowed_entropy_analysis(entropies: list[float], window_frames: int = 25) -> float:
    """Return the minimum window-mean entropy (2s windows at 12.5fps = 25 frames)."""
    if len(entropies) < window_frames:
        return float(np.mean(entropies)) if entropies else 0.0
    step = max(1, window_frames // 2)
    min_window = float("inf")
    for i in range(0, len(entropies) - window_frames + 1, step):
        w = float(np.mean(entropies[i:i + window_frames]))
        min_window = min(min_window, w)
    return min_window


def speech_continuity_analysis(frame_records: list[FrameRecord]) -> dict:
    """Count distinct speech segments and their durations."""
    if not frame_records:
        return {"segments": 0, "longest_s": 0.0}

    segments = []
    in_speech = False
    seg_start = 0

    for i, fr in enumerate(frame_records):
        if not fr.is_pad and not in_speech:
            in_speech = True
            seg_start = i
        elif fr.is_pad and in_speech:
            # Check if gap is > 1 second (12.5 frames)
            gap_ahead = 0
            for j in range(i, min(i + 13, len(frame_records))):
                if frame_records[j].is_pad:
                    gap_ahead += 1
                else:
                    break
            if gap_ahead >= 13:
                segments.append((seg_start, i))
                in_speech = False

    if in_speech:
        segments.append((seg_start, len(frame_records)))

    durations = [(end - start) / FRAME_RATE for start, end in segments]
    longest = max(durations) if durations else 0.0

    return {"segments": len(segments), "longest_s": longest}


def save_diagnostic_spectrogram(audio: np.ndarray, sr: int, output_path: str, title: str):
    """Save a diagnostic spectrogram PNG for failed samples."""
    plt = _import_matplotlib()
    librosa = _import_librosa()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, ax=axes[0], x_axis="time", y_axis="mel")
    axes[0].set_title(f"{title} - Mel Spectrogram")
    fig.colorbar(img, ax=axes[0], format="%+2.0f dB")

    # Spectral flatness over time
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512)[0]
    times = np.arange(len(flatness)) * 512 / sr
    axes[1].plot(times, flatness, linewidth=0.5)
    axes[1].axhline(y=0.15, color="r", linestyle="--", alpha=0.7, label="global threshold")
    axes[1].axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="window threshold")
    axes[1].set_title("Spectral Flatness Over Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()

    # Waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[2])
    axes[2].set_title("Waveform")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Whisper transcription & coherence
# ---------------------------------------------------------------------------

def whisper_transcribe(audio: np.ndarray, sr: int, whisper_model) -> dict:
    """Transcribe audio with Whisper, return text and metadata."""
    librosa = _import_librosa()
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio

    result = whisper_model.transcribe(
        audio_16k.astype(np.float32),
        language="en",
        fp16=torch.cuda.is_available(),
        no_speech_threshold=0.6,
    )
    text = result["text"].strip()
    segments = result.get("segments", [])
    avg_no_speech = float(np.mean([s["no_speech_prob"] for s in segments])) if segments else 1.0

    return {
        "text": text,
        "word_count": len(text.split()),
        "avg_no_speech_prob": avg_no_speech,
    }


def check_domain_keywords(text: str) -> list[str]:
    """Return domain keywords found in the transcript."""
    words = set(text.lower().split())
    return sorted(words & DOMAIN_KEYWORDS)


# ---------------------------------------------------------------------------
# Role adherence detection
# ---------------------------------------------------------------------------

# Phrases that indicate the model is acting as a BARISTA (wrong if system prompt says customer)
BARISTA_PHRASES = [
    "what can i get", "what would you like", "can i get you", "here's your",
    "your total is", "that'll be", "coming right up", "anything else",
    "for here or to go", "what size", "can i help you", "welcome to",
    "we have", "our special", "let me make", "would you like to try",
    "i'll get that started", "your order", "here you go",
]

# Phrases that indicate the model is acting as a CUSTOMER (wrong if system prompt says barista)
CUSTOMER_PHRASES = [
    "i'd like", "i'll have", "can i get", "i want", "i need",
    "do you have", "what's in", "how much is", "i'm looking for",
    "i ordered", "that's not what i", "i asked for", "where's my",
    "i've been waiting", "this isn't right", "i'm in a hurry",
]


def parse_expected_role(system_prompt: str) -> str:
    """Determine what role the model should play from the system prompt."""
    prompt_lower = system_prompt.lower()
    # PersonaPlex: AGENT = CUSTOMER, USER = BARISTA
    # System prompts describe the customer persona
    if any(w in prompt_lower for w in ["you are", "you're"]):
        if any(w in prompt_lower for w in ["customer", "patron", "client", "shopper", "commuter", "regular"]):
            return "customer"
        if any(w in prompt_lower for w in ["barista", "server", "cashier", "employee", "worker"]):
            return "barista"
    # Default for PersonaPlex: model is always the customer
    if "customer" in prompt_lower or "agent" in prompt_lower:
        return "customer"
    return "customer"  # safe default for this project


def check_role_adherence(transcript: str, expected_role: str) -> dict:
    """Check if the transcript matches the expected role or shows inversion."""
    transcript_lower = transcript.lower()
    wrong_phrases = []

    if expected_role == "customer":
        # Model should be customer — flag barista phrases
        for phrase in BARISTA_PHRASES:
            if phrase in transcript_lower:
                wrong_phrases.append(phrase)
    elif expected_role == "barista":
        # Model should be barista — flag customer phrases
        for phrase in CUSTOMER_PHRASES:
            if phrase in transcript_lower:
                wrong_phrases.append(phrase)

    return {
        "adherence": len(wrong_phrases) == 0,
        "wrong_phrases": wrong_phrases,
    }


def segment_degeneration_analysis(
    frame_records: list, audio: np.ndarray, sr: int, segment_duration_s: float = 5.0
) -> list[dict]:
    """Analyze degeneration per time segment. Returns per-segment verdicts."""
    librosa = _import_librosa()
    total_duration = len(audio) / sr
    segments = []
    segment_frames = int(segment_duration_s * FRAME_RATE)
    segment_samples = int(segment_duration_s * sr)

    for seg_idx in range(0, max(1, int(total_duration / segment_duration_s))):
        t_start = seg_idx * segment_duration_s
        t_end = min(t_start + segment_duration_s, total_duration)
        sample_start = int(t_start * sr)
        sample_end = min(int(t_end * sr), len(audio))
        frame_start = int(t_start * FRAME_RATE)
        frame_end = min(int(t_end * FRAME_RATE), len(frame_records))

        seg_audio = audio[sample_start:sample_end]
        seg_frames = frame_records[frame_start:frame_end]

        if len(seg_audio) < sr * 0.5:  # skip very short tail segments
            continue

        # Spectral flatness for this segment
        flatness = librosa.feature.spectral_flatness(y=seg_audio, hop_length=512)[0]
        mean_flat = float(np.mean(flatness)) if len(flatness) > 0 else 0.0

        # Entropy for this segment
        entropies = [fr.audio_entropy_cb0 for fr in seg_frames]
        mean_ent = float(np.mean(entropies)) if entropies else 0.0

        # Pad ratio for this segment
        pad_count = sum(1 for fr in seg_frames if fr.is_pad)
        pad_ratio = pad_count / max(len(seg_frames), 1)

        issues = []
        if mean_flat > 0.2:
            issues.append("whooshing")
        if mean_ent < 2.5 and entropies:
            issues.append("low_entropy")
        if pad_ratio > 0.9:
            issues.append("mostly_silent")

        verdict = "ok" if not issues else "degraded"

        segments.append({
            "time_s": f"{t_start:.1f}-{t_end:.1f}",
            "verdict": verdict,
            "spectral_flatness": round(mean_flat, 4),
            "entropy_mean": round(mean_ent, 2),
            "pad_ratio": round(pad_ratio, 3),
            "issues": issues,
        })

    return segments


def text_audio_cross_validation(
    generated_text_tokens: list[int], whisper_text: str, tokenizer
) -> float:
    """Compare model's text tokens to Whisper transcript. Returns word overlap ratio."""
    # Decode non-pad text tokens
    non_pad = [t for t in generated_text_tokens if t not in (0, 3, -1)]
    if not non_pad or not whisper_text:
        return 0.0
    try:
        decoded = tokenizer.decode(non_pad).replace("\u2581", " ").strip()
    except Exception:
        return 0.0

    model_words = set(decoded.lower().split())
    whisper_words = set(whisper_text.lower().split())
    if not model_words or not whisper_words:
        return 0.0
    overlap = len(model_words & whisper_words)
    return overlap / max(len(model_words), len(whisper_words))


# ---------------------------------------------------------------------------
# System prompt lookup
# ---------------------------------------------------------------------------

def load_prompt_lookup(training_json_paths: list[Path]) -> dict[str, str]:
    """Build conv_id → system_prompt mapping from training JSON files."""
    lookup = {}
    for path in training_json_paths:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for conv in data:
            cid = conv.get("conversation_id", "")
            prompt = conv.get("system_prompt", "")
            if cid and prompt:
                lookup[cid] = prompt
    return lookup


def recover_system_prompt(codes: torch.Tensor, tokenizer, prompt_lookup: dict, pt_name: str) -> str:
    """Get system prompt for a .pt file, from lookup or by parsing tokens."""
    # Try lookup first
    conv_id = Path(pt_name).stem
    if conv_id in prompt_lookup:
        return prompt_lookup[conv_id]

    # Fallback: parse text tokens from row 0
    text_row = codes[0].tolist()
    prompt_tokens = [t for t in text_row[:50] if t not in (-1, 0, PAD_TOKEN)]
    if prompt_tokens:
        try:
            return tokenizer.decode(prompt_tokens)
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Mode B: Generation Eval
# ---------------------------------------------------------------------------

def generation_eval_single(
    pt_path: Path,
    system_prompt: str,
    seed: int,
    # Model objects (loaded externally)
    lm,
    mimi,
    other_mimi,
    text_tokenizer,
    whisper_model,
    voice_prompt_path: str,
    device: str,
    output_dir: Path,
    skip_utmos: bool = False,
) -> GenerationSampleResult:
    """Run free generation on a single .pt eval file and evaluate output."""
    from moshi.models import LMGen
    from moshi.models.lm import (
        load_audio as lm_load_audio,
        _iterate_audio as lm_iterate_audio,
        encode_from_sphn as lm_encode_from_sphn,
    )
    from moshi.offline import warmup, decode_tokens_to_pcm, wrap_with_system_tags, seed_all
    import sphn

    result = GenerationSampleResult(
        file=pt_path.name, system_prompt=system_prompt[:200], seed=seed
    )

    seed_all(seed)

    # Load eval .pt and extract user audio
    codes = torch.load(pt_path, weights_only=True)

    # --- Construct LMGen with return_logits=True ---
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=0.9,
        temp_text=1.0,
        top_k=250,
        top_k_text=25,
        return_logits=True,
        text_tokenizer=text_tokenizer,
    )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # Warmup (custom — handles return_logits=True where step() returns tuples)
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        enc_codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(enc_codes.shape[-1]):
            step_out = lm_gen.step(enc_codes[:, :, c:c + 1])
            if step_out is None:
                continue
            # Unpack: with return_logits=True, step returns (tokens, (text_logits, audio_logits))
            tokens = step_out[0] if isinstance(step_out, tuple) else step_out
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Set system prompt
    if voice_prompt_path and voice_prompt_path.endswith(".pt"):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    elif voice_prompt_path:
        lm_gen.load_voice_prompt(voice_prompt_path)

    wrapped_prompt = wrap_with_system_tags(system_prompt) if system_prompt else ""
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrapped_prompt) if wrapped_prompt else None
    )

    # Reset and inject prompts
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    # Decode user audio from .pt tokens → PCM
    user_codes = codes[9:17].unsqueeze(0).to(device)  # [1, 8, T]
    # We need to decode tokens to audio, then re-encode for the streaming pipeline
    # Use sphn to write a temp wav and load it back (matches offline.py pattern)
    temp_wav = output_dir / f"_temp_user_{pt_path.stem}_{seed}.wav"

    # Decode user audio tokens to PCM via Mimi (non-streaming)
    other_mimi.reset_streaming()
    other_mimi.streaming_forever(1)
    user_pcm_frames = []
    T = user_codes.shape[-1]
    for t in range(T):
        frame_codes = user_codes[:, :, t:t+1]
        pcm = other_mimi.decode(frame_codes)
        user_pcm_frames.append(pcm.detach().cpu().numpy()[0, 0])
    user_pcm_full = np.concatenate(user_pcm_frames, axis=-1)
    sphn.write_wav(str(temp_wav), user_pcm_full, int(mimi.sample_rate))

    # Reset other_mimi for generation decode
    other_mimi.reset_streaming()
    other_mimi.streaming_forever(1)

    # Load user audio for streaming
    user_audio = lm_load_audio(str(temp_wav), int(mimi.sample_rate))

    # --- Streaming generation loop ---
    generated_frames = []
    frame_records = []
    generated_text_tokens = []
    step_count = 0
    total_wall_time = 0.0

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
        max_batch=1,
    ):
        steps = user_encoded.shape[-1]
        for c in range(steps):
            step_in = user_encoded[:, :, c:c + 1]
            t0 = time.perf_counter()
            step_result = lm_gen.step(step_in)
            step_time = time.perf_counter() - t0
            total_wall_time += step_time

            if step_result is None:
                continue

            tokens, (text_logits, audio_logits) = step_result

            # Decode PCM
            pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
            generated_frames.append(pcm)

            # Collect frame data
            text_token = tokens[0, 0, 0].item()
            audio_toks = tokens[0, 1:9, 0].cpu().tolist()
            generated_text_tokens.append(text_token)

            text_ent = compute_entropy(text_logits[0, 0, 0])
            audio_ent = compute_entropy(audio_logits[0, 0])

            frame_records.append(FrameRecord(
                step_idx=step_count,
                text_token=text_token,
                audio_tokens=audio_toks,
                is_pad=(text_token == PAD_TOKEN),
                consecutive_pad_count=lm_gen._consecutive_pad,
                wall_time_ms=step_time * 1000,
                text_entropy=text_ent,
                audio_entropy_cb0=audio_ent,
            ))
            step_count += 1

    # Clean up temp file
    if temp_wav.exists():
        temp_wav.unlink()

    if not generated_frames:
        result.degenerate = True
        result.degenerate_reasons.append("no_frames_generated")
        return result

    # --- Concatenate output audio ---
    output_pcm = np.concatenate(generated_frames, axis=-1)
    audio_duration = len(output_pcm) / SAMPLE_RATE

    # Save output WAV
    out_wav = output_dir / f"gen_{pt_path.stem}_seed{seed}.wav"
    sphn.write_wav(str(out_wav), output_pcm, SAMPLE_RATE)

    # --- Evaluate ---

    # A. Spectral flatness
    sf_result = spectral_flatness_analysis(output_pcm, SAMPLE_RATE)
    result.spectral_flatness_mean = sf_result["mean"]
    result.spectral_flatness_max_window = sf_result["max_window"]

    # B. ZCR
    result.zcr_mean = zcr_analysis(output_pcm, SAMPLE_RATE)

    # C. Energy CV
    result.energy_cv = energy_analysis(output_pcm, SAMPLE_RATE)

    # D. Token repetition
    semantic_tokens = [fr.audio_tokens[0] for fr in frame_records if fr.audio_tokens]
    rep = token_repetition_analysis(semantic_tokens)
    result.token_repeat_max_frames = rep["max_repeat_frames"]
    result.token_unique_ratio = rep["unique_ratio"]

    # E. Token entropy
    audio_entropies = [fr.audio_entropy_cb0 for fr in frame_records]
    result.token_entropy_mean = float(np.mean(audio_entropies)) if audio_entropies else 0.0
    result.token_entropy_min_window = windowed_entropy_analysis(audio_entropies)

    # F. Response latency
    first_speech = next((fr.step_idx for fr in frame_records if not fr.is_pad), len(frame_records))
    result.response_latency_s = first_speech / FRAME_RATE

    # G. Silence ratio
    pad_count = sum(1 for fr in frame_records if fr.is_pad)
    result.silence_ratio = pad_count / max(len(frame_records), 1)

    # H. Speech continuity
    cont = speech_continuity_analysis(frame_records)
    result.speech_segments = cont["segments"]
    result.longest_speech_s = cont["longest_s"]

    # I. RTF
    result.rtf = total_wall_time / max(audio_duration, 0.01)

    # J. Whisper transcription
    if whisper_model is not None:
        w_result = whisper_transcribe(output_pcm, SAMPLE_RATE, whisper_model)
        result.transcription = w_result["text"]
        result.word_count = w_result["word_count"]
        result.domain_keywords_found = check_domain_keywords(w_result["text"])

        # K. Text-audio cross-validation
        result.text_audio_word_overlap = text_audio_cross_validation(
            generated_text_tokens, w_result["text"], text_tokenizer
        )

    # L. Role adherence
    result.expected_role = parse_expected_role(system_prompt)
    if result.transcription:
        role_check = check_role_adherence(result.transcription, result.expected_role)
        result.role_adherence = role_check["adherence"]
        result.role_inversion_phrases = role_check["wrong_phrases"]

    # M. Per-segment degeneration
    result.segment_verdicts = segment_degeneration_analysis(
        frame_records, output_pcm, SAMPLE_RATE
    )

    # --- Degenerate checks ---
    reasons = []
    if result.spectral_flatness_mean >= 0.15:
        reasons.append(f"spectral_flatness_mean={result.spectral_flatness_mean:.3f}")
    if result.spectral_flatness_max_window >= 0.3:
        reasons.append(f"spectral_flatness_window={result.spectral_flatness_max_window:.3f}")
    if result.token_repeat_max_frames >= 20:
        reasons.append(f"token_repeat={result.token_repeat_max_frames}frames")
    if result.token_entropy_mean < 3.0 and audio_entropies:
        reasons.append(f"low_entropy_mean={result.token_entropy_mean:.2f}")
    if result.token_entropy_min_window < 2.0 and audio_entropies:
        reasons.append(f"low_entropy_window={result.token_entropy_min_window:.2f}")
    if result.zcr_mean > 0.15 or result.zcr_mean < 0.01:
        reasons.append(f"zcr_out_of_range={result.zcr_mean:.3f}")
    if result.energy_cv < 0.1:
        reasons.append(f"flat_energy_cv={result.energy_cv:.3f}")
    if not result.role_adherence:
        reasons.append(f"role_inversion={','.join(result.role_inversion_phrases[:3])}")

    result.degenerate = len(reasons) > 0
    result.degenerate_reasons = reasons

    # --- Pass/fail ---
    whisper_pass = result.word_count >= 5
    latency_pass = result.response_latency_s < 5.0
    silence_pass = result.silence_ratio < 0.95

    result.pass_all = (
        not result.degenerate
        and whisper_pass
        and latency_pass
        and silence_pass
        and result.role_adherence
    )

    # --- Save artifacts for cron judge ---
    result.output_wav_path = str(out_wav)

    # Save transcript file
    transcript_path = output_dir / f"transcript_{pt_path.stem}_seed{seed}.json"
    transcript_data = {
        "file": pt_path.name,
        "seed": seed,
        "system_prompt": system_prompt,
        "expected_role": result.expected_role,
        "transcription": result.transcription,
        "model_text_tokens_decoded": "",
        "role_adherence": result.role_adherence,
        "role_inversion_phrases": result.role_inversion_phrases,
        "degenerate": result.degenerate,
        "degenerate_reasons": result.degenerate_reasons,
        "segment_verdicts": result.segment_verdicts,
        "judge_verdict": "pending",
    }
    # Decode model text tokens for judge review
    non_pad = [t for t in generated_text_tokens if t not in (0, 3, -1)]
    if non_pad:
        try:
            transcript_data["model_text_tokens_decoded"] = (
                text_tokenizer.decode(non_pad).replace("\u2581", " ").strip()
            )
        except Exception:
            pass
    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=2, default=str)
    result.output_transcript_path = str(transcript_path)

    # Save diagnostic spectrogram (always, not just on failure — useful for judge)
    spec_path = output_dir / f"diag_{pt_path.stem}_seed{seed}.png"
    try:
        status = "FAIL" if not result.pass_all else "PASS"
        save_diagnostic_spectrogram(
            output_pcm, SAMPLE_RATE, str(spec_path),
            f"{pt_path.stem} ({status})"
        )
        result.diagnostic_png_path = str(spec_path)
    except Exception as e:
        print(f"  Warning: could not save spectrogram: {e}")

    return result


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite_score(
    token_results: list[TokenEvalResult],
    gen_results: list[GenerationSampleResult],
) -> float:
    """Compute 0-100 composite score for autoresearcher comparison."""
    # Aggregate token eval
    if token_results:
        avg_text_acc = np.mean([r.text_accuracy for r in token_results])
        avg_text_loss = np.mean([r.text_loss for r in token_results])
        avg_sem_loss = np.mean([r.semantic_loss for r in token_results])
    else:
        avg_text_acc = 0.0
        avg_text_loss = 10.0
        avg_sem_loss = 10.0

    # Aggregate generation eval
    if gen_results:
        avg_utmos = np.mean([r.utmos for r in gen_results]) if any(r.utmos > 0 for r in gen_results) else 0.0
        whisper_pass_rate = np.mean([1.0 if r.word_count >= 5 else 0.0 for r in gen_results])
        no_degen_rate = np.mean([0.0 if r.degenerate else 1.0 for r in gen_results])
        latency_pass_rate = np.mean([1.0 if r.response_latency_s < 5.0 else 0.0 for r in gen_results])
        crossval_rate = np.mean([1.0 if r.text_audio_word_overlap >= 0.2 else 0.0 for r in gen_results])
        role_adherence_rate = np.mean([1.0 if r.role_adherence else 0.0 for r in gen_results])
    else:
        avg_utmos = 0.0
        whisper_pass_rate = 0.0
        no_degen_rate = 0.0
        latency_pass_rate = 0.0
        crossval_rate = 0.0
        role_adherence_rate = 0.0

    score = (
        10 * max(0.0, min(1.0, 1.0 - avg_text_loss / 10.0))  # 10 pts: token prediction
        + 10 * avg_text_acc                                     # 10 pts: text accuracy
        + 10 * max(0.0, min(1.0, 1.0 - avg_sem_loss / 10.0))  # 10 pts: semantic prediction
        + 10 * min(avg_utmos / 5.0, 1.0)                       # 10 pts: audio quality
        + 10 * whisper_pass_rate                                # 10 pts: coherent speech
        + 15 * no_degen_rate                                    # 15 pts: no degenerate audio
        + 10 * latency_pass_rate                                # 10 pts: responsive
        + 10 * crossval_rate                                    # 10 pts: text-audio alignment
        + 15 * role_adherence_rate                              # 15 pts: stays in character
    )
    return float(score)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_results(path_a: str, path_b: str):
    """Compare two benchmark result JSONs and print delta."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    score_a = a.get("composite_score", 0)
    score_b = b.get("composite_score", 0)
    delta = score_b - score_a

    print(f"\n{'='*60}")
    print(f"COMPARISON: {Path(path_a).name} vs {Path(path_b).name}")
    print(f"{'='*60}")
    print(f"  Baseline score: {score_a:.1f}")
    print(f"  Current score:  {score_b:.1f}")
    print(f"  Delta:          {delta:+.1f}")
    print(f"  Result:         {'IMPROVED' if delta > 0 else 'REGRESSED' if delta < 0 else 'UNCHANGED'}")

    # Detail comparison
    if "token_eval" in a and "token_eval" in b:
        print(f"\n  Token Eval:")
        for key in ["text_accuracy", "text_loss", "semantic_loss", "total_weighted_loss"]:
            va = a["token_eval"].get(key, 0)
            vb = b["token_eval"].get(key, 0)
            d = vb - va
            better = d > 0 if "accuracy" in key else d < 0
            print(f"    {key}: {va:.3f} -> {vb:.3f} ({d:+.3f}) {'[better]' if better else '[worse]' if d != 0 else ''}")

    if "generation_eval" in a and "generation_eval" in b:
        ag_a = a["generation_eval"].get("aggregate", {})
        ag_b = b["generation_eval"].get("aggregate", {})
        print(f"\n  Generation Eval:")
        for key in ["mean_utmos", "mean_rtf", "degenerate_count"]:
            va = ag_a.get(key, 0)
            vb = ag_b.get(key, 0)
            print(f"    {key}: {va} -> {vb}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Personaplex-bench: Benchmark for speech-to-speech models"
    )
    parser.add_argument("--mode", choices=["both", "token-eval", "generation-eval"],
                        default="both", help="Evaluation mode")
    parser.add_argument("--moshi-repo-path", type=str, default=None,
                        help="Path to personaplex-fine-coffee repo (added to sys.path)")
    parser.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1",
                        help="HuggingFace repo for model weights")
    parser.add_argument("--moshi-weight", type=str, default=None,
                        help="Path to model .safetensors (auto-downloads if not set)")
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument("--eval-pt-dir", type=str, default=".",
                        help="Directory containing eval .pt files")
    parser.add_argument("--eval-files", nargs="+", default=["test_001.pt"],
                        help="Eval .pt filenames")
    parser.add_argument("--voice-prompt", type=str, default=None,
                        help="Voice prompt (.pt or .wav) for generation eval")
    parser.add_argument("--training-json", type=str, nargs="*", default=[],
                        help="Path(s) to training.json for system prompt lookup")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Override output JSON path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-utmos", action="store_true")
    parser.add_argument("--skip-rtf-check", action="store_true")
    parser.add_argument("--whisper-device", type=str, default=None,
                        help="Device for Whisper (default: same as --device)")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"),
                        help="Compare two result JSONs")

    args = parser.parse_args()

    # --- Compare mode ---
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # --- Setup moshi imports ---
    if args.moshi_repo_path:
        moshi_path = Path(args.moshi_repo_path)
        if (moshi_path / "moshi").exists():
            sys.path.insert(0, str(moshi_path))
        # Also check if moshi is a subdirectory
        if (moshi_path / "moshi" / "moshi").exists():
            sys.path.insert(0, str(moshi_path / "moshi"))

    try:
        from moshi.models import loaders
    except ImportError:
        print("ERROR: Cannot import moshi. Install it or pass --moshi-repo-path.")
        print("  cd /path/to/personaplex-fine-coffee/moshi && pip install -e .")
        sys.exit(1)

    # --- Resolve paths ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_pt_dir = Path(args.eval_pt_dir)
    eval_files = [eval_pt_dir / f for f in args.eval_files]

    # Validate eval files exist
    for f in eval_files:
        if not f.exists():
            print(f"ERROR: Eval file not found: {f}")
            sys.exit(1)

    # --- Load prompt lookup ---
    prompt_lookup = {}
    if args.training_json:
        prompt_lookup = load_prompt_lookup([Path(p) for p in args.training_json])
        print(f"Loaded {len(prompt_lookup)} system prompts from training JSON")

    # --- Load model ---
    print("Loading models...")
    from huggingface_hub import hf_hub_download
    import sentencepiece

    device = args.device
    hf_repo = args.hf_repo

    # Mimi
    mimi_weight = args.mimi_weight
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    # Tokenizer
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # LM
    moshi_weight = args.moshi_weight
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()
    print("Models loaded.")

    # --- Voice prompt ---
    voice_prompt_path = args.voice_prompt
    if voice_prompt_path is None:
        # Try to download default
        try:
            voice_dir_tgz = hf_hub_download(hf_repo, "voices.tgz")
            voice_dir = Path(voice_dir_tgz).parent / "voices"
            if not voice_dir.exists():
                import tarfile
                with tarfile.open(voice_dir_tgz, "r:gz") as tar:
                    tar.extractall(path=Path(voice_dir_tgz).parent)
            # Use first available voice
            voices = sorted(voice_dir.glob("*.pt"))
            if voices:
                voice_prompt_path = str(voices[0])
                print(f"Using voice prompt: {voice_prompt_path}")
        except Exception:
            print("Warning: No voice prompt available. Generation eval may be affected.")

    # --- Whisper ---
    whisper_model = None
    if args.mode in ("both", "generation-eval"):
        print(f"Loading Whisper ({args.whisper_model})...")
        whisper = _import_whisper()
        w_device = args.whisper_device or args.device
        whisper_model = whisper.load_model(args.whisper_model, device=w_device)
        print("Whisper loaded.")

    # --- Run evaluations ---
    t_start = time.time()
    token_results = []
    gen_results = []

    # Mode A: Token Eval
    if args.mode in ("both", "token-eval"):
        print(f"\n{'='*60}")
        print("MODE A: Token Eval (teacher-forcing)")
        print(f"{'='*60}")
        token_results = token_eval(lm, eval_files, device)

    # Mode B: Generation Eval
    if args.mode in ("both", "generation-eval"):
        print(f"\n{'='*60}")
        print("MODE B: Generation Eval (free generation)")
        print(f"{'='*60}")

        for pt_path in eval_files:
            system_prompt = recover_system_prompt(
                torch.load(pt_path, weights_only=True),
                text_tokenizer, prompt_lookup, pt_path.name
            )
            print(f"\n  Evaluating: {pt_path.name}")
            print(f"  System prompt: {system_prompt[:100]}...")
            print(f"  Seed: {args.seed}")

            sample_result = generation_eval_single(
                pt_path=pt_path,
                system_prompt=system_prompt,
                seed=args.seed,
                lm=lm,
                mimi=mimi,
                other_mimi=other_mimi,
                text_tokenizer=text_tokenizer,
                whisper_model=whisper_model,
                voice_prompt_path=voice_prompt_path,
                device=device,
                output_dir=output_dir,
                skip_utmos=args.skip_utmos,
            )
            gen_results.append(sample_result)

            status = "PASS" if sample_result.pass_all else "FAIL"
            print(f"  Result: {status}")
            if sample_result.transcription:
                print(f"  Transcription: {sample_result.transcription[:120]}...")
            print(f"  Role: expected={sample_result.expected_role} "
                  f"adherence={'OK' if sample_result.role_adherence else 'INVERTED'}")
            if sample_result.role_inversion_phrases:
                print(f"  Wrong-role phrases: {sample_result.role_inversion_phrases}")
            if sample_result.degenerate_reasons:
                print(f"  Degenerate: {', '.join(sample_result.degenerate_reasons)}")
            degraded_segs = [s for s in sample_result.segment_verdicts if s["verdict"] == "degraded"]
            if degraded_segs:
                print(f"  Degraded segments: {[s['time_s'] for s in degraded_segs]}")
            print(f"  Metrics: flatness={sample_result.spectral_flatness_mean:.3f} "
                  f"entropy={sample_result.token_entropy_mean:.2f} "
                  f"latency={sample_result.response_latency_s:.1f}s "
                  f"rtf={sample_result.rtf:.2f}")
            print(f"  Artifacts: {sample_result.output_transcript_path}")

    total_time = time.time() - t_start

    # --- Composite score ---
    composite = compute_composite_score(token_results, gen_results)

    # --- Build output JSON ---
    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": moshi_weight,
            "device": device,
            "eval_files": [f.name for f in eval_files],
            "total_time_s": round(total_time, 1),
        },
        "composite_score": round(composite, 1),
    }

    if token_results:
        # Average token results
        output["token_eval"] = {
            "text_accuracy": round(np.mean([r.text_accuracy for r in token_results]), 4),
            "text_loss": round(np.mean([r.text_loss for r in token_results]), 4),
            "text_perplexity": round(np.mean([r.text_perplexity for r in token_results]), 2),
            "semantic_accuracy": round(np.mean([r.semantic_accuracy for r in token_results]), 4),
            "semantic_loss": round(np.mean([r.semantic_loss for r in token_results]), 4),
            "acoustic_loss": round(np.mean([r.acoustic_loss for r in token_results]), 4),
            "total_weighted_loss": round(np.mean([r.total_weighted_loss for r in token_results]), 2),
            "per_file": [asdict(r) for r in token_results],
        }

    if gen_results:
        output["generation_eval"] = {
            "samples": [asdict(r) for r in gen_results],
            "aggregate": {
                "mean_utmos": round(np.mean([r.utmos for r in gen_results]), 2),
                "mean_rtf": round(np.mean([r.rtf for r in gen_results]), 2),
                "mean_spectral_flatness": round(np.mean([r.spectral_flatness_mean for r in gen_results]), 4),
                "mean_entropy": round(np.mean([r.token_entropy_mean for r in gen_results]), 2),
                "pass_rate": f"{sum(1 for r in gen_results if r.pass_all)}/{len(gen_results)}",
                "degenerate_count": sum(1 for r in gen_results if r.degenerate),
                "role_inversion_count": sum(1 for r in gen_results if not r.role_adherence),
                "pending_judge_review": sum(1 for r in gen_results if r.output_transcript_path),
            },
        }

    # Write JSON
    json_path = args.output_json or str(output_dir / "results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    if token_results:
        te = output["token_eval"]
        print(f"  Token Eval:")
        print(f"    Text accuracy:      {te['text_accuracy']:.3f}")
        print(f"    Text loss:          {te['text_loss']:.3f}")
        print(f"    Semantic loss:      {te['semantic_loss']:.3f}")
        print(f"    Total weighted:     {te['total_weighted_loss']:.1f}")
    if gen_results:
        ag = output["generation_eval"]["aggregate"]
        print(f"  Generation Eval:")
        print(f"    Pass rate:          {ag['pass_rate']}")
        print(f"    Degenerate count:   {ag['degenerate_count']}")
        print(f"    Mean RTF:           {ag['mean_rtf']:.2f}")
        print(f"    Mean flatness:      {ag['mean_spectral_flatness']:.4f}")

    print(f"\n  COMPOSITE SCORE: {composite:.1f} / 100")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Results: {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
