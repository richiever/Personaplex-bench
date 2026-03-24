# Personaplex-bench

Automated benchmark for [PersonaPlex](https://github.com/richiever/personaplex-fine-coffee) full-duplex speech-to-speech models. Designed for autoresearcher training loops.

## Why

Training loss alone doesn't catch real audio failures:
- **Whooshing noise** from degenerate token generation
- **Infinite loops** producing repetitive incoherent audio
- **Persona drift** where the model forgets its assigned role
- **Silent model** that generates padding tokens instead of speech
- **Text-audio divergence** where the text stream is fine but audio is garbage

This benchmark goes beyond loss with two evaluation modes and produces a single **composite score (0-100)** that an autoresearcher can use to compare training runs.

## Evaluation Modes

### Mode A: Token Eval (fast, ~5s per file)

Teacher-forcing evaluation using `forward_train()` on held-out `.pt` files. Computes:

| Metric | What it measures |
|--------|-----------------|
| Text accuracy | Can the model predict the right words? |
| Text loss / perplexity | Text prediction confidence |
| Semantic accuracy | Can it predict semantic audio tokens (codebook 0)? |
| Semantic loss | Core meaning prediction quality |
| Acoustic loss | Sound quality prediction (codebooks 1-7) |
| Total weighted loss | `text + 100x semantic + 1x acoustic` (matches training) |

### Mode B: Generation Eval (thorough, ~30s per file)

Free-generation evaluation: extracts user (barista) audio from `.pt` files, feeds it through the streaming pipeline, and evaluates the model's freely-generated customer response.

**Degenerate detection** (windowed for intermittent issues):
| Check | What it catches | Pass threshold |
|-------|----------------|----------------|
| Spectral flatness (2s windows) | Whooshing, noise | No window > 0.3 |
| Token repetition | Infinite loops | No repeat > 20 frames |
| Token entropy (2s windows) | Stuck sampling | No window mean < 2.0 |
| Energy dynamics | Uniform noise | CV > 0.1 |
| Zero-crossing rate | Noise vs speech | 0.01 < mean < 0.15 |

**Speech coherence**:
| Check | What it catches | Pass threshold |
|-------|----------------|----------------|
| Whisper transcription | Incoherent audio | >= 5 words |
| Domain keywords | Off-topic speech | >= 1 keyword |
| Text-audio cross-validation | Stream divergence | >= 20% word overlap |
| UTMOS | Low audio quality | > 2.5 / 5.0 |

**Behavioral**:
| Check | What it catches | Pass threshold |
|-------|----------------|----------------|
| Response latency | Slow startup | < 5.0 seconds |
| Silence ratio | Dead model | < 95% padding |
| Speech continuity | Burst-only output | >= 2 segments, longest > 2s |
| RTF | Too slow for real-time | < 1.0 |

**Persona & robustness** (multi-file runs):
| Check | What it catches | Pass threshold |
|-------|----------------|----------------|
| Persona diversity | Persona collapse | TF-IDF similarity < 0.8 |
| Multi-seed stability | Seed-fragile model | Both seeds pass |

## RunPod Setup

### Prerequisites

- RunPod instance with **A40 or better** (48GB+ VRAM recommended for 7B model + Whisper)
- HuggingFace account with access to `nvidia/personaplex-7b-v1`

### Step-by-step

```bash
# 1. SSH into your RunPod instance

# 2. Clone repos
cd /workspace
git clone https://github.com/richiever/personaplex-fine-coffee.git
git clone https://github.com/richiever/Personaplex-bench.git

# 3. Install Moshi (PersonaPlex model framework)
cd /workspace/personaplex-fine-coffee/moshi
pip install -e .

# 4. Install benchmark dependencies
cd /workspace/Personaplex-bench
pip install -r requirements.txt

# 5. Login to HuggingFace (needed for model weights)
huggingface-cli login
# Or: export HF_TOKEN=hf_your_token_here

# 6. Download eval data
huggingface-cli download AnthrolyticB/personaplex-training-data-test \
  test_001.pt training.json --repo-type dataset --local-dir /workspace/eval_data
```

## Usage

### Quick token eval (~30 seconds)

```bash
python benchmark_audio.py \
  --mode token-eval \
  --moshi-repo-path /workspace/personaplex-fine-coffee \
  --eval-pt-dir /workspace/eval_data \
  --eval-files test_001.pt \
  --device cuda
```

### Full benchmark (~5 minutes)

```bash
python benchmark_audio.py \
  --moshi-repo-path /workspace/personaplex-fine-coffee \
  --eval-pt-dir /workspace/eval_data \
  --eval-files test_001.pt \
  --voice-prompt NATF2.pt \
  --training-json /workspace/eval_data/training.json \
  --output-dir /workspace/benchmark_results \
  --device cuda
```

### With custom fine-tuned weights

```bash
python benchmark_audio.py \
  --moshi-repo-path /workspace/personaplex-fine-coffee \
  --moshi-weight /workspace/finetuned_model.safetensors \
  --eval-pt-dir /workspace/eval_data \
  --eval-files test_001.pt \
  --output-dir /workspace/benchmark_results \
  --device cuda
```

### Compare two runs

```bash
python benchmark_audio.py --compare baseline.json current.json
```

## Autoresearcher Integration

```python
import subprocess
import json

def run_benchmark(model_path, eval_files=["test_001.pt"]):
    """Run benchmark and return composite score."""
    result = subprocess.run([
        "python", "/workspace/Personaplex-bench/benchmark_audio.py",
        "--moshi-repo-path", "/workspace/personaplex-fine-coffee",
        "--moshi-weight", model_path,
        "--eval-pt-dir", "/workspace/eval_data",
        "--eval-files", *eval_files,
        "--output-dir", "/workspace/benchmark_results",
        "--output-json", "/workspace/benchmark_results/results.json",
        "--device", "cuda",
    ], capture_output=True, text=True)

    with open("/workspace/benchmark_results/results.json") as f:
        results = json.load(f)

    return results["composite_score"]  # 0-100

# In your training loop:
baseline_score = run_benchmark("/workspace/weights/model.safetensors")
# ... train ...
new_score = run_benchmark("/workspace/finetuned_model.safetensors")
improved = new_score > baseline_score
print(f"Score: {baseline_score:.1f} -> {new_score:.1f} ({'improved' if improved else 'regressed'})")
```

## Output Format

```json
{
  "metadata": {
    "timestamp": "2026-03-24T12:00:00Z",
    "model_path": "/workspace/finetuned_model.safetensors",
    "device": "cuda",
    "eval_files": ["test_001.pt"]
  },
  "token_eval": {
    "text_accuracy": 0.42,
    "text_loss": 3.21,
    "text_perplexity": 24.8,
    "semantic_accuracy": 0.31,
    "semantic_loss": 4.12,
    "acoustic_loss": 5.67,
    "total_weighted_loss": 419.88
  },
  "generation_eval": {
    "samples": [
      {
        "file": "test_001.pt",
        "system_prompt": "You are Angry Alex...",
        "transcription": "Yeah I need a large oat milk latte...",
        "utmos": 3.1,
        "spectral_flatness_mean": 0.06,
        "spectral_flatness_max_window": 0.12,
        "token_entropy_mean": 7.2,
        "response_latency_s": 1.4,
        "rtf": 0.43,
        "degenerate": false,
        "pass": true
      }
    ],
    "aggregate": {
      "mean_utmos": 3.1,
      "mean_rtf": 0.43,
      "pass_rate": "1/1",
      "degenerate_count": 0
    }
  },
  "composite_score": 62.4
}
```

## Composite Score Breakdown

| Component | Points | Source |
|-----------|--------|--------|
| Token prediction (1 - loss/10) | 15 | Mode A |
| Text accuracy | 15 | Mode A |
| Semantic prediction | 10 | Mode A |
| Audio quality (UTMOS) | 15 | Mode B |
| Coherent speech (Whisper) | 10 | Mode B |
| No degenerate audio | 15 | Mode B |
| Responsive (latency) | 10 | Mode B |
| Text-audio alignment | 10 | Mode B |
| **Total** | **100** | |

## Data Format

Eval `.pt` files have shape `[17, T]`:

| Row(s) | Content |
|--------|---------|
| 0 | Text tokens (system prompt + conversation) |
| 1-8 | Agent (customer) audio codebooks — model generates these |
| 9-16 | User (barista) audio codebooks — fed as input |

System prompts are looked up from `training.json` by matching `conv_XXXX.pt` filenames to `conversation_id` fields.

## Dependencies

| Package | Version | Purpose | GPU Required |
|---------|---------|---------|-------------|
| torch | >=2.2.0 | Model inference, tensor ops | Yes |
| numpy | >=1.26 | Array operations | No |
| sphn | >=0.1.4 | Audio I/O (WAV read/write) | No |
| sentencepiece | ==0.2 | Text tokenizer (Moshi) | No |
| safetensors | >=0.4.0 | Model weight loading | No |
| huggingface_hub | latest | Download models and data | No |
| librosa | >=0.10.0 | Spectral analysis | No |
| matplotlib | latest | Diagnostic spectrograms | No |
| openai-whisper | latest | ASR coherence checking | GPU recommended |
| scikit-learn | latest | TF-IDF for persona diversity | No |

Also requires `personaplex-fine-coffee/moshi` installed (provides `moshi.models`, `moshi.offline`).

## Troubleshooting

### CUDA out of memory

The 7B model (~14GB bf16) + 2x Mimi (~500MB) + Whisper base (~300MB) needs ~20GB VRAM. If tight:
- Use `--skip-utmos` to skip UTMOS evaluation
- Use `--whisper-device cpu` to run Whisper on CPU
- Use `--mode token-eval` for Mode A only (no generation)

### "No module named moshi"

Make sure `personaplex-fine-coffee/moshi` is installed:
```bash
cd /workspace/personaplex-fine-coffee/moshi && pip install -e .
```
Or pass `--moshi-repo-path /workspace/personaplex-fine-coffee` to add it to sys.path.

### Empty transcription

If Whisper returns empty text, the model likely produced silence or noise. Check:
1. The output WAV in `benchmark_results/` — listen to it
2. The diagnostic spectrogram PNG — look for flat energy or high spectral flatness
3. Increase audio temperature: the base model may need different sampling params

### Slow RTF (> 1.0)

RTF > 1.0 means the model is slower than real-time. This is expected on smaller GPUs (< A40). The RTF check can be disabled with `--skip-rtf-check`.
