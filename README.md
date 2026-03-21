# Speculative Decoding and Multi-Token Pipelines

CECS 530
Nathan Lai
Tommy Long
Angel Mendez

Implementation and analysis of speculative decoding using TinyLlama-1.1B as the draft model and Llama-2-7B as the target model, with an analytical performance model and a multi-token pipeline architecture.

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (we used RTX 4070 super in our results)
- ~16 GB VRAM recommended (we had 16 GB shared VRAM)
- ~15 GB disk space for models (for LLama2-7B and TinyLlama-1.1B)

---

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA, install separately matching your CUDA version:

```bash
# CUDA 13 (used in this project)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 2. Set up HuggingFace token

Llama-2 requires a HuggingFace account with access granted at [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).

Create a `.env` file in the project root:

```
HF_TOKEN="hf_your_token_here"
```

Get your token from huggingface.co/settings/tokens. The `.env` file is gitignored and will not be committed since that's a secret key

### 3. Download models

Edit `scripts/install_llama2.py` to uncomment the models you want to download, then run:

```bash
python scripts/install_llama2.py
```

Available models (comment/uncomment in the script):

```python
# TinyLlama 1.1B — draft model (~2.2 GB)
download_hf_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./models/tinyllama-1.1b")

# Llama-2 7B — target model (~13 GB)
download_hf_model("meta-llama/Llama-2-7b-hf", "./models/llama2-7b")
```

Models are saved to `./models/` which is gitignored.

---

## Running Experiments

### Speculative Decoding Experiment

Runs baseline (1.1B), baseline (7B), and four speculative decoding configurations across the same prompt. Results saved to `./results/`.

```bash
python run_experiment.py
```

Scenarios run:
- Llama2 1.1B baseline
- Llama2 7B baseline
- Speculative (no KV cache, no adaptive k)
- Speculative + adaptive k
- Speculative + KV cache
- Speculative + KV cache + adaptive k (full)

### Multi-Token Pipeline Experiment

Runs the multi-token pipeline architecture with and without adaptive k. Results saved to `./results/`.

```bash
python multi_token_pipeline_experiment.py
```

Requires a CUDA GPU — will raise an error immediately if none is found.

---

## Project Structure

```
.
├── run_experiment.py                   # Speculative decoding experiment runner
├── multi_token_pipeline_experiment.py  # Multi-token pipeline experiment runner
├── requirements.txt
│
├── src/
│   ├── speculative_decoder.py          # Core speculative decoding implementation
│   ├── baseline_decoder.py             # Autoregressive baseline decoder
│   ├── multi_token_pipeline.py         # Multi-token pipeline architecture
│   ├── kv_cache.py                     # Logical KV cache (committed/speculative state)
│   └── performance_model.py            # Analytical speedup model
│
├── scripts/
│   └── install_llama2.py               # Model downloader (requires HF token)
│
├── tests/
│   └── test_decoders.py
│
├── results/                            # CSV output from experiment runs
│   └── multitoken/
│       ├── n50/
│       ├── n100/
│       ├── n200/
│       └── combined.csv                # Merged multi-token results across all N
│
├── models/                             # Downloaded model weights (gitignored)
│   ├── tinyllama-1.1b/
│   └── llama2-7b/
│
├── figures/                            # Plots and diagrams
├── paper/                              # Paper writeup
└── slides/                             # Presentation slides
```

---

## Key Files

| File | Description |
|---|---|
| [src/speculative_decoder.py](src/speculative_decoder.py) | Draft generation, parallel verification, adaptive k, HF KV cache integration |
| [src/baseline_decoder.py](src/baseline_decoder.py) | Token-by-token autoregressive decoding for benchmarking |
| [src/multi_token_pipeline.py](src/multi_token_pipeline.py) | Pipeline architecture with speculative buffer, rollback, and pipeline metrics |
| [src/kv_cache.py](src/kv_cache.py) | Tracks committed vs speculative token state for cache correctness |
| [scripts/performance_model.py](scripts/performance_model.py) | Analytical speedup formula S(alpha, k, beta), optimal k tables, empirical validation |

---

## Configuration

Key parameters in each experiment file:

| Parameter | Description |
|---|---|
| `PROMPT` | Input prompt text |
| `N` | Number of tokens to generate |
| `k` | Speculation depth (defaults to 10% of N) |
| `kv_cache` | Enable HuggingFace past_key_values for O(1) draft steps |
| `adaptive_k` | Dynamically adjust k based on rolling acceptance rate |
