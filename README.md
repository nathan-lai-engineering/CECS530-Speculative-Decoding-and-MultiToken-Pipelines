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

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA, install separately matching your CUDA version:

```bash
# CUDA 13 (used in this project)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. Set up HuggingFace token

Llama-2 requires a HuggingFace account with access granted at [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).

Create a `.env` file in the project root:

```
HF_TOKEN="hf_your_token_here"
```

Get your token from huggingface.co/settings/tokens. The `.env` file is gitignored and will not be committed since that's a secret key

### 4. Download models

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
python run_experiment.py [--n N] [--prompt PROMPT] [--loops LOOPS] [--increment INCREMENT]
```

| Argument | Default | Description |
|---|---|---|
| `--n` | `100` | Number of tokens to generate |
| `--prompt` | `"The first digits of pi are "` | Input prompt |
| `--loops` | `1` | Number of times to repeat the experiment at the current N |
| `--increment` | `1` | Number of times to double N and repeat all loops |

Examples:

```bash
# default run, n=100
python run_experiment.py

# 5 repetitions at n=200
python run_experiment.py --n 200 --loops 5

# n=50 → n=100 → n=200, 3 runs each
python run_experiment.py --n 50 --loops 3 --increment 3

# custom prompt
python run_experiment.py --n 100 --prompt "Once upon a time"
```

Scenarios run per loop:
- Llama2 1.1B baseline
- Llama2 7B baseline
- Speculative (no KV cache, no adaptive k)
- Speculative + adaptive k
- Speculative + KV cache
- Speculative + KV cache + adaptive k (full)

### Multi-Token Pipeline Experiment

Runs the multi-token pipeline architecture with and without adaptive k. Results saved to `./results/`. Requires a CUDA GPU — will raise an error immediately if none is found.

```bash
python multi_token_pipeline_experiment.py [--n N] [--prompt PROMPT] [--loops LOOPS] [--increment INCREMENT]
```

| Argument | Default | Description |
|---|---|---|
| `--n` | `50` | Number of tokens to generate |
| `--prompt` | `"The first digits of pi are "` | Input prompt |
| `--loops` | `5` | Number of times to repeat the experiment at the current N |
| `--increment` | `1` | Number of times to double N and repeat all loops |

Examples:

```bash
# default run, n=50, 5 loops
python multi_token_pipeline_experiment.py

# single run at n=200
python multi_token_pipeline_experiment.py --n 200 --loops 1

# n=50 → n=100 → n=200, 5 runs each
python multi_token_pipeline_experiment.py --n 50 --loops 5 --increment 3
```

Scenarios run per loop:
- Multi-Token Pipeline (fixed k)
- Multi-Token Pipeline with Adaptive K

### Analytical Performance Model (optional)

After running experiments, you can compute the theoretical speedup model from the sequential baseline results:

```bash
python scripts/performance_model.py
```

Reads `results/sequential/new data/sequential combined.csv` automatically. Outputs:

- Empirical beta (T_draft / T_target) per N and overall
- Speedup table across acceptance rates (alpha) and speculation depths (k)
- Optimal k for each acceptance rate
- Diminishing returns analysis at alpha=0.6
- Speedup vs sequence length N

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

Runtime parameters are passed as command-line arguments (see Running Experiments above). Internal parameters fixed in source:

| Parameter | Location | Description |
|---|---|---|
| `k` | `predict()` in each script | Speculation depth — defaults to 10% of N, minimum 2 |
| `kv_cache` | `run_experiment.py` | Enable HuggingFace `past_key_values` for O(1) draft steps |
| `adaptive_k` | both scripts | Dynamically adjust k based on rolling acceptance rate |
| `buffer_capacity` | `multi_token_pipeline_experiment.py` | Number of speculative batches held in the pipeline buffer |

---

## Output Metrics

### Common metrics (both scripts)

These columns appear in every output CSV.

| Column | Definition | Calculation |
|---|---|---|
| `tokens_per_second` | Output tokens produced per wall-clock second | `total_tokens / total_time` |
| `total_tokens` | Number of output tokens generated (excludes prompt) | Incremented by 1 per accepted/emitted token |
| `total_time` | Total wall-clock time for the generation loop | Sum of all forward-pass elapsed times (draft + target); for the pipeline script, equals `pipeline_total_time` (simulated parallel time, not the sequential sum) |
| `mean_time_per_token` | Average wall time per output token | `total_time / total_tokens` |
| `max_time_per_token` | Slowest single forward pass recorded | Running max over draft-pass wall times; for the pipeline script, `max(max_draft_time, max_target_time)` |
| `min_time_per_token` | Fastest single forward pass recorded | Running min over draft-pass wall times; for the pipeline script, `min(min_draft_time, min_target_time)` |
| `accepted_tokens` | Draft tokens accepted by the target model | Count of positions where `argmax(target_logit) == draft_token` |
| `total_draft_tokens` | Total draft tokens generated before verification | Incremented by 1 for each token the draft model produces |
| `verification_rounds` | Number of parallel target-model verification calls | Incremented by 1 per call to `parallel_verification` / `_verify_batch` |
| `total_draft_time` | Cumulative wall time spent in draft forward passes | Sum of elapsed time for every draft model forward pass |
| `total_target_time` | Cumulative wall time spent in target forward passes | Sum of elapsed time for every target model forward pass |
| `peak_memory_MB` | Peak GPU memory allocated during generation | `torch.cuda.max_memory_allocated() / 1e6` after generation loop |
| `model_memory_MB` | Static size of loaded model weights | `sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6`; both draft and target summed for speculative runs |
| `memory_bandwidth_GB_per_s` | Average memory bandwidth over the full run | `(model_bytes * num_passes / 1e9) / total_time`; assumes each forward pass reads all weights once |
| `peak_memory_bandwidth_GB_per_s` | Highest per-pass bandwidth observed | `max(model_bytes / 1e9 / elapsed)` taken over every individual forward pass |

### Pipeline-only metrics (multi_token_pipeline_experiment.py)

These additional columns appear only in pipeline CSV output.

| Column | Definition | Calculation |
|---|---|---|
| `pipeline_total_time` | Simulated total time assuming draft and verify run on separate hardware in parallel | `max(verify_end_timestamp)` across all batches, where each verify stage starts as soon as the batch is ready and the verify stage is free |
| `draft_stage_busy_time` | Total time the draft stage was actively generating tokens | Sum of elapsed time for every `_draft_batch` call |
| `verify_stage_busy_time` | Total time the verify stage was actively verifying batches | Sum of elapsed time for every `_verify_batch` call |
| `pipeline_bubbles` | Number of times the verify stage had to wait because no batch was ready | Incremented when `batch.ready_time > verify_stage_free_at` (verify idle waiting for draft), and when the queue is empty |
| `rollback_events` | Number of verification rounds that produced a token mismatch | Incremented once per `_verify_batch` call that returns `mismatch=True` |
| `flushed_batches` | Total speculative batches discarded due to rollbacks | Sum of `len(queue)` at the moment of each rollback flush |
| `max_buffer_occupancy` | Largest number of unverified batches in the queue at any point | Running max of `len(queue)` after each `_draft_batch` |
| `buffer_capacity` | Configured maximum queue depth | Set at construction (`buffer_capacity` argument), held constant |
| `batches_drafted` | Total number of draft batches produced | Incremented by 1 in every `_draft_batch` call |
| `batches_verified` | Total number of batches sent to the target for verification | Incremented by 1 in every `_verify_batch` call |
| `cpu_ram_delta_MB` | Increase in process RSS memory during generation | `(psutil.Process().memory_info().rss_after - rss_before) / 1e6`; positive values indicate tensors were moved to system RAM |
| `gpu_utilization_pct` | Peak GPU memory as a percentage of total device VRAM | `peak_memory_MB / (torch.cuda.get_device_properties(0).total_memory / 1e6) * 100` |
| `acceptance_rate` | Fraction of draft tokens accepted by the target | `accepted_tokens / total_draft_tokens` |
| `rollback_rate` | Fraction of verification rounds that triggered a rollback | `rollback_events / verification_rounds` |
