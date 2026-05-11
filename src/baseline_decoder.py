import time
import torch
from tqdm import tqdm, trange

from transformers import AutoTokenizer, AutoModelForCausalLM

class BaselineDecoder:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self._model_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

    # generate k tokens with k forward passes
    # returns input tokens + k number of output tokens
    def generate_k_tokens(self, input_ids, n, warmup=True):
        if warmup:
            with torch.inference_mode():
                self.model(input_ids=input_ids)

        # reset the metrics for this generation
        self.total_tokens = 0
        self.total_time = 0
        self.max_time_per_token = 0
        self.min_time_per_token = float('inf')
        self.peak_memory_bandwidth_GB_per_s = 0.0

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        tokens = input_ids.clone()
        eos_id = self.tokenizer.eos_token_id

        # k passes 
        for _ in trange(n, desc="Performing forward passes"):
            start_time = time.time()

            next_token = self.forward(tokens)

            # append the single token
            tokens = torch.cat([tokens, next_token], dim=-1)

            # update those metrics
            elapsed_time = time.time() - start_time
            self.total_time += elapsed_time
            self.total_tokens += 1
            self.max_time_per_token = max(self.max_time_per_token, elapsed_time)
            self.min_time_per_token = min(self.min_time_per_token, elapsed_time)
            self.peak_memory_bandwidth_GB_per_s = max(
                self.peak_memory_bandwidth_GB_per_s,
                (self._model_bytes / 1e9) / elapsed_time
            )

            if eos_id is not None and next_token.item() == eos_id:
                break

        if self.device == "cuda":
            self.peak_memory_MB = torch.cuda.max_memory_allocated() / 1e6
        else:
            self.peak_memory_MB = 0.0

        return tokens

    # forward pass for a token
    def forward(self, input_ids):
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids)

        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token_id

    def token_throughput(self):
        if hasattr(self, "total_tokens") and hasattr(self, "total_time"):
            total_bytes = self._model_bytes * self.total_tokens
            return {
                "tokens_per_second": self.total_tokens / self.total_time,
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
                "mean_time_per_token": self.total_time / self.total_tokens,
                "max_time_per_token": self.max_time_per_token,
                "min_time_per_token": self.min_time_per_token,
                "accepted_tokens": 0,
                "total_draft_tokens": 0,
                "verification_rounds": 0,
                "total_draft_time": 0.0,
                "total_target_time": 0.0,
                "peak_memory_MB": getattr(self, "peak_memory_MB", 0.0),
                "model_memory_MB": self._model_bytes / 1e6,
                "memory_bandwidth_GB_per_s": (total_bytes / 1e9) / self.total_time,
                "peak_memory_bandwidth_GB_per_s": getattr(self, "peak_memory_bandwidth_GB_per_s", 0.0),
            }
        else:
            return {
                "tokens_per_second": 0,
                "total_tokens": 0,
                "total_time": 0,
                "mean_time_per_token": 0,
                "max_time_per_token": 0,
                "min_time_per_token": float('inf'),
                "accepted_tokens": 0,
                "total_draft_tokens": 0,
                "verification_rounds": 0,
                "total_draft_time": 0.0,
                "total_target_time": 0.0,
                "peak_memory_MB": 0.0,
                "model_memory_MB": self._model_bytes / 1e6,
                "memory_bandwidth_GB_per_s": 0.0,
                "peak_memory_bandwidth_GB_per_s": 0.0,
            }
        
    # string to input_ids
    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )

    # token_ids to string
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
