# src/multi_token_pipeline.py

from dataclasses import dataclass
import time
import torch

from src.speculative_decoder import SpeculativeDecoder


@dataclass
class PipelineBatch:
    batch_id: int
    base_ids: torch.Tensor
    draft_tokens: torch.Tensor
    prompt_len: int
    draft_count: int
    ready_time: float


class MultiTokenPipeline:
    """
    Independent pipeline model for Goal 2.

    This does NOT replace speculative_decoder.py.
    It models:
    - draft stage
    - verify stage
    - speculative buffer / queue
    - rollback flushing younger speculative batches
    - pipeline metrics
    """

    def __init__(self, draft_model_path, target_model_path, adaptive_k=True, buffer_capacity=2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer_capacity = max(1, int(buffer_capacity))
        self.adaptive_k = adaptive_k

        # Reuse your existing speculative decoder for draft generation helpers
        self.decoder = SpeculativeDecoder(
            draft_model_path=draft_model_path,
            target_model_path=target_model_path,
            adaptive_k=adaptive_k,
            kv_cache=False
        )

        self.draft_model = self.decoder.draft_model
        self.target_model = self.decoder.target_model
        self.draft_tokenizer = self.decoder.draft_tokenizer
        self.target_tokenizer = self.decoder.target_tokenizer

        self.reset_metrics()

    def reset_metrics(self):
        self.decoder.reset_metrics()

        self.accepted_tokens = 0
        self.output_tokens = 0
        self.total_draft_tokens = 0
        self.verification_rounds = 0

        self.total_draft_time = 0.0
        self.total_target_time = 0.0

        self.max_draft_time = 0.0
        self.max_target_time = 0.0
        self.min_draft_time = float("inf")
        self.min_target_time = float("inf")

        # pipeline stats
        self.pipeline_total_time = 0.0
        self.draft_stage_busy_time = 0.0
        self.verify_stage_busy_time = 0.0
        self.pipeline_bubbles = 0
        self.rollback_events = 0
        self.flushed_batches = 0
        self.max_buffer_occupancy = 0
        self.batches_drafted = 0
        self.batches_verified = 0

        # stage clocks
        self.draft_stage_free_at = 0.0
        self.verify_stage_free_at = 0.0

        # adaptive-k window
        self.window_accepted = 0
        self.window_drafts = 0

    def encode(self, text):
        return self.draft_tokenizer.encode(text, return_tensors="pt").to(
            next(self.draft_model.parameters()).device
        )

    def decode(self, token_ids):
        if token_ids.dim() == 1:
            return self.target_tokenizer.decode(token_ids, skip_special_tokens=True)
        return self.target_tokenizer.decode(token_ids[0], skip_special_tokens=True)

    def _draft_batch(self, base_ids, k):
        before = self.decoder.total_draft_time

        draft_tokens, draft_probs, _ = self.decoder.generate_k_draft_tokens(
            input_ids=base_ids,
            k=k,
            past_kv=None
        )

        elapsed = self.decoder.total_draft_time - before

        self.total_draft_time += elapsed
        self.total_draft_tokens += len(draft_probs)
        self.max_draft_time = max(self.max_draft_time, elapsed)
        self.min_draft_time = min(self.min_draft_time, elapsed)

        start = self.draft_stage_free_at
        end = start + elapsed
        self.draft_stage_free_at = end
        self.draft_stage_busy_time += elapsed

        self.batches_drafted += 1

        return PipelineBatch(
            batch_id=self.batches_drafted,
            base_ids=base_ids.clone(),
            draft_tokens=draft_tokens.clone(),
            prompt_len=base_ids.shape[-1],
            draft_count=len(draft_probs),
            ready_time=end
        )

    def _verify_batch(self, batch):
        """
        Verify one batch.
        IMPORTANT: no bonus-token optimization here.
        """
        with torch.inference_mode():
            wall_start = time.time()
            output = self.target_model(batch.draft_tokens)
            elapsed = time.time() - wall_start

        logits = output.logits
        prompt_len = batch.prompt_len
        k = batch.draft_count

        accepted_count = 0
        mismatch = False

        for j in range(k):
            draft_token = batch.draft_tokens[0][prompt_len + j].item()
            target_token = logits[0][prompt_len - 1 + j].argmax().item()

            if draft_token == target_token:
                accepted_count += 1
            else:
                mismatch = True
                break

        result = batch.draft_tokens[:, :prompt_len + accepted_count]

        if mismatch:
            correct_token = logits[0][prompt_len - 1 + accepted_count].argmax().unsqueeze(0)
            result = torch.cat([result[0], correct_token], dim=-1).unsqueeze(0)

        emitted_count = result.shape[-1] - prompt_len

        # stage timing
        verify_start = max(self.verify_stage_free_at, batch.ready_time)
        if verify_start > self.verify_stage_free_at:
            self.pipeline_bubbles += 1

        verify_end = verify_start + elapsed
        self.verify_stage_free_at = verify_end
        self.verify_stage_busy_time += elapsed
        self.pipeline_total_time = max(self.pipeline_total_time, verify_end)

        self.total_target_time += elapsed
        self.max_target_time = max(self.max_target_time, elapsed)
        self.min_target_time = min(self.min_target_time, elapsed)

        self.verification_rounds += 1
        self.batches_verified += 1

        return result, accepted_count, emitted_count, mismatch

    def generate_k_tokens(self, prompt, n=20, k=4, warmup=True):
        self.reset_metrics()

        committed_ids = self.draft_tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        queue = []
        remaining = n
        current_k = k
        eos_id = self.target_tokenizer.eos_token_id

        if warmup:
            with torch.inference_mode():
                self.draft_model(committed_ids)
                self.target_model(committed_ids)

        while remaining > 0:
            # fill buffer
            while len(queue) < self.buffer_capacity and remaining > 0:
                if len(queue) == 0:
                    base_ids = committed_ids
                else:
                    # younger batch drafts off latest speculative context
                    base_ids = queue[-1].draft_tokens

                batch_k = max(1, min(remaining, current_k))
                batch = self._draft_batch(base_ids, batch_k)
                queue.append(batch)

                self.max_buffer_occupancy = max(self.max_buffer_occupancy, len(queue))

                drafted_tail = batch.draft_tokens[0][batch.prompt_len:]
                if eos_id is not None and eos_id in drafted_tail:
                    break

            if not queue:
                self.pipeline_bubbles += 1
                break

            oldest = queue.pop(0)
            verified_ids, accepted_count, emitted_count, mismatch = self._verify_batch(oldest)

            committed_ids = verified_ids
            self.accepted_tokens += accepted_count
            self.output_tokens += emitted_count
            remaining -= emitted_count

            if mismatch:
                self.rollback_events += 1
                if queue:
                    self.flushed_batches += len(queue)
                    queue.clear()

            self.window_accepted += accepted_count
            self.window_drafts += max(1, oldest.draft_count)

            if self.adaptive_k and self.window_drafts >= 10:
                rate = self.window_accepted / self.window_drafts
                optimal_k = int(1 / (1 - rate + 0.01))
                current_k = max(k, min(k * 2, optimal_k))
                self.window_accepted = 0
                self.window_drafts = 0

            if eos_id is not None and committed_ids[0, -1].item() == eos_id:
                break

        self.pipeline_total_time = max(
            self.pipeline_total_time,
            self.draft_stage_free_at,
            self.verify_stage_free_at
        )

        return committed_ids

    def token_throughput(self):
        total_time = self.pipeline_total_time if self.pipeline_total_time > 0 else (
            self.total_draft_time + self.total_target_time
        )

        return {
            "tokens_per_second": self.output_tokens / total_time if total_time > 0 else 0.0,
            "total_tokens": self.output_tokens,
            "total_time": total_time,
            "mean_time_per_token": total_time / self.output_tokens if self.output_tokens > 0 else 0.0,
            "max_time_per_token": max(self.max_draft_time, self.max_target_time),
            "min_time_per_token": min(self.min_draft_time, self.min_target_time)
            if self.min_draft_time != float("inf") and self.min_target_time != float("inf")
            else 0.0,
            "accepted_tokens": self.accepted_tokens,
            "total_draft_tokens": self.total_draft_tokens,
            "verification_rounds": self.verification_rounds,
            "total_draft_time": self.total_draft_time,
            "total_target_time": self.total_target_time,
            "pipeline_total_time": self.pipeline_total_time,
            "draft_stage_busy_time": self.draft_stage_busy_time,
            "verify_stage_busy_time": self.verify_stage_busy_time,
            "pipeline_bubbles": self.pipeline_bubbles,
            "rollback_events": self.rollback_events,
            "flushed_batches": self.flushed_batches,
            "max_buffer_occupancy": self.max_buffer_occupancy,
            "buffer_capacity": self.buffer_capacity,
            "batches_drafted": self.batches_drafted,
            "batches_verified": self.batches_verified,
        }