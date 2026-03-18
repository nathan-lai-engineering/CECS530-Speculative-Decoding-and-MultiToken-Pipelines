import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm, trange
import time
from kv_cache import KVCache


class SpeculativeDecoder:

    def __init__(self, draft_model_path, target_model_path, adaptive_k=True, kv_cache=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.draft_model_path = draft_model_path
        self.target_model_path = target_model_path
        self.adaptive_k = adaptive_k
        self.use_kv_cache = kv_cache

        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.float16).to(self.device)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16).to(self.device)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)

        self.draft_model.eval()
        self.target_model.eval()

        
        # metrics
        self.reset_metrics()


    # reset all the metrics
    def reset_metrics(self):
        self.accepted_tokens = 0
        self.total_draft_tokens = 0
        self.output_tokens = 0

        self.total_time = 0
        self.total_draft_time = 0
        self.total_target_time = 0

        self.max_draft_time = 0
        self.max_target_time = 0

        self.min_draft_time = float('inf')
        self.min_target_time = float('inf')

        self.draft_past_kv = None
        self.cache = KVCache() if self.use_kv_cache else None

    def trim_cache(self, past_key_values, length):
        import copy
        trimmed = copy.deepcopy(past_key_values)
        trimmed.crop(length)
        return trimmed

    # generate n tokens with k speculative depth
    # depending on adaptive_k flag, will use adaptive speculation depth
    def generate_k_tokens(self, prompt, n=20, k=5):
        current_n = n
        current_ids = self.draft_tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        current_k = k
        eos_id = self.draft_tokenizer.eos_token_id


        # keep running batches of k until we have n tokens
        while(current_n > 0):
            # Generate k draft tokens
            draft_ids, draft_probs, draft_kv = self.generate_k_draft_tokens(
                current_ids, k=max(1, min(current_n - 1, current_k)),
                past_kv=self.draft_past_kv if self.use_kv_cache else None
            )

            # Verify tokens in parallel
            accepted_before = self.accepted_tokens
            verified_tokens = self.parallel_verification(draft_ids, draft_probs, current_ids.shape[-1])

            if verified_tokens.dim() == 1:
                verified_tokens = verified_tokens.unsqueeze(0)

            if self.use_kv_cache:
                accepted_this_round = self.accepted_tokens - accepted_before
                self.draft_past_kv = self.trim_cache(draft_kv, current_ids.shape[-1] + accepted_this_round)


            # Update prompt and remaining n
            current_n -= verified_tokens.shape[-1] - current_ids.shape[-1]
            #print(current_n)
            current_ids = verified_tokens

            # adaptive speculative depth, increase draft tokens as we increase confidence
            if self.adaptive_k:
                acceptance_rate = self.accepted_tokens / self.total_draft_tokens
                if acceptance_rate < 0.4:
                    current_k = int(max(2, current_k / 2))
                elif acceptance_rate > 0.85:
                    current_k = int(min(k * 2, current_k + 2))

            if eos_id is not None and eos_id in verified_tokens:
                break

        return verified_tokens

    # generates k draft tokens
    def generate_k_draft_tokens(self, input_ids, k, past_kv=None):
        draft_tokens = input_ids.clone()
        draft_token_probs = []
        new_draft_ids = []
        eos_id = self.draft_tokenizer.eos_token_id

        use_hf_cache = past_kv is not None or self.use_kv_cache

        if use_hf_cache:
            # prime cache: process tokens not yet cached
            cache_len = past_kv.get_seq_length() if past_kv is not None else 0
            if cache_len >= input_ids.shape[-1]:
                past_kv = None
                cache_len = 0
            tokens_to_process = input_ids[:, cache_len:]
            with torch.inference_mode():
                prime_output = self.draft_model(tokens_to_process, past_key_values=past_kv, use_cache=True)
                current_past = prime_output.past_key_values
                next_probs = torch.softmax(prime_output.logits[:, -1, :], dim=-1)

        for _ in trange(k, desc="Generating draft tokens"):
            with torch.inference_mode():
                start_time = time.time()

                if use_hf_cache:
                    token = torch.multinomial(next_probs, num_samples=1)
                else:
                    output = self.draft_model(draft_tokens)
                    probs = torch.softmax(output.logits[:, -1, :], dim=-1)
                    token = torch.multinomial(probs, num_samples=1)

                draft_tokens = torch.cat([draft_tokens, token], dim=-1)
                draft_token_probs.append((next_probs if use_hf_cache else probs).squeeze(0)[token.item()])
                new_draft_ids.append(token.item())

                elapsed_time = time.time() - start_time
                self.total_draft_time += elapsed_time
                self.total_time += elapsed_time
                self.total_draft_tokens += 1
                self.max_draft_time = max(self.max_draft_time, elapsed_time)
                self.min_draft_time = min(self.min_draft_time, elapsed_time)

                if eos_id is not None and token.item() == eos_id:
                    break

                if use_hf_cache:
                    output = self.draft_model(token, past_key_values=current_past, use_cache=True)
                    current_past = output.past_key_values
                    next_probs = torch.softmax(output.logits[:, -1, :], dim=-1)

        if self.cache is not None:
            self.cache.add_speculative(new_draft_ids)
        return draft_tokens, draft_token_probs, current_past if use_hf_cache else None

    # single forward pass across draft tokens and their probs
    def parallel_verification(self, draft_tokens, draft_token_probs, prompt_len):
        eos_id = self.target_tokenizer.eos_token_id
        k = len(draft_token_probs)

        #print("Verifying", k, "draft tokens")
        with torch.inference_mode():
            start_time = time.time()

            # Pass prompt + K draft tokens into target model in one single forward pass
            output = self.target_model(draft_tokens)

            # Generate probabilities from target model
            target_token_probs = output.logits  # use logits directly, argmax is identical

            elapsed_time = time.time() - start_time
            self.total_target_time += elapsed_time
            self.total_time += elapsed_time
            self.max_target_time = max(self.max_target_time, elapsed_time)
            self.min_target_time = min(self.min_target_time, elapsed_time)

            # if kv-cache is being used
            if self.cache is not None:
                target_ids = target_token_probs[0][-(k+1):-1].argmax(dim=-1).tolist()

                accepted_ids = self.cache.verify(target_ids)
                accepted_count = len(accepted_ids)

                # save accepted tokens then clear draft tokens
                self.cache.commit(accepted_ids)
                self.cache.rollback()

                self.accepted_tokens += accepted_count
                self.output_tokens += accepted_count

                result = draft_tokens[:, :prompt_len + accepted_count]

                # there is a rejection, we did not accept full depth
                if accepted_count < k:
                    reject_pos = -(k - accepted_count + 1)
                    correct_token = target_token_probs[0][reject_pos].argmax().unsqueeze(0)
                    result = torch.cat([result[0], correct_token], dim=-1).unsqueeze(0)
                    self.output_tokens += 1
                    self.cache.commit(accepted_ids + [correct_token.item()])
                    self.cache.rollback()
                    if eos_id is not None and correct_token.item() == eos_id:
                        return result
                    
                # we accepted full depth and we can claim a bonus token
                else:
                    bonus = target_token_probs[0][-1].argmax().unsqueeze(0)
                    result = torch.cat([result[0], bonus], dim=-1).unsqueeze(0)
                    self.output_tokens += 1
                    self.cache.commit(accepted_ids + [bonus.item()])
                    self.cache.rollback()

                return result
            else:
                # stochastic acceptance: apply softmax here since target_token_probs holds raw logits
                target_probs = torch.softmax(target_token_probs, dim=-1)
                for i in range(k, 0, -1):
                    index = draft_tokens[0][-i].item()

                    self.output_tokens += 1

                    target_token_prob = target_probs[0][-(i+1)][index].item()
                    draft_token_prob = draft_token_probs[-i].item()
                    accept_prob = min(1.0, target_token_prob / (draft_token_prob + 1e-8))

                    if torch.rand(1).item() < accept_prob:
                        self.accepted_tokens += 1
                        if eos_id is not None and index == eos_id:
                            return draft_tokens
                    else:
                        # adjusted distribution: clamp(p_target - p_draft, min=0) then renormalize
                        draft_full_probs = torch.zeros_like(target_probs[0][-(i+1)])
                        draft_full_probs[index] = draft_token_prob
                        adjusted = torch.clamp(target_probs[0][-(i+1)] - draft_full_probs, min=0)
                        adjusted = adjusted / (adjusted.sum() + 1e-8)

                        draft_tokens = draft_tokens[:, :-i]
                        correct_token = torch.multinomial(adjusted, num_samples=1)
                        draft_tokens = torch.cat([draft_tokens[0], correct_token], dim=-1).unsqueeze(0)

                        if eos_id is not None and correct_token.item() == eos_id:
                            return draft_tokens
                        break


        return draft_tokens

    # string to input_ids
    def encode(self, text):
        return self.draft_tokenizer.encode(text, return_tensors="pt").to(
            next(self.draft_model.parameters()).device
        )

    # token_ids to string
    def decode(self, token_ids):
        if token_ids.dim() == 1:
            return self.target_tokenizer.decode(token_ids, skip_special_tokens=True)
        return self.target_tokenizer.decode(token_ids[0], skip_special_tokens=True)

    # the metrics to look at
    def token_throughput(self):
        return {
            "tokens_per_second": self.output_tokens / self.total_time,
            "total_draft_tokens": self.total_draft_tokens,
            "total_time": self.total_time,
            "mean_time_per_token": self.total_time / self.output_tokens,
            "max_time_per_draft_token": self.max_draft_time, 
            "min_time_per_draft_token": self.min_draft_time,
            "acceptance_rate": self.accepted_tokens / self.total_draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "output_tokens": self.output_tokens
        }
