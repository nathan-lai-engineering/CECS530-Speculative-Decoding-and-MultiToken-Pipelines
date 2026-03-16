import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm, trange

class SpeculativeDecoder:
    def __init__(self, draft_model_path, target_model_path):
        self.draft_model_path = draft_model_path
        self.target_model_path = target_model_path

        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.float16)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)

        self.draft_model.eval()
        self.target_model.eval()

    def generate_k_tokens(self, prompt, n=20, k=5):
        current_n = n
        current_ids = self.draft_tokenizer(prompt, return_tensors="pt")["input_ids"]
        eos_id = self.draft_tokenizer.eos_token_id


        # keep running batches of k until we have n tokens
        while(current_n > 0):
            # Generate k draft tokens
            draft_ids, draft_probs = self.generate_k_draft_tokens(current_ids, k=min(current_n, k))

            # Verify tokens in parallel
            verified_tokens = self.parallel_verification(draft_ids, draft_probs)
            
            if verified_tokens.dim() == 1:
                verified_tokens = verified_tokens.unsqueeze(0)

            # Update prompt and remaining n
            current_n -= verified_tokens.shape[-1] - current_ids.shape[-1]
            print(current_n)
            current_ids = verified_tokens

            if eos_id is not None and eos_id in verified_tokens:
                break

        return verified_tokens

    def generate_k_draft_tokens(self, input_ids, k):
        # As model generates draft tokens, it is added into draft inputs
        draft_tokens = input_ids.clone()
        draft_token_probs = []
        eos_id = self.draft_tokenizer.eos_token_id

        for _ in trange(k, desc="Generating draft tokens"):
            with torch.no_grad():
                # Pass tokenized input to draft model
                output = self.draft_model(draft_tokens)

                # Apply softmax on token logits to produce probabilities
                probs = torch.softmax(output.logits[:, -1, :], dim=-1)

                # Sample from probabilities; output is an index of that token from probs
                token = torch.multinomial(probs, num_samples=1)

                # Add new tokens into draft tokens
                draft_tokens = torch.cat([draft_tokens, token], dim=-1)
                draft_token_probs.append(probs.squeeze(0)[token.item()])

                if eos_id is not None and token.item() == eos_id:
                    break

        return draft_tokens, draft_token_probs

    def parallel_verification(self, draft_tokens, draft_token_probs):
        eos_id = self.target_tokenizer.eos_token_id
        k = len(draft_token_probs)

        print("Verifying", k, "draft tokens")
        with torch.no_grad():

            # Pass prompt + K draft tokens into target model in one single forward pass
            output = self.target_model(draft_tokens)

            # Generate probabilities from target model
            target_token_probs = torch.softmax(output.logits, dim=-1)

            # Compare probability output of draft token from target model vs. draft model
            for i in range(k, 0, -1):
                index = draft_tokens[0][-i].item()
                
                target_token_prob = target_token_probs[0][-(i+1)][index].item()
                draft_token_prob = draft_token_probs[-i].item()

                if target_token_prob >= draft_token_prob:

                    # early stopping
                    if eos_id is not None and index == eos_id:
                        return draft_tokens

                    # Accept token into sequence
                    continue

                else:
                    # Rollback behavior
                    draft_tokens = draft_tokens[0][:-i]

                    # Sample from target token probabilities the correct word
                    correct_token = torch.multinomial(target_token_probs[0][-(i+1)], num_samples=1)

                    # Add correct token into sequence
                    draft_tokens = torch.cat([draft_tokens, correct_token], dim=-1)
                
                    # early stopping
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

    # placeholder — timing will be added in Week 3 benchmark
    def token_throughput(self):
        return {"note": "speculative throughput metrics coming in Week 3"}