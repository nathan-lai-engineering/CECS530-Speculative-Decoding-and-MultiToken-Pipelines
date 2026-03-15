import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

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

    def generate_k_draft_tokens(self, prompt, k):
        # Tokenize input prompt
        tokenized_inputs = self.draft_tokenizer(prompt, return_tensors="pt")

        # As model generates draft tokens, it is added into draft inputs
        draft_tokens = tokenized_inputs["input_ids"].clone()
        draft_token_probs = []

        for _ in range(k):
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

        return draft_tokens, draft_token_probs

    def parallel_verification(self, draft_tokens, draft_token_probs, k):
        with torch.no_grad():
            # Pass prompt + K draft tokens into target model in one single forward pass
            output = self.target_model(draft_tokens)

            # Generate probabilities from target model
            target_token_probs = torch.softmax(output.logits, dim=-1)

            # Compare probability output of draft token from target model vs. draft model
            for i in range(k, 0, -1):
                index = draft_tokens[0][-i].item()
                
                target_token_prob = target_token_probs[0][-i][index].item()
                draft_token_prob = draft_token_probs[-i].item()

                if target_token_prob >= draft_token_prob:
                    # Accept token into sequence
                    continue

                else:
                    # Rollback behavior
                    draft_tokens = draft_tokens[0][:-i]

                    # Sample from target token probabilities the correct word
                    correct_token = torch.multinomial(target_token_probs[0][-i], num_samples=1)

                    # Add correct token into sequence
                    draft_tokens = torch.cat([draft_tokens, correct_token], dim=-1)
                    break
            
        return draft_tokens