import time
import torch
from tqdm import tqdm, trange

from transformers import AutoTokenizer, AutoModelForCausalLM

# maintained by nathan feel free to edit
# baseline decoder to do full forward pass for a token
class BaselineDecoder:

    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # so you can choose big model or small model
        self.model_path = model_path 

        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()




    # generate k tokens with k forward passes
    # returns input tokens + k number of output tokens
    def generate_k_tokens(self, input_ids, n):

        # reset the metrics for this generation
        self.total_tokens = 0
        self.total_time = 0
        self.max_time_per_token = 0
        self.min_time_per_token = float('inf')

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

            if eos_id is not None and next_token.item() == eos_id:
                break
        
        return tokens

    # forward pass for a token
    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token_id

    # the metrics to look at
    def token_throughput(self):
        if hasattr(self, "total_tokens") and hasattr(self, "total_time"):
            return {
                "tokens_per_second": self.total_tokens / self.total_time,
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
                "mean_time_per_token": self.total_time / self.total_tokens,
                "max_time_per_token": self.max_time_per_token, 
                "min_time_per_token": self.min_time_per_token
            }
        else:
            print("no generations ran yet, so no throughput really")
            return {
                "tokens_per_second": 0,
                "total_tokens": 0,
                "total_time": 0,
                "mean_time_per_token": 0,
                "max_time_per_token": 0, 
                "min_time_per_token": float('inf')
            }
        
    # string to input_ids
    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )

    # token_ids to string
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)