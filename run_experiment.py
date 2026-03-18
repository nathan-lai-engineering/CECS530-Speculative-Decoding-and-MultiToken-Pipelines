import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".","src"))
from baseline_decoder import BaselineDecoder
from speculative_decoder import SpeculativeDecoder

# predict using either baseline or speculative method on given model
def predict(prompt, model_path, n, type="baseline", target_path=None, k=None, kv_cache=True, adaptive_k=True):
    if k is None:
        k = max(2, int(0.10 * n))

    match type:
        case "baseline":
            decoder = BaselineDecoder(model_path)
            input_ids = decoder.encode(prompt)
            output_ids = decoder.generate_k_tokens(input_ids, n)
            output_text = decoder.decode(output_ids)
            metrics = decoder.token_throughput()
        case "speculative":
            # default to draft model as target model if none provided
            target = target_path if target_path is not None else model_path

            decoder = SpeculativeDecoder(model_path, target, kv_cache=kv_cache, adaptive_k=adaptive_k)
            output_ids = decoder.generate_k_tokens(prompt, n, k=k)
            output_text = decoder.decode(output_ids)
            metrics = decoder.token_throughput()

        case _:
            output_text, metrics = None, None
    return output_text, metrics


PROMPT = "The first digits of pi are "
TINYLLAMA_PATH = "./models/tinyllama-1.1b"
LLAMA2_7B_PATH = "./models/llama2-7b"
N = 50

print("Starting predictions on prompt:", PROMPT)

# baseline of draft
output1, metrics1 = predict(PROMPT, TINYLLAMA_PATH, N)

# baseline of target 
output2, metrics2 = predict(PROMPT, LLAMA2_7B_PATH, N)

# basic speculative decoder
output3, metrics3 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH, kv_cache=False, adaptive_k=False)

# speculative decoder + adaptive k
output4, metrics4 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH, kv_cache=False)

# speculative decoder + kv_cache
output5, metrics5 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH, adaptive_k=False)

# full speculative decoder
output6, metrics6 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH)

print("Llama2 1.1b - Baseline")
print(output1)
print(metrics1)

print("Llama2 7b - Baseline")
print(output2)
print(metrics2)

print("Llama2 1.1b -> 7b - Speculative")
print(output3)
print(metrics3)

print("Llama2 1.1b -> 7b - Speculative with Adaptive K")
print(output4)
print(metrics4)

print("Llama2 1.1b -> 7b - Speculative with KV Cache")
print(output5)
print(metrics5)

print("Llama2 1.1b -> 7b - Speculative with Full Decoder")
print(output6)
print(metrics6)