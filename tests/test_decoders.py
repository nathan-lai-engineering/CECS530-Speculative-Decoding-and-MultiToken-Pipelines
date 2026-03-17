import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","src"))
from baseline_decoder import BaselineDecoder
from speculative_decoder import SpeculativeDecoder


# predict using either baseline or speculative method on given model
def predict(prompt, model_path, n, type="baseline", target_path=None, k=None):
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

            decoder = SpeculativeDecoder(model_path, target)
            output_ids = decoder.generate_k_tokens(prompt, n, k=k)
            output_text = decoder.decode(output_ids)
            metrics = decoder.token_throughput()

        case _:
            output_text, metrics = None, None
    return output_text, metrics

prompt = "The definition of Speculative decoding in LLM models is"

print("Starting predictions on prompt:", prompt)

output1, metrics1 = predict(prompt, "./models/tinyllama-1.1b", 50)
output2, metrics2 = predict(prompt, "./models/llama2-7b", 50)
output3, metrics3 = predict(prompt, "./models/tinyllama-1.1b", 50, type="speculative", target_path="./models/llama2-7b")


print("Llama2 1.1b - Baseline")
try:
    print(output1)
    print(metrics1)
except NameError:
    print("skipping Llama2 1.1b - Baseline")

print("Llama2 7b - Baseline")
try:
    print(output2)
    print(metrics2)
except NameError:
    print("skipping Llama2 7b - Baseline")


print("Llama2 1.1b -> 7b - Speculative")
try:
    print(output3)
    print(metrics3)
except NameError:
    print("skipping Llama2 7b - Baseline")