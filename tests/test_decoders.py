import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","src"))
from baseline_decoder import BaselineDecoder
from speculative_decoder import SpeculativeDecoder


def predict(prompt, model_path, k, type="baseline", target_path=None):
    match type:
        case "baseline":
            decoder = BaselineDecoder(model_path)
            input_ids = decoder.encode(prompt)
            output_ids = decoder.generate_k_tokens(input_ids, k=k)
            output_text = decoder.decode(output_ids)
            metrics = decoder.token_throughput()
        case "speculative":
            # default to draft model as target model if none provided
            target = target_path if target_path is not None else model_path

            decoder = SpeculativeDecoder(model_path, target)
            input_ids = decoder.encode(prompt)
            draft_ids, draft_probs = decoder.generate_k_draft_tokens(prompt, k=k)
            output_ids = decoder.parallel_verification(draft_ids, draft_probs, k=k)
            output_text = decoder.decode(output_ids)
            metrics = decoder.token_throughput()

        case _:
            output_text, metrics = None, None
    return output_text, metrics




prompt = "Speculative decoding is"

output1, metrics1 = predict(prompt, "./models/tinyllama-1.1b", 20)
output2, metrics2 = predict(prompt, "./models/llama2-7b", 20)
output3, metrics3 = predict(prompt, "./models/tinyllama-1.1b", 20, type="speculative", target_path="./models/llama2-7b")


print("Llama2 7b - Baseline")
print(output1)
print(metrics1)

print("Llama2 1.1b - Baseline")
print(output2)
print(metrics2)


print("Llama2 1.1b -> 7b - Speculative")
print(output3)
print(metrics3)