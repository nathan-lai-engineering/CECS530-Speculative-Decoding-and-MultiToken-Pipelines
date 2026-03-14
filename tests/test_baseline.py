import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","src"))
from baseline_decoder import BaselineDecoder

def predict(prompt, model_path, k):
    decoder = BaselineDecoder(model_path)
    input_ids = decoder.encode(prompt)
    output_ids = decoder.generate_k_tokens(input_ids, k=k)
    output_text = decoder.decode(output_ids)
    metrics = decoder.token_throughput()
    return output_text, metrics

prompt = "Speculative decoding is"

output1, metrics1 = predict(prompt, "./models/tinyllama-1.1b", 20)
output2, metrics2 = predict(prompt, "./models/llama2-7b", 20)

print("Llama2 7b parameters")
print(output1)
print(metrics1)

print("Llama2 1.1b parameters")
print(output2)
print(metrics2)
