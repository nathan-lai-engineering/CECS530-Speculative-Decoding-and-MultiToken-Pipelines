import sys
import os
import csv
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Runs the experiment and generates a csv file")
parser.add_argument("--n", help="the maximum output sequence length", default=100, type=int)
parser.add_argument("--prompt", help="the prompt to input into the model", default="The first digits of pi are ")
parser.add_argument("--loops", help="how many times to repeat the experiemnt", default=1, type=int)
parser.add_argument("--increment", help="how many times to loop the loops while incrementing n by double n", default=1, type=int)
args = parser.parse_args()

N = args.n
PROMPT = args.prompt
LOOPS = args.loops
INCREMENT_LOOPS = args.increment


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".","src"))
from baseline_decoder import BaselineDecoder
from speculative_decoder import SpeculativeDecoder

# predict using either baseline or speculative method on given model
def predict(prompt, model_path, n, type="baseline", target_path=None, k=None, kv_cache=True, adaptive_k=True):
    if k is None:
        k = max(2, int(0.10 * n))
        # k = 4
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

def append_csv_data(csv_array, scenario, output, metrics):
    csv_array.append([scenario, output] + list(metrics.values()))
    return csv_array

TINYLLAMA_PATH = "./models/tinyllama-1.1b"
LLAMA2_7B_PATH = "./models/llama2-7b"

print("Starting predictions on prompt:", PROMPT)

for i in range(INCREMENT_LOOPS):

    for j in range(LOOPS):

        print(f"Beginning loop {j + 1} of {LOOPS} for N = {N} ")
        csv_data = [[
            "Scenario",
            "Output",
            "tokens_per_second",
            "total_tokens",
            "total_time",
            "mean_time_per_token",
            "max_time_per_token",
            "min_time_per_token",
            "accepted_tokens",
            "total_draft_tokens",
            "verification_rounds",
            "total_draft_time",
            "total_target_time",
            "peak_memory_MB",
            "model_memory_MB",
            "memory_bandwidth_GB_per_s",
            "peak_memory_bandwidth_GB_per_s",
        ]]

        # baseline of draft
        scenario1 = "Llama2 1.1b - Baseline"
        output1, metrics1 = predict(PROMPT, TINYLLAMA_PATH, N)
        append_csv_data(csv_data, scenario1, output1, metrics1)

        # baseline of target 
        scenario2 = "Llama2 7b - Baseline"
        output2, metrics2 = predict(PROMPT, LLAMA2_7B_PATH, N)
        append_csv_data(csv_data, scenario2, output2, metrics2)

        # basic speculative decoder
        scenario3 = "Llama2 1.1b -> 7b - Speculative"
        output3, metrics3 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH, kv_cache=False, adaptive_k=False)
        append_csv_data(csv_data, scenario3, output3, metrics3)

        # speculative decoder + adaptive k
        scenario4 = "Llama2 1.1b -> 7b - Speculative with Adaptive K"
        output4, metrics4 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH, kv_cache=False)
        append_csv_data(csv_data, scenario4, output4, metrics4)

        # speculative decoder + kv_cache
        scenario5 = "Llama2 1.1b -> 7b - Speculative with KV Cache"
        output5, metrics5 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH, adaptive_k=False)
        append_csv_data(csv_data, scenario5, output5, metrics5)

        # full speculative decoder
        scenario6 = "Llama2 1.1b -> 7b - Speculative with Full Decoder"
        output6, metrics6 = predict(PROMPT, TINYLLAMA_PATH, N, type="speculative", target_path=LLAMA2_7B_PATH)
        append_csv_data(csv_data, scenario6, output6, metrics6)

        # save the date to a csv file, we'll save all individual runs
        output_filename = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_N{N}.csv'
        output_path = os.path.join(os.path.dirname(__file__), "results", output_filename)
        with open(output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(csv_data)

    N *= 2
