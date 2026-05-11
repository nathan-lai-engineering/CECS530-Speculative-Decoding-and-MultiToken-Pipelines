import sys
import os
import csv
import argparse
import torch
from datetime import datetime
from src.multi_token_pipeline import MultiTokenPipeline

parser = argparse.ArgumentParser(description="Runs the multi-token pipeline experiment and generates a csv file")
parser.add_argument("--n", help="the maximum output sequence length", default=50, type=int)
parser.add_argument("--prompt", help="the prompt to input into the model", default="The first digits of pi are ")
parser.add_argument("--loops", help="how many times to repeat the experiment", default=5, type=int)
parser.add_argument("--increment", help="how many times to loop while incrementing n by double n", default=1, type=int)
args = parser.parse_args()

N = args.n
PROMPT = args.prompt
LOOPS = args.loops
INCREMENT_LOOPS = args.increment

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This experiment requires a GPU.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".", "src"))

DRAFT_MODEL_PATH = "./models/tinyllama-1.1b"
TARGET_MODEL_PATH = "./models/llama2-7b"


def predict(prompt_text, model_path, num_tokens, decoder_type="pipeline", target_model_path=None, draft_k=None, use_adaptive_k=True):
    if draft_k is None:
        draft_k = max(2, int(0.10 * num_tokens))

    match decoder_type:
        case "pipeline":
            resolved_target_path = target_model_path if target_model_path is not None else model_path

            pipeline_decoder = MultiTokenPipeline(
                draft_model_path=model_path,
                target_model_path=resolved_target_path,
                adaptive_k=use_adaptive_k,
                buffer_capacity=2
            )
            generated_token_ids = pipeline_decoder.generate_k_tokens(prompt_text, n=num_tokens, k=draft_k)
            generated_text = pipeline_decoder.decode(generated_token_ids)
            performance_metrics = pipeline_decoder.token_throughput()

        case _:
            generated_text, performance_metrics = None, None

    return generated_text, performance_metrics


def append_csv_data(csv_rows, scenario_name, generated_output, metrics_dict):
    acceptance_rate = metrics_dict["accepted_tokens"] / max(1, metrics_dict["total_draft_tokens"])
    rollback_rate = metrics_dict.get("rollback_events", 0) / max(1, metrics_dict["verification_rounds"])

    csv_rows.append(
        [scenario_name, generated_output] +
        list(metrics_dict.values()) +
        [acceptance_rate, rollback_rate]
    )
    return csv_rows


print("Starting predictions on prompt:", PROMPT)

for i in range(INCREMENT_LOOPS):

    for j in range(LOOPS):

        print(f"Beginning loop {j + 1} of {LOOPS} for N = {N}")

        csv_rows = [[
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
            "pipeline_total_time",
            "draft_stage_busy_time",
            "verify_stage_busy_time",
            "pipeline_bubbles",
            "rollback_events",
            "flushed_batches",
            "max_buffer_occupancy",
            "buffer_capacity",
            "batches_drafted",
            "batches_verified",
            "peak_memory_MB",
            "model_memory_MB",
            "memory_bandwidth_GB_per_s",
            "peak_memory_bandwidth_GB_per_s",
            "cpu_ram_delta_MB",
            "gpu_utilization_pct",
            "acceptance_rate",
            "rollback_rate",
        ]]

        scenario_name_1 = "Llama2 1.1b -> 7b - Multi-Token Pipeline"
        generated_output_1, metrics_1 = predict(
            PROMPT,
            DRAFT_MODEL_PATH,
            N,
            target_model_path=TARGET_MODEL_PATH,
            use_adaptive_k=False
        )
        append_csv_data(csv_rows, scenario_name_1, generated_output_1, metrics_1)

        scenario_name_2 = "Llama2 1.1b -> 7b - Multi-Token Pipeline with Adaptive K"
        generated_output_2, metrics_2 = predict(
            PROMPT,
            DRAFT_MODEL_PATH,
            N,
            target_model_path=TARGET_MODEL_PATH,
            use_adaptive_k=True
        )
        append_csv_data(csv_rows, scenario_name_2, generated_output_2, metrics_2)

        output_filename = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_N{N}.csv'
        output_file_path = os.path.join(os.path.dirname(__file__), "results", output_filename)

        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(csv_rows)

    N *= 2
