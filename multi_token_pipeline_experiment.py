import sys
import os
import csv
import torch
from datetime import datetime
from src.multi_token_pipeline import MultiTokenPipeline

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This experiment requires a GPU.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".", "src"))
from src.baseline_decoder import BaselineDecoder
from src.speculative_decoder import SpeculativeDecoder


# predict using either baseline or speculative method on given model
def predict(prompt_text, model_path, num_tokens, decoder_type="baseline", target_model_path=None, draft_k=None, use_kv_cache=True, use_adaptive_k=True):
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
    rollback_rate = metrics_dict.get("rollback_count", 0) / max(1, metrics_dict["verification_rounds"])

    csv_rows.append(
        [scenario_name, generated_output] +
        list(metrics_dict.values()) +
        [acceptance_rate, rollback_rate]
    )
    return csv_rows


PROMPT_TEXT = "The first digits of pi are "
DRAFT_MODEL_PATH = "./models/tinyllama-1.1b"
TARGET_MODEL_PATH = "./models/llama2-7b"
N = 100

print("Starting predictions on prompt:", PROMPT_TEXT)

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
    "acceptance_rate",
    "rollback_rate"
]]

# multi-token pipeline
scenario_name_7 = "Llama2 1.1b -> 7b - Multi-Token Pipeline"
generated_output_7, metrics_7 = predict(
    PROMPT_TEXT,
    DRAFT_MODEL_PATH,
    N,
    decoder_type="pipeline",
    target_model_path=TARGET_MODEL_PATH,
    use_adaptive_k=False
)
append_csv_data(csv_rows, scenario_name_7, generated_output_7, metrics_7)

scenario_name_8 = "Llama2 1.1b -> 7b - Multi-Token Pipeline with Adaptive K"
generated_output_8, metrics_8 = predict(
    PROMPT_TEXT,
    DRAFT_MODEL_PATH,
    N,
    decoder_type="pipeline",
    target_model_path=TARGET_MODEL_PATH,
    use_adaptive_k=True
)
append_csv_data(csv_rows, scenario_name_8, generated_output_8, metrics_8)

# save the data to a csv file
output_filename = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
output_file_path = os.path.join(os.path.dirname(__file__), "results", output_filename)

with open(output_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(csv_rows)

print(scenario_name_7)
print(generated_output_7)
print(metrics_7)

print(scenario_name_8)
print(generated_output_8)
print(metrics_8)