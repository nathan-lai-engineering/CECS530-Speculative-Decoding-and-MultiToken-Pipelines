from src.multi_token_pipeline import MultiTokenPipeline

def test_pipeline_basic():
    pipe = MultiTokenPipeline(
        draft_model_path="gpt2",
        target_model_path="gpt2",
        adaptive_k=False,
        buffer_capacity=2
    )

    prompt = "The first digits of pi are "

    output_ids = pipe.generate_k_tokens(prompt, n=4, k=2)

    assert output_ids is not None
    assert len(output_ids) > 0

    decoded = pipe.decode(output_ids)

    print("\nOUTPUT:", decoded)
    print("THROUGHPUT:", pipe.token_throughput())