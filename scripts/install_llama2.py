import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    api = HfApi()
    model_id = "meta-llama/Llama-2-7b-hf"
    print(api.model_info(model_id, token=token))

    output_path = Path("./models/llama2-7b")
    output_path.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=model_id,
        token=token,
        local_dir=str(output_path),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
else:
    print("download the .env file with the token or get one yourself")

