import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def download_hf_model(model_id, output):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        #if os.path.exists(output):
        #    shutil.rmtree(output) # for clean install

        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            token=token,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            cache_dir=str(output_path),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
    else:
        print("download the .env file with the token or get one yourself")

#download_hf_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./models/tinyllama-1.1b") #2.2 GB
download_hf_model("meta-llama/Llama-2-7b-hf", "./models/llama2-7b") #13 GB
#download_hf_model("meta-llama/Llama-2-13b-hf", "./models/llama2-13b") #52 GB
