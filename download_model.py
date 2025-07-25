import argparse
import time
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def convert_model_name_to_model_dirname(model_name):
    return f"models--{model_name.replace('/', '--')}"


def get_output_model_dir(model_name, download_dir):
    transformed_model_name = convert_model_name_to_model_dirname(model_name)
    candidate_pattern = os.path.join(download_dir, transformed_model_name, "snapshots", "*")
    candidate_folders = glob.glob(candidate_pattern)
    assert len(candidate_folders), (
        f"No valid downloaded HF model found under {candidate_pattern}. "
        f"Please verify success of download."
    )
    return candidate_folders[0]


def download_model(model_id: str, cache_dir: str):
    print(f"📥 Downloading model: {model_id}")
    print(f"📁 Cache directory: {cache_dir}")

    start_time = time.time()

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Download model weights to cache only
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

    elapsed_time = time.time() - start_time

    # Verify actual download path
    output_model_dir = get_output_model_dir(model_id, cache_dir)

    print(f"\n✅ Download completed in {elapsed_time:.2f} seconds.")
    print(f"📂 Model and tokenizer cached in: {os.path.abspath(cache_dir)}")
    print(f"📦 Model snapshot directory: {output_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF model/tokenizer to cache dir.")
    parser.add_argument("--model_id", required=True, help="Model ID from Hugging Face Hub")
    parser.add_argument("--cache_dir", required=True, help="Directory to cache the model/tokenizer")

    args = parser.parse_args()
    download_model(args.model_id, args.cache_dir)