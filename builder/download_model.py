import argparse
import os

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model into the image.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--target-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN") or None

    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.target_dir,
        local_dir_use_symlinks=False,
        token=token,
    )


if __name__ == "__main__":
    main()
