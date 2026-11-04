"""Serve Qwen3.6-35B-A3B on Kaggle's two T4 GPUs with llama.cpp."""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import requests


WORK_DIR = Path(os.environ.get("ODYSSEUS_KAGGLE_WORK", "/kaggle/working"))
RELEASE = "b12110"
RELEASE_DIR = WORK_DIR / "llama-prebuilt"
BIN_DIR = RELEASE_DIR / f"llama-{RELEASE}"
CUDA_DIR = RELEASE_DIR / f"cudart-llama-{RELEASE}-bin-ubuntu-cuda-13.4-x64"
SERVER = BIN_DIR / "llama-server"
API_KEY_FILE = Path(os.environ.get("ODYSSEUS_API_KEY_FILE", "/tmp/odysseus-api-key"))
MODEL_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
# The 16.6 GB quant fits Kaggle's roughly 19.5 GiB working volume.
MODEL_QUANT = "UD-Q3_K_M"
MODEL_ALIAS = "Qwen/Qwen3.6-35B-A3B"


def download_and_extract(filename: str, extracted_dir: Path) -> None:
    archive = RELEASE_DIR / filename
    if not extracted_dir.is_dir():
        if not archive.is_file():
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{filename}"
            print(f"Downloading official llama.cpp release asset: {filename}", flush=True)
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with archive.open("wb") as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(RELEASE_DIR, filter="data")


def main() -> None:
    if not API_KEY_FILE.is_file() or not API_KEY_FILE.read_text().strip():
        raise RuntimeError(f"Create a private API key file at {API_KEY_FILE} before starting the server.")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    download_and_extract(f"llama-{RELEASE}-bin-ubuntu-cuda-13.4-x64.tar.gz", BIN_DIR)
    download_and_extract(f"cudart-llama-{RELEASE}-bin-ubuntu-cuda-13.4-x64.tar.gz", CUDA_DIR)
    if not SERVER.is_file():
        raise RuntimeError(f"llama-server was not found at {SERVER} after extracting the release.")

    os.environ.setdefault("HF_HOME", str(WORK_DIR / ".cache" / "huggingface"))
    library_dirs = f"{BIN_DIR}:{CUDA_DIR}"
    os.environ["LD_LIBRARY_PATH"] = library_dirs + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    command = [
        str(SERVER),
        "--hf-repo", f"{MODEL_REPO}:{MODEL_QUANT}",
        "--alias", MODEL_ALIAS,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--ctx-size", "8192",
        "--n-gpu-layers", "99",
        "--split-mode", "layer",
        "--tensor-split", "1,1",
        "--parallel", "1",
        "--api-key-file", str(API_KEY_FILE),
        "--no-webui",
        "--no-mmproj",
    ]
    print("Starting the authenticated Qwen3.6-35B-A3B server on both T4 GPUs.", flush=True)
    subprocess.run(command, cwd=BIN_DIR, check=True)


if __name__ == "__main__":
    main()
