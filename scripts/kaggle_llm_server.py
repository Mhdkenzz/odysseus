"""Build llama.cpp with CUDA and serve Qwen3.6-35B-A3B on Kaggle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


WORK_DIR = Path(os.environ.get("ODYSSEUS_KAGGLE_WORK", "/kaggle/working"))
LLAMA_DIR = WORK_DIR / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build"
SERVER = BUILD_DIR / "bin" / "llama-server"
API_KEY_FILE = Path(os.environ.get("ODYSSEUS_API_KEY_FILE", "/tmp/odysseus-api-key"))
MODEL_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_QUANT = "UD-Q4_K_M"
MODEL_ALIAS = "Qwen/Qwen3.6-35B-A3B"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not shutil.which("nvcc"):
        raise RuntimeError("CUDA compiler nvcc is missing. Select Kaggle GPU T4 x2 and restart the session.")
    if not API_KEY_FILE.is_file() or not API_KEY_FILE.read_text().strip():
        raise RuntimeError(f"Create a private API key file at {API_KEY_FILE} before starting the server.")

    if not LLAMA_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_DIR)])
    if not SERVER.exists():
        run([
            "cmake", "-S", str(LLAMA_DIR), "-B", str(BUILD_DIR),
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "llama-server", "-j", "4"])

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
    os.environ.setdefault("HF_HOME", str(WORK_DIR / ".cache" / "huggingface"))
    run(command, cwd=LLAMA_DIR)


if __name__ == "__main__":
    main()
