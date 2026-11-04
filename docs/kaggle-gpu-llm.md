# Use a Kaggle GPU model with Odysseus

This setup runs Qwen3.6-35B-A3B on Kaggle's two T4 GPUs and gives Odysseus a
temporary OpenAI-compatible endpoint through a Cloudflare Quick Tunnel. Kaggle
sessions and Quick Tunnel URLs are temporary: keep the notebook running, and
repeat the connection steps whenever Kaggle restarts it.

## Start the Kaggle server

1. Open the private Kaggle notebook **Odysseus GPU LLM + Cloudflare Tunnel**.
2. In **Settings → Accelerator**, choose **GPU T4 x2**, and make sure
   **Internet** is enabled. Save the settings and restart the session if Kaggle
   asks.
3. Run the notebook's setup cell and wait for its `READY` message. It prints a
   base URL ending in `/v1` and a bearer API key. Keep this notebook private;
   its output contains the key.

The setup cell downloads `scripts/kaggle_llm_server.py` from this repository's
`dev` branch, fetches the official llama.cpp CUDA 12.8 release binaries, and
downloads the Qwen GGUF quantization. It uses the `UD-Q3_K_M` quant (about
16.6 GB), which fits Kaggle's notebook working volume. The notebook limits
context to 8,192 tokens to leave room for the KV cache and server overhead.
CUDA 12.8 stays within the PTX support provided by Kaggle's T4 driver; the
server also disables CUDA PDL for T4 compatibility.

## Add the endpoint in Odysseus

In Odysseus, open **Settings → Models → Add endpoint** and enter:

- Name: `Kaggle GPU Qwen3.6`
- Base URL: the notebook's printed URL ending in `/v1`
- API key: the notebook's printed bearer key
- Endpoint type: `OpenAI-compatible`

Save the endpoint, refresh its model list, and choose
`Qwen/Qwen3.6-35B-A3B` in the chat model picker. Prompts sent to that model
will travel from Odysseus to Kaggle through the tunnel and be processed by the
GPU.

The bearer key protects the model routes if the tunnel URL becomes known. The
Kaggle notebook is intentionally private, the model server binds only to
loopback, and the key is stored encrypted in Odysseus' local endpoint record.
Quick Tunnel addresses are not stable or suitable for always-on deployments.
