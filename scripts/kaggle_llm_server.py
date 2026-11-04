"""OpenAI-compatible chat server for the Odysseus Kaggle GPU notebook.

The notebook starts this process on loopback and exposes it through a
Cloudflare Quick Tunnel. Keep the Kaggle notebook private: its output contains
the bearer key used by Odysseus.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Iterator

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_ID = os.environ.get("ODYSSEUS_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
API_KEY = os.environ.get("ODYSSEUS_API_KEY", "")
HOST = os.environ.get("ODYSSEUS_LLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("ODYSSEUS_LLM_PORT", "8000"))

if not API_KEY:
    raise RuntimeError("ODYSSEUS_API_KEY must be set")
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is available. Enable a Kaggle GPU accelerator and restart the session.")

print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0)}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded; starting the API server.", flush=True)

app = FastAPI(title="Odysseus Kaggle LLM", docs_url=None, redoc_url=None)
generation_lock = threading.Lock()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    stream: bool = False


def _check_key(authorization: str | None) -> None:
    expected = f"Bearer {API_KEY}"
    if not authorization or not __import__("hmac").compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _generate(messages: list[ChatMessage], request: ChatRequest) -> tuple[TextIteratorStreamer, threading.Thread]:
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The GPU is handling another request; retry shortly.")
    try:
        conversation = [{"role": item.role, "content": item.content} for item in messages]
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "top_p": request.top_p,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            generation["temperature"] = request.temperature
        thread = threading.Thread(target=model.generate, kwargs=generation, daemon=True)
        thread.start()
        return streamer, thread
    except Exception:
        generation_lock.release()
        raise


def _release_when_done(thread: threading.Thread) -> None:
    thread.join()
    generation_lock.release()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready", "model": MODEL_ID}


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_key(authorization)
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "odysseus-kaggle"}]}


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    _check_key(authorization)
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    streamer, thread = _generate(request.messages, request)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        def event_stream() -> Iterator[str]:
            try:
                yield "data: " + __import__("json").dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }) + "\n\n"
                for part in streamer:
                    yield "data: " + __import__("json").dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                    }) + "\n\n"
                thread.join()
                yield "data: " + __import__("json").dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if thread.is_alive():
                    thread.join()
                if generation_lock.locked():
                    generation_lock.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        content = "".join(streamer)
        thread.join()
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    finally:
        if thread.is_alive():
            thread.join()
        if generation_lock.locked():
            generation_lock.release()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
