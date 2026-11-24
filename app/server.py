"""FastAPI server for Qwen3-VL-Embedding-2B.

POST /v1/embeddings and POST /embeddings accept the Simplismart chat payload
and return an OpenAI-style embedding object.
"""

import base64
import io
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from PIL import Image

from app.gpu_gate import QueueFull, WaitTimeout
from app.service import (
    MODEL_ID,
    PayloadTooLarge,
    embed_direct,
    embed_payload,
    get_embedder,
    get_gate,
    gpu_stats,
    mark_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("qwen_embedding")

_EMBED_PATHS = {"/embedding", "/embeddings", "/v1/embeddings"}
_TEXT_PREVIEW = 80
_URL_PREVIEW = 240
_ERROR_PREVIEW = 400


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _image_ref(value) -> str:
    if isinstance(value, dict):
        value = value.get("url") or value.get("image") or ""
    if not isinstance(value, str) or not value:
        return "image:?"
    if value.startswith("data:"):
        header, _, payload = value.partition(",")
        mime = header[5:].split(";", 1)[0] or "image"
        return f"image:data:{mime}:{len(payload)}b64"
    return "image:" + _clip(value, _URL_PREVIEW)


def _part_label(part) -> str:
    if isinstance(part, str):
        return f'text:{len(part)} "{_clip(part, _TEXT_PREVIEW)}"'
    if not isinstance(part, dict):
        return f"?:{type(part).__name__}"
    part_type = part.get("type")
    if part_type in (None, "text") and "text" in part and "image_url" not in part:
        text = part.get("text") or ""
        return f'text:{len(str(text))} "{_clip(text, _TEXT_PREVIEW)}"'
    if part_type in ("image", "image_url") or "image_url" in part or (
        "image" in part and part_type != "video"
    ):
        image = part.get("image")
        if image is None and "image_url" in part:
            image = part["image_url"]
        return _image_ref(image)
    if part_type == "video" or "video" in part:
        video = part.get("video")
        if isinstance(video, str) and video:
            return "video:" + _clip(video, _URL_PREVIEW)
        return "video:?"
    return f"type:{part_type}"


def _message_label(message) -> str:
    if not isinstance(message, dict):
        return f"?:{type(message).__name__}"
    role = message.get("role") or "?"
    content = message.get("content")
    if content is None:
        return f"{role}:empty"
    if isinstance(content, str):
        return f'{role}:text:{len(content)} "{_clip(content, _TEXT_PREVIEW)}"'
    if isinstance(content, list):
        return f"{role}:[" + ", ".join(_part_label(part) for part in content) + "]"
    return f"{role}:{type(content).__name__}"


def _summarize_payload(raw: bytes) -> str:
    if not raw:
        return "body=empty"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return f"body=invalid_json bytes={len(raw)} preview={_clip(raw.decode('utf-8', 'replace'), 120)!r}"
    if not isinstance(payload, dict):
        return f"body={type(payload).__name__} bytes={len(raw)}"
    messages = payload.get("messages")
    if isinstance(messages, list):
        rendered = " | ".join(_message_label(message) for message in messages)
        messages_field = f"messages={len(messages)} [{rendered}]"
    elif messages is None:
        messages_field = "messages=missing"
    else:
        messages_field = f"messages={type(messages).__name__}"
    return " ".join(
        [
            f"model={payload.get('model')!r}",
            f"source={payload.get('source')!r}",
            f"encoding_format={payload.get('encoding_format', 'float')!r}",
            f"continue_final_message={payload.get('continue_final_message', True)!r}",
            f"add_special_tokens={payload.get('add_special_tokens', True)!r}",
            messages_field,
        ]
    )


def _error_detail(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _clip(text, _ERROR_PREVIEW)
    if isinstance(payload, dict) and "detail" in payload:
        detail = payload["detail"]
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False)
        return _clip(detail, _ERROR_PREVIEW)
    return _clip(text, _ERROR_PREVIEW)


class EmbedAccessLog:
    """Log embedding calls with the inputs that explain a success or failure."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in _EMBED_PATHS:
            await self.app(scope, receive, send)
            return

        parts = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        raw = b"".join(parts)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}

        status_code = 500
        response_headers: list[tuple[bytes, bytes]] = []
        error_chunks: list[bytes] = []

        async def send_wrapper(message):
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = list(message.get("headers") or [])
            elif message["type"] == "http.response.body" and status_code >= 400:
                error_chunks.append(message.get("body", b""))
            await send(message)

        client = scope["client"][0] if scope.get("client") else "-"
        path = scope.get("path", "")
        summary = _summarize_payload(raw)
        started = time.perf_counter()
        try:
            await self.app(scope, replay, send_wrapper)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "embed fail status=500 client=%s path=%s elapsed_ms=%.1f error=unhandled %s",
                client,
                path,
                elapsed_ms,
                summary,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        if status_code >= 400:
            logger.warning(
                "embed fail status=%s client=%s path=%s elapsed_ms=%.1f error=%r %s",
                status_code,
                client,
                path,
                elapsed_ms,
                _error_detail(b"".join(error_chunks)),
                summary,
            )
            return

        headers = {key.decode().lower(): value.decode("utf-8", "replace") for key, value in response_headers}
        logger.info(
            "embed ok status=%s client=%s path=%s elapsed_ms=%.1f tokens=%s queue_ms=%s inference_ms=%s peak_vram_bytes=%s %s",
            status_code,
            client,
            path,
            elapsed_ms,
            headers.get("x-input-tokens", "-"),
            headers.get("x-queue-ms", "-"),
            headers.get("x-inference-ms", "-"),
            headers.get("x-peak-vram-bytes", "-"),
            summary,
        )

_WARM_TEXT = {
    "model": MODEL_ID,
    "encoding_format": "float",
    "continue_final_message": True,
    "add_special_tokens": True,
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": [{"type": "text", "text": "warmup"}]},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ],
}


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("QWEN_EMBED_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="QWEN_EMBED_API_KEY is not configured.")
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("QWEN_EMBED_API_KEY"):
        raise RuntimeError("QWEN_EMBED_API_KEY is not configured.")
    app.state.ready = False
    started = time.perf_counter()
    get_embedder()
    embed_direct(_WARM_TEXT)
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    embed_direct(
        {
            "model": MODEL_ID,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": ""},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ],
        }
    )
    mark_baseline()
    app.state.load_seconds = time.perf_counter() - started
    app.state.ready = True
    yield


app = FastAPI(title="Qwen3-VL-Embedding", version="0.2.0", lifespan=lifespan)
app.add_middleware(EmbedAccessLog)


def _status() -> dict:
    return {
        "model": MODEL_ID,
        "dtype": "float16",
        "ready": bool(getattr(app.state, "ready", False)),
        "load_seconds": getattr(app.state, "load_seconds", None),
        "queue": get_gate().snapshot(),
        "gpu": gpu_stats(),
        "max_input_tokens": int(os.environ.get("QWEN_MAX_INPUT_TOKENS", "32768")),
    }


@app.get("/health")
def health() -> dict:
    body = _status()
    body["status"] = "ok"
    return body


@app.get("/ready")
def ready(response: Response) -> dict:
    body = _status()
    if not body["ready"]:
        response.status_code = 503
        body["status"] = "starting"
        return body
    body["status"] = "ready"
    return body


def _embed(body: dict, response: Response) -> dict:
    try:
        result = embed_payload(body)
    except PayloadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except QueueFull as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": str(exc.retry_after)}
        ) from exc
    except WaitTimeout as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": str(exc.retry_after)}
        ) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response.headers["X-Queue-Ms"] = f"{result['queue_ms']:.2f}"
    response.headers["X-Inference-Ms"] = f"{result['inference_ms']:.2f}"
    response.headers["X-Peak-Vram-Bytes"] = str(result["peak_vram_bytes"])
    response.headers["X-Input-Tokens"] = str(result["input_tokens"])
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": result["embedding"],
            }
        ],
        "model": result["model"],
        "usage": {
            "prompt_tokens": result["input_tokens"],
            "total_tokens": result["input_tokens"],
        },
    }


@app.post("/embedding", dependencies=[Depends(require_api_key)])
@app.post("/embeddings", dependencies=[Depends(require_api_key)])
@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
def embeddings(body: dict, response: Response) -> dict:
    return _embed(body, response)


def main() -> None:
    _load_dotenv()
    if not os.environ.get("QWEN_EMBED_API_KEY"):
        raise SystemExit("QWEN_EMBED_API_KEY is not configured.")
    host = os.environ.get("QWEN_EMBED_HOST", "0.0.0.0")
    port = int(os.environ.get("QWEN_EMBED_PORT", "8090"))
    uvicorn.run("app.server:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
