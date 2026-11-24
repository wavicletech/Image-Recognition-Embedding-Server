"""Map the Simplismart / vLLM chat-embedding payload onto Qwen3-VL-Embedding."""

import base64
import gc
import io
import ipaddress
import logging
import os
import socket
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import torch
from PIL import Image

from app.gpu_gate import PriorityGate
from app.qwen3_vl_embedding import Qwen3VLEmbedder

logger = logging.getLogger("qwen_embedding")

MODEL_ID = os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
IMAGE_TIMEOUT_SECONDS = 10
MAX_IMAGE_REDIRECTS = 5
_VRAM_SLACK_BYTES = 150 * 1024 * 1024
_FATAL_CUDA = ("illegal memory access", "device-side assert", "unspecified launch failure")

_embedder: Qwen3VLEmbedder | None = None
_load_lock = threading.Lock()
_gate: PriorityGate | None = None
_baseline_allocated: int | None = None


class PayloadTooLarge(Exception):
    def __init__(self, message: str, tokens: int | None = None):
        super().__init__(message)
        self.tokens = tokens


def max_image_bytes() -> int:
    return int(os.environ.get("QWEN_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))


def max_input_tokens() -> int:
    return int(os.environ.get("QWEN_MAX_INPUT_TOKENS", "32768"))


def get_gate() -> PriorityGate:
    global _gate
    if _gate is None:
        _gate = PriorityGate.from_env()
    return _gate


def get_embedder() -> Qwen3VLEmbedder:
    global _embedder
    if _embedder is None:
        with _load_lock:
            if _embedder is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. This POC targets the RTX 2080.")
                _embedder = Qwen3VLEmbedder(
                    model_name_or_path=MODEL_ID,
                    dtype=torch.float16,
                    attn_implementation="sdpa",
                    low_cpu_mem_usage=True,
                )
    return _embedder


def _decode_data_url(url: str) -> Image.Image:
    _header, encoded = url.split(",", 1)
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def _blocked_ip(value: str) -> bool:
    addr = ipaddress.ip_address(value)
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _assert_public_host(hostname: str) -> None:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve image host: {hostname}") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or any(_blocked_ip(ip) for ip in addresses):
        raise ValueError("Image URL host is not a public address.")


def _read_limited(response: requests.Response) -> bytes:
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_image_bytes():
            raise ValueError(f"Image exceeds the {max_image_bytes()} byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_remote_image(url: str) -> Image.Image:
    current = url
    for _ in range(MAX_IMAGE_REDIRECTS):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Image URL must be http or https.")
        _assert_public_host(parsed.hostname)
        try:
            response = requests.get(
                current,
                stream=True,
                timeout=IMAGE_TIMEOUT_SECONDS,
                allow_redirects=False,
                headers={"User-Agent": "qwen-embedding/1.0"},
            )
        except requests.RequestException as exc:
            raise ValueError(f"Could not download image: {exc}") from exc
        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Image URL redirect was missing a location.")
                current = urljoin(current, location)
                continue
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise ValueError(f"Could not download image: {exc}") from exc
            raw = _read_limited(response)
        finally:
            response.close()
        try:
            image = Image.open(io.BytesIO(raw))
            return image.convert("RGB")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Downloaded content is not a valid image.") from exc
    raise ValueError("Too many image URL redirects.")


def _as_image(value: Any, embedder: Qwen3VLEmbedder) -> dict:
    image = value
    if isinstance(image, dict):
        image = image.get("url") or image.get("image")
    if isinstance(image, Image.Image):
        loaded = image.convert("RGB")
    elif isinstance(image, str) and image.startswith("data:"):
        loaded = _decode_data_url(image)
    elif isinstance(image, str) and image.startswith(("http://", "https://")):
        loaded = _fetch_remote_image(image)
    else:
        raise ValueError("Only http(s) image URLs and data URLs are supported.")
    return {
        "type": "image",
        "image": loaded,
        "min_pixels": embedder.min_pixels,
        "max_pixels": embedder.max_pixels,
    }


def _convert_part(part: Any, embedder: Qwen3VLEmbedder) -> dict:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        raise TypeError(f"Unsupported content part: {type(part)}")

    part_type = part.get("type")
    if part_type in (None, "text") and "text" in part and "image_url" not in part:
        return {"type": "text", "text": part.get("text") or ""}
    if part_type in ("image", "image_url") or "image_url" in part or (
        "image" in part and part_type != "video"
    ):
        image = part.get("image")
        if image is None and "image_url" in part:
            image = part["image_url"]
        return _as_image(image, embedder)
    if part_type == "video" or "video" in part:
        video = part.get("video")
        if not isinstance(video, str) or not video.startswith(("http://", "https://")):
            raise ValueError("Only http(s) video URLs are supported.")
        return {
            "type": "video",
            "video": video,
            "fps": part.get("fps", embedder.fps),
            "max_frames": part.get("max_frames", embedder.max_frames),
        }
    raise TypeError(f"Unsupported content type: {part_type}")


def messages_to_conversation(messages: list, embedder: Qwen3VLEmbedder, continue_final_message: bool) -> list:
    """Turn the chat payload into the conversation the embedder tokenizes.

    continue_final_message with an empty assistant turn is the vLLM form of
    add_generation_prompt: the prompt ends at `<|im_start|>assistant\\n`.
    """
    kept = list(messages)
    if continue_final_message and kept and kept[-1].get("role") == "assistant":
        kept = kept[:-1]

    conversation = []
    for message in kept:
        content = message.get("content")
        if content is None:
            parts = []
        elif isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = [_convert_part(part, embedder) for part in content]
        conversation.append({"role": message["role"], "content": parts})
    return conversation


def _prepare(payload: dict):
    encoding_format = payload.get("encoding_format", "float")
    if encoding_format != "float":
        raise ValueError("Only encoding_format='float' is supported.")
    if payload.get("add_special_tokens", True) is False:
        raise ValueError("add_special_tokens=false is not supported. The model expects the chat template.")
    messages = payload.get("messages")
    if not messages:
        raise ValueError("messages is required.")
    continue_final_message = bool(payload.get("continue_final_message", True))
    embedder = get_embedder()
    conversation = messages_to_conversation(messages, embedder, continue_final_message)
    processed = embedder._preprocess_inputs([conversation], add_generation_prompt=continue_final_message)
    tokens = embedder.input_token_count(processed)
    if tokens > max_input_tokens():
        _release_processed(processed)
        raise PayloadTooLarge(
            f"Input is {tokens} tokens, over the limit of {max_input_tokens()}.",
            tokens,
        )
    return processed, tokens


def _release_processed(processed: dict | None) -> None:
    if not processed:
        return
    for value in list(processed.values()):
        if torch.is_tensor(value):
            del value
    processed.clear()


def _cleanup_cuda(processed: dict | None, embeddings: torch.Tensor | None) -> None:
    _release_processed(processed)
    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = int(torch.cuda.memory_allocated())
        if _baseline_allocated is not None and allocated > _baseline_allocated + _VRAM_SLACK_BYTES:
            logger.warning(
                "VRAM did not return to baseline: allocated=%s baseline=%s",
                allocated,
                _baseline_allocated,
            )


def _fatal_cuda(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_CUDA)


def _forward(processed: dict) -> tuple[list[float], int, float]:
    embeddings = None
    started = time.perf_counter()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        embeddings = get_embedder()._embed_processed(processed, normalize=True)
        vector = embeddings[0].detach().float().cpu().tolist()
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        return vector, peak, (time.perf_counter() - started) * 1000.0
    except torch.cuda.OutOfMemoryError as exc:
        raise PayloadTooLarge("Input is too large for GPU memory.", None) from exc
    except RuntimeError as exc:
        if _fatal_cuda(exc):
            logger.exception("Fatal CUDA error; exiting so the server can restart")
            os._exit(1)
        raise
    finally:
        _cleanup_cuda(processed, embeddings)


def _run(payload: dict, use_gate: bool) -> dict:
    processed, tokens = _prepare(payload)
    source = payload.get("source")
    queue_ms = 0.0
    if use_gate:
        with get_gate().acquire(source) as waited:
            queue_ms = waited
            vector, peak, inference_ms = _forward(processed)
    else:
        vector, peak, inference_ms = _forward(processed)
    return {
        "embedding": vector,
        "dimensions": len(vector),
        "peak_vram_bytes": peak,
        "input_tokens": tokens,
        "queue_ms": queue_ms,
        "inference_ms": inference_ms,
        "model": payload.get("model") or MODEL_ID,
    }


def embed_payload(payload: dict) -> dict:
    return _run(payload, use_gate=True)


def embed_direct(payload: dict) -> dict:
    """Warmup path. Does not wait on the priority gate."""
    return _run(payload, use_gate=False)


def mark_baseline() -> int:
    global _baseline_allocated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _baseline_allocated = int(torch.cuda.memory_allocated())
    else:
        _baseline_allocated = 0
    return _baseline_allocated


def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "baseline_allocated_bytes": _baseline_allocated}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_bytes": int(total),
        "free_vram_bytes": int(free),
        "allocated_vram_bytes": int(torch.cuda.memory_allocated()),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved()),
        "baseline_allocated_bytes": _baseline_allocated,
    }
