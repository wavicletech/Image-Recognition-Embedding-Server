"""Capacity benchmark for the self-hosted Qwen embedding server.

Writes bench/results/<timestamp>.json and prints the token budget and the
KBM concurrency that keeps Bot Builder responsive.
"""

import base64
import io
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("QWEN_BENCH_URL", "http://127.0.0.1:8090")
KEY = os.environ["QWEN_EMBED_API_KEY"]
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
GIB = 1024 ** 3
PEAK_BUDGET = 7.0 * GIB


def post(content, source, timeout=180):
    payload = {
        "model": MODEL,
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
        "source": source,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
    }
    request = urllib.request.Request(
        BASE + "/v1/embeddings",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
            elapsed = (time.perf_counter() - started) * 1000.0
            return {
                "status": response.status,
                "client_ms": round(elapsed, 1),
                "inference_ms": float(response.headers.get("X-Inference-Ms", "0")),
                "queue_ms": float(response.headers.get("X-Queue-Ms", "0")),
                "peak_vram_bytes": int(response.headers.get("X-Peak-Vram-Bytes", "0")),
                "input_tokens": int(response.headers.get("X-Input-Tokens", "0")),
                "dim": len(body["data"][0]["embedding"]),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "client_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "detail": exc.read().decode()[:240],
            "peak_vram_bytes": 0,
            "input_tokens": 0,
        }


def health():
    with urllib.request.urlopen(BASE + "/health", timeout=5) as response:
        return json.loads(response.read().decode())


def text_of(target_tokens: int) -> str:
    piece = "The product_type is 'silk saree'. The color is 'maroon'. "
    text = piece
    while len(TOKENIZER(text, add_special_tokens=False)["input_ids"]) < target_tokens:
        text += piece
    ids = TOKENIZER(text, add_special_tokens=False)["input_ids"]
    return TOKENIZER.decode(ids[:target_tokens])


def jpeg(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), (230, 226, 218))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, width - 8, height - 8], outline=(20, 20, 20), width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_content(width: int, height: int, text: str = ""):
    return [
        {"type": "image_url", "image_url": {"url": jpeg(width, height)}},
        {"type": "text", "text": text},
    ]


def record(kind, label, content):
    before = health()["gpu"]["allocated_vram_bytes"]
    result = post(content, "kbm")
    after = health()["gpu"]["allocated_vram_bytes"]
    result.update({
        "kind": kind,
        "label": label,
        "allocated_before": before,
        "allocated_after": after,
        "returned_to_baseline": abs(after - before) <= 150 * 1024 * 1024,
    })
    print(
        f"{kind:12} {label:22} status={result['status']} tokens={result.get('input_tokens')} "
        f"peak_mib={result['peak_vram_bytes'] / 1048576:.0f} ms={result.get('inference_ms', result['client_ms'])} "
        f"released={result['returned_to_baseline']}"
    )
    return result


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def variant(image_url: str, text: str):
    image = post(image_content_from_url(image_url), "kbm")
    words = post([{"type": "text", "text": text}], "kbm")
    time.sleep(float(os.environ.get("QWEN_BENCH_AZURE_S", "2")))
    return image, words


def image_content_from_url(data_url: str, text: str = ""):
    return [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": text},
    ]


def concurrency_sweep(image_url: str, attribute_text: str):
    rows = []
    for workers in (1, 2, 3, 4, 5, 6):
        variants = 8
        started = time.perf_counter()
        errors = 0
        latencies = []

        def one(_index):
            image, words = variant(image_url, attribute_text)
            return image, words

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(one, range(variants)))
        wall = time.perf_counter() - started
        for image, words in pairs:
            for item in (image, words):
                latencies.append(item["client_ms"])
                if item["status"] != 200:
                    errors += 1
        gpu = health()["gpu"]
        row = {
            "concurrency": workers,
            "variants": variants,
            "wall_s": round(wall, 2),
            "variants_per_s": round(variants / wall, 3),
            "embed_p50_ms": percentile(latencies, 0.50),
            "embed_p95_ms": percentile(latencies, 0.95),
            "errors": errors,
            "allocated_vram_bytes": gpu["allocated_vram_bytes"],
        }
        rows.append(row)
        print(
            f"concurrency={workers} variants/s={row['variants_per_s']} "
            f"p95_ms={row['embed_p95_ms']} errors={errors}"
        )
    return rows


def priority_check(image_url: str, attribute_text: str):
    stop_at = time.perf_counter() + 16
    bulk_errors = 0

    def bulk(_index):
        nonlocal bulk_errors
        while time.perf_counter() < stop_at:
            image, words = variant(image_url, attribute_text)
            if image["status"] != 200 or words["status"] != 200:
                bulk_errors += 1
        return True

    user_latencies = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(bulk, i) for i in range(5)]
        while time.perf_counter() < stop_at:
            result = post([{"type": "text", "text": "The product_type is 'dress'. The color is 'red'. "}], "bot_builder")
            user_latencies.append(result["client_ms"])
            time.sleep(2)
        for future in futures:
            future.result()
    return {
        "user_requests": len(user_latencies),
        "user_p50_ms": percentile(user_latencies, 0.50),
        "user_p95_ms": percentile(user_latencies, 0.95),
        "user_max_ms": max(user_latencies) if user_latencies else None,
        "bulk_errors": bulk_errors,
    }


def soak():
    failures = []
    baseline = health()["gpu"]["baseline_allocated_bytes"]
    small = jpeg(512, 512)
    for index in range(1000):
        if index % 5 == 0:
            content = image_content_from_url(small)
        else:
            content = [{"type": "text", "text": "The product_type is 'dress'. The color is 'red'. "}]
        result = post(content, "kbm" if index % 2 else "bot_builder")
        allocated = health()["gpu"]["allocated_vram_bytes"]
        drifted = baseline is not None and abs(allocated - baseline) > 150 * 1024 * 1024
        if result["status"] != 200 or drifted:
            failures.append({"index": index, "status": result["status"], "allocated": allocated})
            if len(failures) > 10:
                break
    ending = health()["gpu"]["allocated_vram_bytes"]
    print(f"soak failures={len(failures)} end_allocated={ending} baseline={baseline}")
    return {"failures": failures, "end_allocated": ending, "baseline": baseline}


def recommend(rows, priority):
    clean = [row for row in rows if row["errors"] == 0]
    if not clean:
        return 1
    best = max(row["variants_per_s"] for row in clean)
    user_p95 = priority.get("user_p95_ms") or 0
    eligible = [
        row["concurrency"]
        for row in clean
        if row["variants_per_s"] >= 0.9 * best and (user_p95 <= 2500 or row["concurrency"] == 1)
    ]
    return min(eligible) if eligible else 1


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "single": [], "concurrency": []}
    print("--- single requests ---")
    for tokens in (32, 128, 512, 1024, 2048, 4096, 6144, 8192):
        report["single"].append(record("text", f"{tokens}", [{"type": "text", "text": text_of(tokens)}]))
    sizes = {
        "0.25mp": (500, 500),
        "0.50mp": (707, 707),
        "1.00mp": (1024, 1024),
        "1.31mp": (1280, 1024),
        "4000x3000": (4000, 3000),
        "6000x1000": (6000, 1000),
    }
    for label, (width, height) in sizes.items():
        report["single"].append(record("image", label, image_content(width, height)))
    cap = jpeg(1280, 1024)
    for tokens in (0, 128, 512, 1024, 2048):
        text = "" if tokens == 0 else text_of(tokens)
        report["single"].append(record("image_text", f"cap+{tokens}", image_content_from_url(cap, text)))

    under_budget = [
        item["input_tokens"]
        for item in report["single"]
        if item["status"] == 200 and item["peak_vram_bytes"] < PEAK_BUDGET and item["input_tokens"]
    ]
    token_budget = max(under_budget) if under_budget else 1024
    report["max_input_tokens"] = token_budget
    print(f"token_budget={token_budget}")

    print("--- concurrency ---")
    attribute = "The product_type is 'running sneaker'. The color is 'white'. "
    report["concurrency"] = concurrency_sweep(cap, attribute)
    print("--- priority ---")
    report["priority"] = priority_check(cap, attribute)
    print(report["priority"])
    print("--- soak ---")
    report["soak"] = soak()
    report["recommended_concurrency"] = recommend(report["concurrency"], report["priority"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}")
    print(f"recommended_concurrency={report['recommended_concurrency']}")


if __name__ == "__main__":
    main()
