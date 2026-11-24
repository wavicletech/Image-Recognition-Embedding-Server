"""Load Qwen3-VL-Embedding-2B in fp16 and time the example chat payload."""

import json
import statistics
import time
from pathlib import Path

import torch

from app.service import MODEL_ID, embed_payload, gpu_stats

PAYLOAD = {
    "model": "Qwen/Qwen3-VL-Embedding-2B",
    "encoding_format": "float",
    "continue_final_message": True,
    "add_special_tokens": True,
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Represent the user's input."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "red floral maxi dress under 3000"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    ],
}

RUNS = 5


def main() -> None:
    print(f"model={MODEL_ID} dtype=float16")
    print(f"gpu_before_load={gpu_stats()}")

    started = time.perf_counter()
    warmup = embed_payload(PAYLOAD)
    load_and_first_s = time.perf_counter() - started
    print(f"load_plus_first_request_s={load_and_first_s:.3f}")
    print(f"gpu_after_first={gpu_stats()}")

    timings = []
    last = warmup
    for index in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = embed_payload(PAYLOAD)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        print(f"run_{index + 1}_ms={elapsed_ms:.2f}")

    vector = last["embedding"]
    norm = sum(value * value for value in vector) ** 0.5
    report = {
        "model": MODEL_ID,
        "dtype": "float16",
        "device": gpu_stats(),
        "dimensions": last["dimensions"],
        "l2_norm": norm,
        "load_plus_first_request_s": load_and_first_s,
        "warmup_peak_vram_mib": warmup["peak_vram_bytes"] / (1024 * 1024),
        "steady_peak_vram_mib": last["peak_vram_bytes"] / (1024 * 1024),
        "latency_ms": {
            "runs": timings,
            "min": min(timings),
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
        },
        "embedding_head": vector[:8],
        "embedding": vector,
    }
    out = Path(__file__).resolve().parent / "example_result.json"
    out.write_text(json.dumps(report))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "embedding"},
            indent=2,
        )
    )
    print(f"wrote={out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
