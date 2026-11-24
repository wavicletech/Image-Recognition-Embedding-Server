"""Compare the live server with stored Simplismart reference vectors.

Exit 1 if cosine similarity drops below 0.999.
"""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE = os.environ.get("QWEN_BENCH_URL", "http://127.0.0.1:8090")
KEY = os.environ["QWEN_EMBED_API_KEY"]
REFERENCES = Path(os.environ.get("QWEN_PARITY_DIR", "/home/customer/.cursor/plans"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gull.jpg"
MINIMUM = 0.999


def load_json(path: Path) -> dict:
    raw = path.read_text()
    return json.loads(raw[raw.index("{"):])


def embed(content) -> np.ndarray:
    payload = {
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
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
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode())
        tokens = response.headers.get("X-Input-Tokens")
    vector = np.asarray(body["data"][0]["embedding"], dtype=np.float64)
    return vector / np.linalg.norm(vector), tokens


def cosine(reference: dict, vector: np.ndarray) -> float:
    expected = np.asarray(reference["embedding"], dtype=np.float64)
    expected = expected / np.linalg.norm(expected)
    return float(expected @ vector)


def main() -> int:
    failed = False
    text_ref = load_json(REFERENCES / "simplimsart.json")
    text_vector, text_tokens = embed([{"type": "text", "text": text_ref["input_text"]}])
    text_score = cosine(text_ref, text_vector)
    print(f"text cosine={text_score:.6f} tokens={text_tokens} expected_tokens={text_ref['usage']['prompt_tokens']}")
    failed = failed or text_score < MINIMUM

    image_ref = load_json(REFERENCES / "simplimsart_image.json")
    encoded = base64.b64encode(FIXTURE.read_bytes()).decode()
    image_vector, image_tokens = embed([
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
        {"type": "text", "text": ""},
    ])
    image_score = cosine(image_ref, image_vector)
    print(
        f"image cosine={image_score:.6f} tokens={image_tokens} "
        f"expected_tokens={image_ref['usage']['prompt_tokens']}"
    )
    failed = failed or image_score < MINIMUM
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
