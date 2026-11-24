# Calling the embedding server

Self-hosted replacement for the Simplismart visual-search embeddings API. One GPU, so one embedding runs at a time. Send `"source": "bot_builder"` for a user request so it is served ahead of catalog ingestion.

## Endpoint

| | |
|---|---|
| URL | `POST http://74.91.17.202:8090/v1/embeddings` |
| Auth | `Authorization: Bearer <QWEN_EMBED_API_KEY>` |
| Key | `/home/customer/qwen_embedding/.env` |
| Model | `Qwen/Qwen3-VL-Embedding-2B` |
| Deployment `id` header | do not send |
| Client timeout | 60 seconds |

Health check, no auth: `GET http://74.91.17.202:8090/health`

The body is the same chat-style payload Simplismart accepts. The only new field is `source`.

| Field | Value |
|---|---|
| `model` | `Qwen/Qwen3-VL-Embedding-2B` |
| `encoding_format` | `"float"` |
| `continue_final_message` | `true` |
| `add_special_tokens` | `true` |
| `source` | `"bot_builder"` or `"kbm"` |
| system message | `Represent the user's input.` |
| assistant message | empty text |

Read `data[0].embedding`. It is 2048 floats, already L2-normalized. The client may normalize again.

## How many calls at once

The GPU runs **1 embedding at a time**. Other calls wait.

| Caller | What to set | Why |
|---|---|---|
| Bot Builder | one embed per user action, `source` = `bot_builder` | A same-product search sends 1 call. A different-product search sends 2, one after the other. |
| KBM catalog worker | `CONCURRENCY = 4`, `source` = `kbm` | Four variants in flight. Each variant does image, then text, not both at once. |

Do not raise KBM above 4. Five and six were slower in the benchmark because they only waited on the same GPU. Keep one catalog pod, the solo Celery pool, and `BATCH_SIZE` 10.

A full-size image takes about **0.5 s**. A short attribute sentence takes about **35 ms**. A Bot Builder text call while bulk ingestion is running was **p95 0.75 s**.

## Text only

Used for the TEXT Pinecone vector and for a text query.

```bash
curl -sS -X POST "http://74.91.17.202:8090/v1/embeddings" \
  -H "Authorization: Bearer $QWEN_EMBED_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "encoding_format": "float",
  "continue_final_message": true,
  "add_special_tokens": true,
  "source": "bot_builder",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
    {"role": "user", "content": [{"type": "text", "text": "The product_type is 'dress'. The color is 'red'. "}]},
    {"role": "assistant", "content": [{"type": "text", "text": ""}]}
  ]
}
EOF
```

## Image only

The server downloads `image_url`. The URL must be public `http` or `https`. Any pixel size is accepted; the model resizes it to at most 1,310,720 pixels. Downloads over 15 MB are rejected.

Bot Builder sends the text part as `""`. KBM currently sends `" "`. Those are not the same embedding (cosine similarity 0.956 on one photo). Keep sending whichever string built the index you are querying.

```json
{
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "encoding_format": "float",
  "continue_final_message": true,
  "add_special_tokens": true,
  "source": "kbm",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://cdn.example.com/variant.jpg"}},
        {"type": "text", "text": ""}
      ]
    },
    {"role": "assistant", "content": [{"type": "text", "text": ""}]}
  ]
}
```

## Image plus text

One forward pass. Use this only if the index was built the same way. Product Search V2 queries embed the image and the text separately.

```json
"content": [
  {"type": "image_url", "image_url": {"url": "https://cdn.example.com/variant.jpg"}},
  {"type": "text", "text": "The product_type is 'saree'. The color is 'blue'. "}
]
```

## Response

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.01, -0.02]}
  ],
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "usage": {"prompt_tokens": 34, "total_tokens": 34}
}
```

Headers: `X-Queue-Ms`, `X-Inference-Ms`, `X-Input-Tokens`, `X-Peak-Vram-Bytes`.

## Errors

| Status | Meaning | What the client should do |
|---|---|---|
| 401 | Missing or wrong bearer token | Fix the key. |
| 400 | Bad body, or the image URL could not be fetched | Do not retry the same URL. |
| 413 | More than 4,116 tokens, or the GPU could not fit the input | Abandon this item. Retrying will fail again. Normal product text and photos are under the limit. |
| 503 | The wait queue is full or the wait timed out. `Retry-After` is set. | Retry. The OpenAI Python client does this twice by default. |
| 500 | Unexpected server error | Retry once. The process restarts itself if the GPU is left unusable. |

## Python

The OpenAI SDK posts to `{base_url}/embeddings`, so the base URL must end in `/v1`.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://74.91.17.202:8090/v1",
    api_key=API_KEY,
    timeout=60.0,
)

response = client.post(
    "/embeddings",
    cast_to=dict,
    body={
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "encoding_format": "float",
        "continue_final_message": True,
        "add_special_tokens": True,
        "source": "bot_builder",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": [{"type": "text", "text": "The product_type is 'dress'. The color is 'red'. "}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ],
    },
)
vector = response["data"][0]["embedding"]
```

## Where to change each client

| Client | Code | Env vars | Concurrency |
|---|---|---|---|
| Bot Builder | `helpers/visual_search/embedder.py`, add `source: bot_builder` | `SIMPLISMART_IMAGE_EMBEDDING_BASE_URL`, `_API_KEY`, `_MODEL`. Leave `_DEPLOYMENT_ID` empty. | Leave the query path as it is: image and text are separate calls. |
| KBM | `src/product_catalog/simplismart_embedder.py`, add `source: kbm` | `KBM_SIMPLISMART_IMAGE_EMBEDDING_*`, same values. Leave deployment id empty. | `CONCURRENCY = 4` in `visual_search_ingest.py`. |

To go back to Simplismart, point the base URL and key at Simplismart again. `source` can stay in the body.
