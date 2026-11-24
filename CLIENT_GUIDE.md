# Point visual search at this embedding server

The server is `Qwen/Qwen3-VL-Embedding-2B` in fp16 on the RTX 2080. It speaks the same `POST /v1/embeddings` body Simplismart does. Vectors are 2048 dimensions and L2-normalized.

| Setting | Value |
|---|---|
| Base URL | `http://74.91.17.202:8090/v1` |
| API key | `QWEN_EMBED_API_KEY` in `/home/customer/qwen_embedding/.env` |
| Model | `Qwen/Qwen3-VL-Embedding-2B` |
| Deployment id | leave empty |

Checked against Simplismart references on 23 Nov 2026:

| Request | Cosine similarity | Prompt tokens |
|---|---|---|
| Text: `The product_type is 'dress'. The color is 'red'. ` | 0.99984 | 34, same as Simplismart |
| Image only, 2272x1704 photo, empty text part | 0.99953 | 1252, same as Simplismart |

That is close enough to keep the existing Pinecone vectors. Do not mix in vectors from a different instruction or a different image-size cap.

## Bot Builder

In `helpers/visual_search/embedder.py`, add `"source": "bot_builder"` to the JSON body. Nothing else in the body changes.

```
SIMPLISMART_IMAGE_EMBEDDING_BASE_URL=http://74.91.17.202:8090/v1
SIMPLISMART_IMAGE_EMBEDDING_API_KEY=<QWEN_EMBED_API_KEY>
SIMPLISMART_IMAGE_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
SIMPLISMART_IMAGE_EMBEDDING_DEPLOYMENT_ID=
```

Set the client timeout to 60 seconds. `source` is how this server tells a user request from a bulk job. The same API key is used for both.

## KBM catalog worker

In `src/product_catalog/simplismart_embedder.py`, add `"source": "kbm"`.

```
KBM_SIMPLISMART_IMAGE_EMBEDDING_BASE_URL=http://74.91.17.202:8090/v1
KBM_SIMPLISMART_IMAGE_EMBEDDING_API_KEY=<same key>
KBM_SIMPLISMART_IMAGE_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
KBM_SIMPLISMART_IMAGE_EMBEDDING_DEPLOYMENT_ID=
```

Set `CONCURRENCY = 4` in `src/product_catalog/visual_search_ingest.py`. Keep `BATCH_SIZE = 10`, `replicas: 1`, `--pool=solo`, and image-then-text in sequence.

Measured with a 2 second stand-in for the Azure attribute call, 8 variants per setting:

| CONCURRENCY | Variants / s | Embed p95 | Errors |
|---|---|---|---|
| 1 | 0.36 | 0.69 s | 0 |
| 2 | 0.58 | 1.19 s | 0 |
| 3 | 0.69 | 1.70 s | 0 |
| 4 | 0.96 | 2.15 s | 0 |
| 5 | 0.90 | 2.84 s | 0 |
| 6 | 0.85 | 3.20 s | 0 |

Four is the peak. Five and six are slower because the GPU is already busy and the extra requests only wait. While five bulk workers were running, Bot Builder requests (the server lets them go first) had p50 277 ms and p95 753 ms.

If the server starts returning 503 or 413 during a real ingest, set `CONCURRENCY = 2`, then `1`. That needs a catalog-worker deploy. Do not add a second replica.

## What the server does with load

One request uses the GPU at a time. Bot Builder goes next, ahead of KBM, but it still waits for the request already running. A full-size image takes about 0.5 s. A short attribute sentence takes about 35 ms.

Any image size is accepted. The model resizes it to at most 1,310,720 pixels, so a 4000x3000 photo costs the same as one already at that cap (about 0.5 s and 1,250 tokens). The 15 MB limit is on the download, not the dimensions.

Text plus image tokens above 4,116 are rejected with **413** before the GPU is used. Real catalog text and user queries are far under that. A 6,821-token text was rejected in 12 ms.

After every request the GPU allocation returns to the weight baseline, about 4.1 GiB. A 1,000-request soak stayed within 150 MiB of that baseline.

**503** with `Retry-After` means the wait queue is full or the wait timed out. The OpenAI client retries that. **413** will not succeed on retry; that variant should be abandoned.

## Image text: empty string and a single space are different

Bot Builder `embed_image_only` sends an empty text part. KBM sends a single space. On this model those are different inputs: cosine similarity between them for the same photo was **0.9556**, because the space is an extra token and the embedding is the last token. The Simplismart image reference was captured with an empty text part. If Simplismart behaves the same way, the KBM index and Bot Builder queries already disagree. Get a Simplismart vector for the same image with the text part `" "`. If the gap is there, change KBM to send `""` and re-ingest the IMAGE vectors. The server will not rewrite either caller's text.

## Rollout

1. Staging: point Bot Builder and one KBM account at this URL. Watch `GET /health` and `tmux attach -t qwenmodelserver`.
2. Production after staging queries and one ingest look right.
3. To roll back, point the environment settings back at Simplismart. The `source` field can stay; Simplismart ignores it.

`bench/parity.py` reruns the two Simplismart comparisons and fails if cosine similarity drops below 0.999.
