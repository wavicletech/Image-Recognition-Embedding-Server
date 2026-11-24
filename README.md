# Qwen3-VL embedding server

Self-hosted `Qwen/Qwen3-VL-Embedding-2B` in fp16. It accepts the Simplismart chat payload on `POST /v1/embeddings` and returns an OpenAI-style embedding: 2048 floats, L2-normalized. One GPU runs one embedding at a time. Requests with `"source": "bot_builder"` are served ahead of `"source": "kbm"`.

Caller contract, payload examples, and error handling are in [CLIENT.md](CLIENT.md). How Bot Builder and the KBM catalog worker should call it is in [CLIENT_GUIDE.md](CLIENT_GUIDE.md).

## Layout

| Path | What it is |
|---|---|
| `app/` | FastAPI server, payload mapping, and the priority queue |
| `bench/benchmark.py` | Capacity run. Writes `bench/results/<timestamp>.json` |
| `bench/parity.py` | Compares the live server with stored Simplismart vectors. Fails under cosine similarity 0.999 |
| `bench/fixtures/gull.jpg` | Image used by the parity check |
| `run_example.py` | Loads the model and times the example text payload |
| `start_qwen_model_server.sh` | Starts the detached `qwenmodelserver` tmux session |
| `run_qwen_loop.sh` | Restarts the process if it exits, and appends `logs/server.log` |
| `watch_qwen_model_server.sh` | Health check. Kills a hung process after three failures |

## Run

Python is the CUDA 12.8 environment in `.conda` (not in git). Torch comes from the cu128 wheel index, as noted in `requirements.txt`.

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Create `.env` (not in git) and start the server:

```bash
QWEN_EMBED_API_KEY=...
QWEN_EMBED_HOST=0.0.0.0
QWEN_EMBED_PORT=8090
QWEN_MAX_INPUT_TOKENS=4116
QWEN_MAX_IMAGE_BYTES=15728640
QWEN_MAX_WAITING_BOT_BUILDER=16
QWEN_MAX_WAITING_KBM=8
QWEN_WAIT_TIMEOUT_BOT_BUILDER_S=30
QWEN_WAIT_TIMEOUT_KBM_S=120
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```bash
./start_qwen_model_server.sh
tmux attach -t qwenmodelserver
```

`GET /health` needs no key. `GET /ready` returns 503 until the model has finished warming up.

## Request log

Each embedding call writes one line to `logs/server.log` through the `qwen_embedding` logger. A success line is `embed ok` and includes status, client, token count, queue time, inference time, peak VRAM, and the input summary. A failure line is `embed fail` and adds the status and error text. That covers 401, 400, 413, 422, 500, and 503.

The summary records `model`, `source`, `encoding_format`, `continue_final_message`, `add_special_tokens`, and each message: role, text length with a short preview, or the image or video URL. Inline images are logged as `image:data:<type>:<size>b64`. The bearer token and the embedding vector are not logged.

```text
embed ok status=200 client=127.0.0.1 path=/v1/embeddings elapsed_ms=104.9 tokens=26 queue_ms=0.00 inference_ms=31.44 peak_vram_bytes=4274120192 model='Qwen/Qwen3-VL-Embedding-2B' source='bot_builder' encoding_format='float' continue_final_message=True add_special_tokens=True messages=1 [user:[text:26 "The product_type is dress."]]
```

## Not in git

`.conda/`, `.env`, `logs/`, `*.log`, `__pycache__/`, `example_result.json`, and `bench/results/` are local. See `.gitignore`.
