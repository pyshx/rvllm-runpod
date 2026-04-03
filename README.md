# rvLLM Serverless for RunPod

RunPod Serverless wrapper for [`rvLLM`](https://github.com/pyshx/rvllm), keeping the Rust inference server intact and adding only the minimal serverless layer needed for deployment.

## Architecture

```
RunPod Job -> handler.py -> proxy -> rvllm serve (localhost:8000) -> GPU inference
```

The wrapper does three things:

1. Launch `rvllm serve` with env-driven configuration
2. Wait for `/health`
3. Proxy RunPod jobs to the local OpenAI-compatible API

## Quick Start

### Option 1: Generic Image (model downloaded at runtime)

```bash
# Build
./scripts/build.sh --tag pyshx/rvllm-serverless:latest --push

# RunPod env vars
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
DTYPE=half
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_SEQS=16
MAX_CONCURRENCY=4
```

### Option 2: Baked Image (model inside the image)

```bash
HF_TOKEN=hf_xxx ./scripts/build.sh \
  --tag pyshx/rvllm-serverless:qwen25-7b \
  --bake-model \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --push
```

### Deploy on RunPod

1. Create a **Custom deployment** > **Deploy from Docker registry**
2. Use your image tag
3. Set endpoint type to **Queue-based**
4. Add env vars (see Configuration below)

## Usage

### Chat Completion

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "authorization: <RUNPOD_API_KEY>" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "user", "content": "What is rvLLM?"}
      ],
      "temperature": 0.2,
      "max_tokens": 128
    }
  }'
```

### Streaming

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "authorization: <RUNPOD_API_KEY>" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true,
      "max_tokens": 128
    }
  }'
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | - | Hugging Face model id |
| `DTYPE` | `auto` | `auto`, `half`, `bfloat16`, `float32` |
| `MAX_MODEL_LEN` | `2048` | Max sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | VRAM fraction |
| `TENSOR_PARALLEL_SIZE` | `1` | Multi-GPU parallelism |
| `MAX_NUM_SEQS` | `256` | Max concurrent sequences |
| `MAX_CONCURRENCY` | `30` | RunPod worker concurrency |
| `HF_TOKEN` | - | For gated/private models |

## Development

```bash
# Clone both repos side by side
git clone https://github.com/pyshx/rvllm
git clone https://github.com/pyshx/rvllm-serverless

# Run tests
cd rvllm-serverless
./scripts/smoke_test.sh
```

## Layout

```
rvllm-serverless/
├── .runpod/hub.json          # RunPod hub metadata
├── builder/
│   ├── download_model.py     # HF model downloader for baked images
│   └── requirements.txt
├── scripts/
│   ├── build.sh              # Docker build script
│   └── smoke_test.sh
├── src/
│   ├── config.py             # Env-driven config
│   ├── handler.py            # RunPod serverless entry point
│   ├── proxy.py              # HTTP proxy to rvllm serve
│   ├── request_mapping.py    # Job input -> OpenAI API mapping
│   └── server_launcher.py    # rvllm process lifecycle
└── tests/
```
