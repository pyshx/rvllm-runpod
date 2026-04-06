# rvLLM RunPod

RunPod Serverless wrapper for [`rvLLM`](https://github.com/pyshx/rvllm). Launches the Rust inference server as a subprocess and proxies RunPod jobs to its OpenAI-compatible API.

## Quick Start

```bash
# 1. Set your RunPod API key
export RUNPOD_API_KEY=rpa_xxxxx

# 2. Deploy (creates template + endpoint, scales to zero when idle)
make deploy MODEL_ID=Qwen/Qwen2.5-7B-Instruct

# 3. Chat (first request triggers cold start ~2 min)
make chat MSG="What is the capital of Japan?"

# 4. Check status
make status

# 5. Stop (scale to zero)
make stop
```

That's it. The endpoint scales to zero when idle — you only pay during inference.

## Architecture

```
RunPod Job -> handler.py -> proxy -> rvllm serve (localhost:8000) -> GPU
```

The wrapper does three things:

1. Launch `rvllm serve` with env-driven configuration
2. Poll `/health` until the server is ready
3. Proxy RunPod jobs to the local OpenAI-compatible API

No inference logic lives here — that stays in rvLLM.

## Deploy Options

### Option 1: Use the pre-built image (recommended)

The CI publishes `ghcr.io/pyshx/rvllm-runpod:latest`. The deploy script uses this by default:

```bash
make deploy MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

### Option 2: Build your own image

```bash
# Generic (model downloaded at runtime)
./scripts/build.sh --tag myregistry/rvllm-runpod:latest --push

# Baked (model inside the image — faster cold starts)
HF_TOKEN=hf_xxx ./scripts/build.sh \
  --tag myregistry/rvllm-runpod:qwen25-7b \
  --bake-model \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --push

# Deploy with custom image
make deploy MODEL_ID=Qwen/Qwen2.5-7B-Instruct IMAGE=myregistry/rvllm-runpod:latest
```

## Make Targets

```
make help       Show all targets
make test       Run unit tests (93 tests)
make build      Build Docker image locally
make deploy     Deploy to RunPod (creates template + endpoint)
make status     Check endpoint health
make chat       Send a chat message (MSG="hello")
make start      Scale endpoint to 1 worker
make stop       Scale endpoint to 0 workers
make logs       Open RunPod logs in browser
```

## Usage

### Via make

```bash
make chat MSG="Write a Python function that checks if a number is prime"
```

### Via curl

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "authorization: <RUNPOD_API_KEY>" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Hello"}],
      "temperature": 0,
      "max_tokens": 128
    }
  }'
```

### Via test scripts

```bash
RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh all
```

## Configuration

### Runtime

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | - | Hugging Face model id |
| `MODEL_TARGET` | - | Override: actual value passed to `rvllm serve --model` |
| `SERVED_MODEL_NAME` | `MODEL_ID` | Public model name exposed to clients |
| `HF_TOKEN` | - | For gated/private models |
| `MAX_CONCURRENCY` | `30` | RunPod worker concurrency |
| `SERVER_READY_TIMEOUT` | `900` | Startup health-check timeout (seconds) |
| `REQUEST_TIMEOUT` | `600` | Proxy request timeout (seconds) |

### rvLLM Launch

| Variable | Default |
| --- | --- |
| `DTYPE` | `auto` |
| `MAX_MODEL_LEN` | `2048` |
| `GPU_MEMORY_UTILIZATION` | `0.9` |
| `TENSOR_PARALLEL_SIZE` | `1` |
| `MAX_NUM_SEQS` | `256` |
| `RUST_LOG` | `info` |

## Development

```bash
git clone https://github.com/pyshx/rvllm
git clone https://github.com/pyshx/rvllm-runpod

cd rvllm-runpod
cp .env.example .env  # add your RUNPOD_API_KEY
uv venv && uv pip install -e ".[dev]"
make test
```

## Layout

```
rvllm-runpod/
├── .runpod/hub.json          # RunPod hub metadata
├── builder/
│   ├── download_model.py     # HF model downloader for baked images
│   └── requirements.txt
├── examples/
│   ├── test_input*.json      # Sample job inputs
│   ├── test_endpoint.sh      # Test a live endpoint
│   └── test_local.sh         # Test locally against running rvllm
├── scripts/
│   ├── build.sh              # Docker build
│   ├── deploy.sh             # RunPod deploy (template + endpoint)
│   └── smoke_test.sh         # Run tests
├── src/
│   ├── config.py             # Env-driven configuration
│   ├── handler.py            # RunPod serverless entry point
│   ├── proxy.py              # HTTP proxy to rvllm serve
│   ├── request_mapping.py    # Job input -> OpenAI API mapping
│   └── server_launcher.py    # rvllm process lifecycle
└── tests/                    # 93 tests
```
