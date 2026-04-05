# rvLLM RunPod

RunPod Serverless wrapper for [`rvLLM`](https://github.com/pyshx/rvllm). Launches the Rust inference server as a subprocess and proxies RunPod jobs to its OpenAI-compatible API.

## Architecture

```
RunPod Job -> handler.py -> proxy -> rvllm serve (localhost:8000) -> GPU
```

The wrapper does three things:

1. Launch `rvllm serve` with env-driven configuration
2. Poll `/health` until the server is ready
3. Proxy RunPod jobs to the local OpenAI-compatible API

No inference logic lives here — that stays in rvLLM.

## Quick Start

### Generic Image (model downloaded at runtime)

```bash
./scripts/build.sh --tag pyshx/rvllm-runpod:latest --push
```

### Baked Image (model inside the image)

```bash
HF_TOKEN=hf_xxx ./scripts/build.sh \
  --tag pyshx/rvllm-runpod:qwen25-7b \
  --bake-model \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --push
```

### Deploy on RunPod

1. Create a **Custom deployment** > **Deploy from Docker registry**
2. Use your image tag
3. Set endpoint type to **Queue-based**
4. Add env vars:

```env
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
DTYPE=half
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_SEQS=16
MAX_CONCURRENCY=4
```

For gated/private models, add `HF_TOKEN` as a RunPod Secret.

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

Then poll the stream:

```bash
curl "https://api.runpod.ai/v2/<ENDPOINT_ID>/stream/<JOB_ID>" \
  -H "authorization: <RUNPOD_API_KEY>"
```

### Explicit Proxy Input

For direct control over the local endpoint:

```json
{
  "input": {
    "path": "/v1/chat/completions",
    "method": "POST",
    "body": {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": true
    }
  }
}
```

## Configuration

### Runtime

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | - | Hugging Face model id |
| `MODEL_TARGET` | - | Override: actual value passed to `rvllm serve --model` |
| `SERVED_MODEL_NAME` | `MODEL_ID` | Public model name exposed to clients |
| `TOKENIZER_ID` | - | Optional tokenizer override |
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
| `DISABLE_TELEMETRY` | `false` |

## Examples

Test inputs live in `examples/`. Each covers a different job format:

| File | What it tests |
| --- | --- |
| `test_input.json` | Chat completion (non-streaming) |
| `test_input_stream.json` | Chat completion (streaming) |
| `test_input_completion.json` | Text completion |
| `test_input_models.json` | List models |
| `test_input_explicit.json` | Explicit proxy (path + method + body) |

### Test a live RunPod endpoint

```bash
RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh          # chat
RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh stream   # streaming
RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh all      # everything
```

### Test locally (against a running rvllm)

```bash
# Terminal 1: start rvllm
rvllm serve --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Terminal 2: run the handler
MODEL_ID=Qwen/Qwen2.5-7B-Instruct ./examples/test_local.sh
```

## Development

```bash
git clone https://github.com/pyshx/rvllm
git clone https://github.com/pyshx/rvllm-runpod

cd rvllm-runpod
uv venv && uv pip install -e ".[dev]"
```

### Run Tests

```bash
./scripts/smoke_test.sh
```

93 tests covering config, request mapping, proxy (streaming + non-streaming), and server launcher.

### Dry Run Build

```bash
./scripts/build.sh --tag test:latest --dry-run
```

## Layout

```
rvllm-runpod/
├── .runpod/hub.json          # RunPod hub metadata
├── builder/
│   ├── download_model.py     # HF model downloader for baked images
│   └── requirements.txt
├── examples/
│   ├── test_input.json       # Chat completion
│   ├── test_input_stream.json
│   ├── test_input_completion.json
│   ├── test_input_models.json
│   ├── test_input_explicit.json
│   ├── test_endpoint.sh      # Test a live RunPod endpoint
│   └── test_local.sh         # Test locally against running rvllm
├── scripts/
│   ├── build.sh              # Docker build script
│   └── smoke_test.sh
├── src/
│   ├── config.py             # Env-driven configuration
│   ├── handler.py            # RunPod serverless entry point
│   ├── proxy.py              # HTTP proxy to rvllm serve
│   ├── request_mapping.py    # Job input -> OpenAI API mapping
│   └── server_launcher.py    # rvllm process lifecycle
└── tests/
    ├── test_config.py
    ├── test_proxy.py
    ├── test_request_mapping.py
    └── test_server_launcher.py
```
# 1775341059
# 1775392937
# 1775395088
# 1775398501
# 1775400047
# 1775403448
# 1775409889
