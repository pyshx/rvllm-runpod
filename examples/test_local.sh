#!/usr/bin/env bash
#
# Test handler locally against a running rvllm server.
#
# Prerequisites:
#   1. rvllm must be running: rvllm serve --model <model> --port 8000
#   2. Install deps: uv venv && uv pip install -e ".[dev]"
#
# Usage:
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct ./examples/test_local.sh
#
# This starts RunPod's local test server on port 8080, then sends
# a sample request to it.
#
set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to the model rvllm is serving}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_ID
export RVLLM_PORT="${RVLLM_PORT:-8000}"
export SERVER_READY_TIMEOUT=5

echo "==> Starting RunPod local test server on :8080"
echo "    Expecting rvllm on localhost:${RVLLM_PORT}"
echo ""

cd "${REPO_DIR}/src"

# RunPod's local server reads RUNPOD_WEBHOOK_GET_JOB to decide mode.
# Without it, it falls back to test_input.json or --test_input flag.
PYTHONPATH="${REPO_DIR}/src" python3 -c "
import asyncio, json, sys
sys.path.insert(0, '.')
from handler import handler

job = json.load(open('${REPO_DIR}/examples/test_input.json'))

async def run():
    results = []
    async for item in handler(job):
        results.append(item)
    print(json.dumps(results, indent=2))

asyncio.run(run())
" 2>&1

echo ""
echo "==> Done"
