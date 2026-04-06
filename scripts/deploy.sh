#!/usr/bin/env bash
#
# Deploy rvllm-runpod to RunPod Serverless.
# Creates a template + endpoint, or updates an existing one.
#
# Usage:
#   # First deploy (creates template + endpoint):
#   RUNPOD_API_KEY=xxx ./scripts/deploy.sh --model Qwen/Qwen2.5-7B-Instruct
#
#   # Scale existing endpoint:
#   RUNPOD_API_KEY=xxx ./scripts/deploy.sh --endpoint <id> --workers 0
#
set -euo pipefail

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
IMAGE="ghcr.io/pyshx/rvllm-runpod:latest"

MODEL_ID=""
GPU="AMPERE_80"
ENDPOINT_ID=""
WORKERS=""
MAX_MODEL_LEN="2048"
DTYPE="half"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_ID="$2";       shift 2 ;;
    --gpu)         GPU="$2";            shift 2 ;;
    --image)       IMAGE="$2";          shift 2 ;;
    --endpoint)    ENDPOINT_ID="$2";    shift 2 ;;
    --workers)     WORKERS="$2";        shift 2 ;;
    --max-len)     MAX_MODEL_LEN="$2";  shift 2 ;;
    --dtype)       DTYPE="$2";          shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

gql() {
  curl -s "${API}" -H "content-type: application/json" -d "{\"query\": \"$1\"}"
}

# --- Scale existing endpoint ---
if [[ -n "${ENDPOINT_ID}" && -n "${WORKERS}" ]]; then
  echo "Scaling endpoint ${ENDPOINT_ID} to ${WORKERS} workers..."
  gql "mutation { saveEndpoint(input: { id: \\\"${ENDPOINT_ID}\\\", workersMax: ${WORKERS} }) { id workersMax } }" | python3 -c "
import sys, json
r = json.loads(sys.stdin.read())
d = r.get('data', {}).get('saveEndpoint', {})
print(f\"  workersMax={d.get('workersMax', '?')}\")
" 2>/dev/null || echo "  (done)"
  exit 0
fi

# --- Full deploy ---
if [[ -z "${MODEL_ID}" ]]; then
  echo "Usage: $0 --model <hf_model_id> [--gpu AMPERE_80] [--image <image>]" >&2
  exit 1
fi

SAFE_NAME=$(echo "${MODEL_ID}" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

echo "==> Creating template for ${MODEL_ID}..."
TEMPLATE_ID=$(gql "mutation { saveTemplate(input: { name: \\\"rvllm-${SAFE_NAME}\\\", imageName: \\\"${IMAGE}\\\", dockerArgs: \\\"\\\", containerDiskInGb: 30, volumeInGb: 0, env: [{key: \\\"MODEL_ID\\\", value: \\\"${MODEL_ID}\\\"}, {key: \\\"DTYPE\\\", value: \\\"${DTYPE}\\\"}, {key: \\\"MAX_MODEL_LEN\\\", value: \\\"${MAX_MODEL_LEN}\\\"}, {key: \\\"GPU_MEMORY_UTILIZATION\\\", value: \\\"0.90\\\"}, {key: \\\"MAX_NUM_SEQS\\\", value: \\\"4\\\"}, {key: \\\"MAX_CONCURRENCY\\\", value: \\\"2\\\"}, {key: \\\"SERVER_READY_TIMEOUT\\\", value: \\\"600\\\"}], isServerless: true }) { id name } }" \
  | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['data']['saveTemplate']['id'])" 2>/dev/null)

if [[ -z "${TEMPLATE_ID}" ]]; then
  echo "Failed to create template." >&2
  exit 1
fi
echo "  Template: ${TEMPLATE_ID}"

echo "==> Creating endpoint..."
RESULT=$(gql "mutation { saveEndpoint(input: { name: \\\"rvllm-${SAFE_NAME}\\\", templateId: \\\"${TEMPLATE_ID}\\\", gpuIds: \\\"${GPU}\\\", workersMin: 0, workersMax: 1, idleTimeout: 60, scalerType: \\\"QUEUE_DELAY\\\", scalerValue: 1 }) { id name gpuIds } }")

NEW_ENDPOINT_ID=$(echo "${RESULT}" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['data']['saveEndpoint']['id'])" 2>/dev/null)

if [[ -z "${NEW_ENDPOINT_ID}" ]]; then
  echo "Failed to create endpoint." >&2
  echo "${RESULT}"
  exit 1
fi

echo "${NEW_ENDPOINT_ID}" > "${REPO_DIR}/.endpoint"
echo "  Endpoint: ${NEW_ENDPOINT_ID}"
echo ""
echo "==> Deployed! The worker will start on first request."
echo ""
echo "  Test:   make chat MSG=\"hello\""
echo "  Status: make status"
echo "  Stop:   make stop"
echo "  Logs:   make logs"
