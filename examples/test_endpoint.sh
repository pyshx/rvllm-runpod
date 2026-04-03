#!/usr/bin/env bash
#
# Test a live RunPod endpoint with sample inputs.
#
# Usage:
#   RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh
#   RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh stream
#   RUNPOD_API_KEY=xxx ENDPOINT_ID=yyy ./examples/test_endpoint.sh models
#
set -euo pipefail

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
: "${ENDPOINT_ID:?Set ENDPOINT_ID}"

BASE="https://api.runpod.ai/v2/${ENDPOINT_ID}"
AUTH="authorization: ${RUNPOD_API_KEY}"
CT="content-type: application/json"

MODE="${1:-chat}"

case "${MODE}" in
  chat)
    echo "==> Chat completion (runsync)"
    curl -s -X POST "${BASE}/runsync" \
      -H "${AUTH}" -H "${CT}" \
      -d @examples/test_input.json | python3 -m json.tool
    ;;

  stream)
    echo "==> Streaming chat (run + poll)"
    JOB=$(curl -s -X POST "${BASE}/run" \
      -H "${AUTH}" -H "${CT}" \
      -d @examples/test_input_stream.json)
    JOB_ID=$(echo "${JOB}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "    Job ID: ${JOB_ID}"
    echo "    Polling stream..."
    sleep 1
    curl -s "${BASE}/stream/${JOB_ID}" -H "${AUTH}" | python3 -m json.tool
    ;;

  completion)
    echo "==> Text completion (runsync)"
    curl -s -X POST "${BASE}/runsync" \
      -H "${AUTH}" -H "${CT}" \
      -d @examples/test_input_completion.json | python3 -m json.tool
    ;;

  models)
    echo "==> List models (runsync)"
    curl -s -X POST "${BASE}/runsync" \
      -H "${AUTH}" -H "${CT}" \
      -d @examples/test_input_models.json | python3 -m json.tool
    ;;

  explicit)
    echo "==> Explicit proxy (runsync)"
    curl -s -X POST "${BASE}/runsync" \
      -H "${AUTH}" -H "${CT}" \
      -d @examples/test_input_explicit.json | python3 -m json.tool
    ;;

  all)
    for m in chat stream completion models explicit; do
      "$0" "${m}"
      echo ""
    done
    ;;

  *)
    echo "Usage: $0 {chat|stream|completion|models|explicit|all}" >&2
    exit 1
    ;;
esac
