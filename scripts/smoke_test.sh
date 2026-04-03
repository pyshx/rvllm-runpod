#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

if [[ -f .venv/bin/python ]]; then
  .venv/bin/python -m pytest tests/ -v "$@"
else
  python3 -m pytest tests/ -v "$@"
fi
