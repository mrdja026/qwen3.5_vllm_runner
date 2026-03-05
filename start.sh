#!/usr/bin/env bash
set -euo pipefail

proxy_pid=""

cleanup() {
  if [[ -n "${proxy_pid}" ]]; then
    kill "${proxy_pid}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

python -m qwen_app.api &
proxy_pid=$!

python -m qwen_app.tui
