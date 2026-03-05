#!/usr/bin/env bash
set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "python or python3 is required" >&2
  exit 1
fi

PROMPT="${1:-What is your name?}"
export QWEN_PROMPT="${PROMPT}"

PROXY_URL="${QWEN_PROXY_URL:-http://localhost:9000}"
MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
TIMEOUT="${QWEN_STREAM_TIMEOUT:-20}"

payload_file="$(mktemp)"
response_file="$(mktemp)"

cleanup() {
  rm -f "${payload_file}" "${response_file}"
}

trap cleanup EXIT

"${PYTHON_BIN}" - <<'PY' > "${payload_file}"
import json
import os

prompt = os.environ.get("QWEN_PROMPT", "What is your name?")
model = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-4B")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7,
    "max_tokens": 200,
    "stream": True,
}

print(json.dumps(payload))
PY

status=$(curl -sS -N --max-time "${TIMEOUT}" \
  -H "Content-Type: application/json" \
  -d @"${payload_file}" \
  -o "${response_file}" \
  -w "%{http_code}" \
  "${PROXY_URL}/v1/chat/completions")

cat "${response_file}"

if [[ "${status}" != "200" ]]; then
  echo "HTTP ${status}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY' "${response_file}"
import sys

path = sys.argv[1]
data = open(path, "r", encoding="utf-8", errors="replace").read()
if "data:" not in data:
    print("No SSE data received", file=sys.stderr)
    sys.exit(1)
if "[DONE]" not in data:
    print("Stream did not finish cleanly", file=sys.stderr)
    sys.exit(1)
print("OK")
PY
