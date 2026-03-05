Great! Now that your local Qwen3.5-4B + vLLM setup is running nicely, here's a practical guide how to create a clean `requirements.txt` and a useful `README.md` for a repository that shares this project.

### 1. Creating `requirements.txt`

The easiest and most reliable way is to use `uv` (since you're already using it) or `pip`.

#### Option A – Recommended (using uv – clean & exact)
```bash
# Make sure you're in the activated venv
source qwen-env/bin/activate

# Export exact versions (including transitive deps)
uv pip freeze > requirements.txt
```

Or if you prefer **only the top-level packages** (cleaner for most repos):

```bash
# List only what you explicitly installed
uv pip list --format=freeze | grep -E '^(vllm|torch|transformers|huggingface-hub|accelerate|numpy|pillow|requests|jinja2|uvicorn|fastapi)' > requirements.txt
```

Typical minimal content for your setup (as of your current versions – adjust if needed):

```txt
# requirements.txt
vllm>=0.16.1  # or exact hash if you want: vllm @ git+https://github.com/vllm-project/vllm.git@main
transformers>=4.51
torch>=2.4.0  # usually comes with vllm
accelerate>=0.33
huggingface-hub>=0.23
uvicorn>=0.30
fastapi>=0.111
pydantic>=2.7
jinja2>=3.1
pillow>=10.0  # sometimes pulled by vllm for vision (optional)
```

#### Option B – Using plain pip (if you ever switch)
```bash
pip freeze > requirements.txt
```

Then edit it down to remove unnecessary packages (uvicorn[standard], etc.).

### 2. Creating a good `README.md`

Create a file called `README.md` in the root of your repository with this content (copy-paste and customize):

```markdown
# Qwen3.5-4B Local Inference with vLLM

Simple setup to run **Qwen/Qwen3.5-4B** (very capable 4B-parameter model with thinking/reasoning mode) locally using vLLM on a machine with ~16 GB VRAM (e.g. RTX 4080 Laptop).

Tested on **Windows + WSL2 + Ubuntu** with NVIDIA GPU.

## Features

- OpenAI-compatible API endpoint (`http://localhost:8000/v1`)
- Reasoning/thinking mode enabled (`--reasoning-parser qwen3`)
- Supports up to 32k–128k context (adjustable)
- Tool calling support (partial – works best with prompt engineering on 4B)
- Very fast inference (~60–90 tokens/s on RTX 4080)

## Prerequisites

- Windows 11 + WSL2 (Ubuntu 22.04/24.04 recommended)
- NVIDIA driver ≥ 550 (with WSL CUDA support)
- CUDA toolkit visible in WSL (`nvidia-smi` works)
- At least 16 GB VRAM GPU

## Installation

1. Create & activate virtual environment (using uv – fast)

```bash
uv venv qwen-env
source qwen-env/bin/activate
```

2. Install vLLM (use nightly for best Qwen3.5 support)

```bash
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
# Optional: force latest from git
# uv pip install -U "vllm @ git+https://github.com/vllm-project/vllm.git@main"
```

3. Optional – upgrade key dependencies

```bash
uv pip install -U transformers accelerate huggingface-hub
```

## Usage

Start the server:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --reasoning-parser qwen3
```

With tool calling (if you want to experiment):

```bash
vllm serve Qwen/Qwen3.5-4B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes    # or qwen_xml / mistral depending on version
```

Test with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Explain quantum entanglement like Im 12 years old"}],
    "temperature": 0.7,
    "max_tokens": 400
  }'
```

## Tips & Tricks

- Use `--max-model-len 65536` or `131072` for longer conversations
- Add `--enable-reasoning` if your vLLM version still supports it (older builds)
- For better tool calling → try Qwen3.5-7B or 14B (still fits in 16 GB)
- Monitor VRAM: `nvidia-smi` in another terminal

## License

Model: Apache 2.0 (Qwen license)  
Code in this repo:

Enjoy!
```

### Final steps for the repo

```bash
# In your project folder
echo "# Qwen3.5-4B Local vLLM Setup" > README.md
# paste the content above

uv pip freeze > requirements.txt
# or use the minimal version I showed

git init
git add README.md requirements.txt
git commit -m "Initial commit: Qwen3.5-4B local inference setup"

# Create repo on GitHub/GitLab → then
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

If you want to make it even nicer later:
- Add a `start.sh` script with the serve command
- Include screenshots of nvidia-smi + model output
- Add `.gitignore` (ignore `__pycache__`, `.cache/`, etc.)

# Non AI generated

# How to run (WSL setup)
https://grok.com/c/45a042e2-03e2-4f97-830c-381910846b6a?rid=08054280-373e-49a6-95b4-07cf5d70301e guide, 
Official guieds from hf, and qwen3 cookbok fail because this is a wsl setup, so a lot of rc version of transformers and vllm had to be instaled since main version of VLLM has some architercutral clashes with multimodal support with qwen3.5 (it does not recognize the param as valid)


Tested on 16 gb vram NVIDIA-RTX-4080 needs more testing and scripting around to see how many tokens per second

# Text completion
[17:55:09] mrdjan  $   ~/workspace/qwen_grok  vllm serve Qwen/Qwen3.5-4B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 

Call with 

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Plan a trip from Belgrade to Novi Sad by train, including costs and schedule. Think step by step."}],
    "temperature": 0.6,
    "max_tokens": 400
  }'

# Tool calling partialy working Qwen being qwen(maybe) + skill issues + wsl (maybe)
  vllm serve Qwen/Qwen3.5-4B   --dtype bfloat16   --gpu-memory-utilization 0.90   --max-model-len 32768   --reasoning-parser qwen3   --enable-auto-tool-choice   --tool-call-parser hermeus

Call with  

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {
        "role": "system",
        "content": "You are a tool-calling agent. When you decide to use a tool, respond ONLY with valid JSON in this exact format, nothing else:\n{\"tool_calls\": [{\"type\": \"function\", \"function\": {\"name\": \"tool_name\", \"arguments\": {}}}]}.\nDo NOT use XML, tags, or explanations outside JSON."
      },
      {
        "role": "user",
        "content": "Call the wazza function right now please."
      }
    ],
    "temperature": 0.1,
    "max_tokens": 150,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "wazza",
          "description": "Say wazzzaaa followed by the current unix timestamp",
          "parameters": {
            "type": "object",
            "properties": {},
            "required": []
          }
  }'"tool_choice": "auto" 

# Qwen TUI + Proxy (new)

This repo now includes a small Textual TUI and a FastAPI proxy that adds
streaming diagnostics (tokens/sec, first-token latency, total latency).

## Run the proxy

```bash
python -m qwen_app.api
```

## Run the TUI

```bash
python -m qwen_app.tui
```

## Environment variables

- `QWEN_UPSTREAM_URL` (default `http://localhost:8000`)
- `QWEN_PROXY_URL` (default `http://localhost:9000`)
- `QWEN_MODEL` (default `Qwen/Qwen3.5-4B`)
- `QWEN_HISTORY_PATH` (default `~/.qwen_tui/history.json`)
- `QWEN_TIMEOUT` (default `120`)
- `QWEN_TEMPERATURE` (default `0.7`)
- `QWEN_MAX_TOKENS` (default `512`)

## Smoke tests

Streaming proxy test:

```bash
chmod +x smoke_test_stream.sh
./smoke_test_stream.sh "What is your name?"
```

Non-streaming proxy test:

```bash
chmod +x smoke_test_nonstream.sh
./smoke_test_nonstream.sh "What is your name?"
```

# TODO
Remove the thinkking proces, on expand
Make the text selectable
Add tool supoport for what is your name where your name is hardocded as "Bob"
