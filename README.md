# AttentionAtlas

Visualize transformer attention patterns in real time using vLLM.

## Quickstart

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A C compiler (Xcode Command Line Tools on macOS: `xcode-select --install`)

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to set your Hugging Face token and model:

```
HF_TOKEN = "your_hf_token_here"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
COMPARE_MODEL = ""
PORT = "8003"
```

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face API token (required for gated models like Mistral) |
| `MODEL_NAME` | Primary model for inference and attention capture |
| `COMPARE_MODEL` | Optional second model for side-by-side comparison. Leave empty (`""`) on macOS — loading two vLLM engines in one process deadlocks with `VLLM_ENABLE_V1_MULTIPROCESSING=0`. |
| `PORT` | Server port (default `8000`) |

### 2. Start the API server

```bash
uv run api_vllm.py
```

First run installs all dependencies (including building vLLM from source on macOS, which takes a few minutes). Subsequent runs start in ~10 seconds.

### 3. Open the frontend

Open `frontend/index.html` in a browser. The API config is auto-generated at `frontend/js/config.js` on server startup.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health message |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List loaded models |
| `POST` | `/generate` | Generate text and capture attention |
| `POST` | `/analyze` | Analyze text and return attention patterns |
| `POST` | `/compare` | Analyze with the comparison model |
| `POST` | `/extract_pdf` | Extract text from an uploaded PDF |

### Generate

```bash
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is attention?", "max_tokens": 50}'
```

### Analyze

```bash
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"answer": "The cat sat on the mat"}'
```

Returns attention patterns as `[num_layers, num_heads, seq_len, seq_len]`.

## Platform Notes

### macOS (Apple Silicon / Intel)

vLLM builds from source with a CPU-only backend. The `enforce_eager=True` and `TORCH_COMPILE_DISABLE=1` flags in `api_vllm.py` bypass torch inductor compilation, avoiding C++ toolchain issues.

### Linux (with CUDA)

Works out of the box with GPU acceleration. You can set `COMPARE_MODEL` to load a second model for comparison.
