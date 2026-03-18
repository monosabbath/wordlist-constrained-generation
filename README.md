# Wordlist-Constrained Generation

FastAPI server for vocabulary-constrained text generation using HuggingFace Transformers v5 + lm-format-enforcer. Designed to run on remote GPU instances as the generation backend for the language-app pipeline.

## Quick Start

```bash
# Install dependencies
uv sync

# Configure
cat > .env << 'EOF'
MODEL_NAME=Qwen/Qwen3.5-27B
HF_TOKEN=your_token_here
BATCH_JOB_PIPELINE_SIZE=16
EOF

# Run
uv run uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010
```

## Deployment on vast.ai

1. Create an instance using the **NVIDIA CUDA** template with port **8010** exposed
2. Add your SSH public key to vast.ai
3. SSH in and run the setup:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone https://github.com/monosabbath/wordlist-constrained-generation
cd wordlist-constrained-generation
# Create .env as above
uv sync
uv run uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010
```

4. Point the generation-pipeline at the server by setting `GENERATION_SERVER_URL` in its `.env`

## API

### Chat Completions

`POST /v1/chat/completions` (OpenAI-compatible)

```bash
curl -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Escribe una oración simple."}],
    "max_tokens": 128,
    "vocab_lang": "es",
    "vocab_n_words": 500,
    "num_beams": 5
  }' \
  http://localhost:8010/v1/chat/completions
```

### Batch API

**Submit** — upload a JSON file (array of chat completion requests) with generation params as query parameters:

```bash
curl -X POST "http://localhost:8010/v1/batch/jobs?token=changeme&max_tokens=250&num_beams=5&vocab_lang=es&vocab_n_words=500&vocab_constraint_mode=hard" \
  -F "file=@batch.json"
```

**Poll status:**
```bash
curl "http://localhost:8010/v1/batch/jobs/<job_id>?token=changeme"
```

**Download results:**
```bash
curl -o results.json "http://localhost:8010/v1/batch/jobs/<job_id>/results?token=changeme"
```

## Vocabulary Constraint Modes

- **hard** — lm-format-enforcer blocks all tokens outside the allowed wordlist. Guarantees output stays within vocabulary.
- **soft** — tiered penalty logits processor. Penalizes out-of-vocabulary tokens but doesn't block them entirely. Allows more natural output at the cost of occasional out-of-vocab words.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | *(required)* | HuggingFace model ID (e.g., `Qwen/Qwen3.5-27B`) |
| `HF_TOKEN` | | HuggingFace token for gated/faster downloads |
| `BATCH_JOB_PIPELINE_SIZE` | `8` | Prompts per GPU forward pass |
| `VOCAB_CONSTRAINT_MODE` | `hard` | Default: `hard` or `soft` |
| `ENABLE_THINKING` | `false` | Enable model reasoning/thinking tokens |
| `SECRET_TOKEN` | `changeme` | API authentication token |
| `DEVICE_MAP` | `auto` | PyTorch device mapping |
| `DTYPE` | `auto` | Model dtype (`auto`, `bf16`, `fp16`) |
| `ALLOWED_MAX_NEW_TOKENS` | `64,128,256,512` | Allowed max token buckets |
| `MAX_INPUT_TOKENS` | `512` | Max input token length |

## GPU Memory (96GB VRAM with 27B model)

| Batch Size | Beams | Peak VRAM | Status |
|---|---|---|---|
| 8 | 5 | ~67 GB | Safe |
| 16 | 5 | ~80 GB | Recommended |
| 20+ | 5 | ~90+ GB | Risk of OOM |

## Project Structure

```
wordlist_generation/
  main.py              # FastAPI app
  settings.py          # Environment config
  model_service.py     # Model loading (CausalLM + multimodal fallback)
  batch_processor.py   # Batch job processing
  api/routers/
    chat.py            # /v1/chat/completions
    batch.py           # /v1/batch/jobs
  inference/
    runner.py          # Generation orchestration
    generation.py      # Token decoding, gen kwargs
    vocab_constraints/ # Wordlist constraint logic
wordlists/             # Frequency-ranked word lists by language
third_party/           # Vendored group beam search
```
