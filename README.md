# Wordlist‑Constrained Generation (Transformers v5)

FastAPI server for wordlist‑constrained text generation using Hugging Face **Transformers v5** + **lm‑format‑enforcer**.

## Run

```bash
uv run uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010 --workers 1
```

## Chat API

`POST /v1/chat/completions` (OpenAI‑compatible)

```bash
curl -H "Authorization: Bearer $SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write 3 lines, each on a new line."}
    ],
    "max_tokens": 128
  }' \
  http://127.0.0.1:8010/v1/chat/completions
```

Optional constrained vocab:

```json
{"vocab_lang":"es","vocab_n_words":3000}
```

## Batch API

1) Upload a JSON file containing a list of chat completion requests:

```bash
curl -X POST "http://127.0.0.1:8010/v1/batch/jobs?max_tokens=128&num_beams=4" \
  -H "Authorization: Bearer $SECRET_TOKEN" \
  -F "file=@/path/to/my_requests.json"
```

2) Poll status:

```bash
curl -H "Authorization: Bearer $SECRET_TOKEN" \
  http://127.0.0.1:8010/v1/batch/jobs/<job_id>
```

3) Download results:

```bash
curl -L -H "Authorization: Bearer $SECRET_TOKEN" \
  -o results.json \
  http://127.0.0.1:8010/v1/batch/jobs/<job_id>/results
```

The file is an array of OpenAI‑shaped chat completion objects (token counts are `null` in batch):

```json
[
  {
    "id": "chatcmpl-batch-<job_id>-0",
    "object": "chat.completion",
    "created": 1730000123,
    "model": "google/gemma-3-27b-it",
    "choices": [
      {
        "index": 0,
        "message": { "role": "assistant", "content": "..." },
        "finish_reason": "stop"
      }
    ],
    "usage": { "prompt_tokens": null, "completion_tokens": null, "total_tokens": null }
  },
  { "... second item ..." }
]
```

Implementation details:
- Uploaded files are stored in `BATCH_JOB_TEMP_DIR` (defaults to OS temp) and removed after processing.
- The output file remains available for download after completion (path is tracked in memory).
- `BATCH_JOB_PIPELINE_SIZE` controls how many prompts the pipeline feeds per forward pass.
