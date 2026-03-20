# Agent Instructions

## What This Is

GPU backend server for the language-app content generation pipeline. Paired with the `generation-pipeline` repo (`~/Projects/language-app/generation-pipeline/`) which handles batch orchestration and Supabase ingestion.
The git repo is at `https://github.com/monosabbath/wordlist-constrained-generation`

## How to Run

- Use `uv` for dependency management in both repos
- Start server: `uv run uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010`
- Configuration is via `.env` file — see `settings.py` for all options

## Batch Generation Workflow

1. Deploy this server on a remote GPU instance (vast.ai). User handles instance creation; agent manages setup and operation via SSH.
2. The server always runs on port 8010 internally. On vast.ai, expose port 8010 and use the mapped external port in the URL (e.g., `http://<ip>:<external-port>` where the external port maps to 8010/tcp).
3. Pass the server URL as an inline env var (do **not** edit `.env` — multiple agents may target different instances concurrently):
   `GENERATION_SERVER_URL=http://<ip>:<external-port> uv run python -m pipeline.cli generate <config_id> --num-items <N>`
4. The pipeline materializes items from Supabase, renders prompts, submits a batch to this server, polls until done, then ingests results back to Supabase.

## Pipeline Defaults

- Always use the provider batch API (`--batch`, which is the default) for evaluate/process/translate steps unless the user explicitly asks for sequential. Batch API is faster overall and 50% cheaper — it processes immediately despite the "up to 24h" docs.

## vast.ai SSH Notes

- **SSH multiplexing** is configured in `~/.ssh/config` with a `Host vast-*` wildcard block that provides `ControlMaster auto`, `ControlPersist 4h`, keepalives, and suppresses banners. Multiple instances can run concurrently with different aliases.
- **When the user provides SSH details for a new instance**, add a `Host` entry to `~/.ssh/config` under the active instances section with a descriptive name (e.g., `vast-gemma27b`, `vast-qwen32b`). Then establish the master connection with `ssh -fN vast-<name>`. Use `ssh vast-<name> '<command>'` for all subsequent remote commands.
  ```
  # Example entry to append to ~/.ssh/config
  Host vast-qwen32b
      HostName <ip>
      Port <ssh-port>
  ```
- **When done with an instance**, close the master connection with `ssh -O exit vast-<name>` and remove the entry from `~/.ssh/config`.
- `pkill` returning exit code 1 (no matching process) causes SSH to report exit 255 since it forwards the remote exit code. Always append `; true` or `2>/dev/null; true` to commands that may legitimately fail.
- The HF model cache location depends on the `HF_HOME` env var. On vast.ai the shell default (`$HF_HOME` from the container) may differ from the `.env` file value. Check **both** the shell env (`echo $HF_HOME`) and the `.env` `HF_HOME=` line. The actual downloaded models live at `$HF_HOME/hub/`.
- When switching models, delete the old model cache first — vast.ai root overlay disk is typically small (70GB) and fills fast.
- Use `nohup ... &` to start the server in background over SSH, and write logs to a file (e.g. `/workspace/server.log`) for later inspection.

## BATCH_JOB_PIPELINE_SIZE Tuning (gemma3 27b, RTX Pro 6000 96GB)

The model uses ~54GB VRAM (bf16), leaving ~42GB for KV cache. The key constraint is **effective batch = pipeline_size × num_beams** — this determines peak KV cache memory. KV cache also scales with total sequence length (input + output tokens).

Actual output length is driven by `min_turns`/`max_turns` in the generation config, not `max_tokens` (which is just a ceiling — ensure it is set high enough to avoid truncating high-turn configs). With current configs (`max_input_tokens=512`), observed p95 output tokens by turn range:
- 4-8 turns: ~215 tokens
- 6-10 turns: ~240 tokens
- 8-12 turns: ~250 tokens (saturates max_tokens cap)

Recommended `BATCH_JOB_PIPELINE_SIZE` values (empirically derived, max_tokens=250):

| num_beams | 4-8 turns | 6-10 turns | 8-12 turns |
|-----------|-----------|------------|------------|
| 10        | 12        | 10         | 10         |
| 15        | 8         | 8          | 7          |
| 20        | 6         | 5          | 4          |
| 25        | 4         | 4          | 3          |

When `max_tokens` is raised for higher-turn configs (it should scale with turn count to avoid truncation), reduce pipeline_size proportionally. If OOM occurs, halve the pipeline_size and restart the server. The setting is in the server's `.env` file and requires a server restart to take effect.

## Related Repos

- `~/Projects/language-app/generation-pipeline/` — batch orchestration CLI
- `~/Projects/language-app/supabase-backend/` — database schema (cloud Supabase is source of truth)
