# Agent Instructions

## What This Is

GPU backend server for the language-app content generation pipeline. Paired with the `generation-pipeline` repo (`~/Projects/language-app/generation-pipeline/`) which handles batch orchestration and Supabase ingestion.

## How to Run

- Use `uv` for dependency management in both repos
- Start server: `uv run uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010`
- Configuration is via `.env` file — see `settings.py` for all options

## Batch Generation Workflow

1. Deploy this server on a remote GPU instance (vast.ai). User handles instance creation; agent manages setup and operation via SSH.
2. The server always runs on port 8010 internally. On vast.ai, expose port 8010 and use the mapped external port in the URL (e.g., `http://<ip>:<external-port>` where the external port maps to 8010/tcp).
3. Set `GENERATION_SERVER_URL` in the generation-pipeline's `.env` to point at the remote instance's external URL.
3. Run generation from the pipeline repo: `uv run python -m pipeline.cli generate <config_id> --num-items <N>`
4. The pipeline materializes items from Supabase, renders prompts, submits a batch to this server, polls until done, then ingests results back to Supabase.

## Pipeline Defaults

- Always use the provider batch API (`--batch`, which is the default) for evaluate/process/translate steps unless the user explicitly asks for sequential. Batch API is faster overall and 50% cheaper — it processes immediately despite the "up to 24h" docs.

## Related Repos

- `~/Projects/language-app/generation-pipeline/` — batch orchestration CLI
- `~/Projects/language-app/supabase-backend/` — database schema (cloud Supabase is source of truth)
