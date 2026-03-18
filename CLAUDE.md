# Agent Instructions

## What This Is

GPU backend server for the language-app content generation pipeline. Paired with the `generation-pipeline` repo (`~/Projects/language-app/generation-pipeline/`) which handles batch orchestration and Supabase ingestion.

## How to Run

- Use `uv` for dependency management in both repos
- Start server: `uv run uvicorn wordlist_generation.main:app --host 0.0.0.0 --port 8010`
- Configuration is via `.env` file — see `settings.py` for all options

## Batch Generation Workflow

1. Deploy this server on a remote GPU instance (vast.ai). User handles instance creation; agent manages setup and operation via SSH.
2. Set `GENERATION_SERVER_URL` in the generation-pipeline's `.env` to point at the remote instance.
3. Run generation from the pipeline repo: `uv run python -m pipeline.cli generate <config_id> --num-items <N>`
4. The pipeline materializes items from Supabase, renders prompts, submits a batch to this server, polls until done, then ingests results back to Supabase.

## Related Repos

- `~/Projects/language-app/generation-pipeline/` — batch orchestration CLI
- `~/Projects/language-app/supabase-backend/` — database schema (cloud Supabase is source of truth)
