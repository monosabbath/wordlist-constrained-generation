#!/usr/bin/env python3
"""Entry point for distributed expert-parallel inference via torchrun.

Usage:
    torchrun --nproc-per-node <NUM_GPUS> launch_distributed.py [--host HOST] [--port PORT]

Environment variables (in .env or exported):
    EXPERT_PARALLEL=true              Required
    EXPERTS_IMPLEMENTATION=eager      One of: eager, batched_mm, grouped_mm
    MODEL_NAME=MiniMaxAI/MiniMax-M2.5 The MoE model to load
    DTYPE=bf16                        Model precision
    TRUST_REMOTE_CODE=true            Required for custom model architectures

All other settings from .env are respected (WORDLIST_DIR, SECRET_TOKEN, etc.).

The number of GPUs (--nproc-per-node) must evenly divide the model's total
number of experts (e.g., 256 experts -> 1, 2, 4, 8, 16, 32, 64, 128, or 256 GPUs).

Rank 0 runs the FastAPI HTTP server.
Ranks 1-N run a generation worker loop, participating in distributed forward
passes via expert + tensor parallelism.
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rank %(process)d] %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Distributed generation server")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8000, help="Server bind port")
    args = parser.parse_args()

    from wordlist_generation.distributed import DistributedCoordinator, worker_loop
    from wordlist_generation.settings import Settings

    coordinator = DistributedCoordinator()
    settings = Settings()

    if not coordinator.is_distributed:
        logger.warning(
            "WORLD_SIZE=1: running in single-process mode. "
            "Use 'uvicorn wordlist_generation.main:app' instead for non-distributed inference."
        )

    if coordinator.is_main:
        logger.info(f"Rank 0: starting FastAPI server on {args.host}:{args.port}")

        from wordlist_generation.main import create_app
        import uvicorn

        app = create_app(coordinator=coordinator)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logger.info(f"Rank {coordinator.rank}: starting worker loop")

        from wordlist_generation.model_service import ModelService

        model_service = ModelService.from_settings(settings, coordinator=coordinator)
        worker_loop(coordinator, model_service, settings)

    logger.info(f"Rank {coordinator.rank}: exiting")


if __name__ == "__main__":
    main()
