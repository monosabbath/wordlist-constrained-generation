"""Distributed coordination for expert-parallel MoE inference.

When running with torchrun and EXPERT_PARALLEL=true, this module coordinates
generation across multiple GPU ranks. Rank 0 runs the FastAPI server and
broadcasts generation tasks to worker ranks, which participate in the
distributed forward pass (expert + tensor parallelism) and discard results.

All constraint logic (prefix_allowed_tokens_fn, logits processors) is
reconstructed independently on each rank from serializable parameters,
ensuring identical behavior across ranks.
"""

import logging
import math
import os
import pickle

import torch
import torch.distributed as dist

from wordlist_generation.inference.runner import (
    build_prefix_fn,
    build_presence_penalty_processor,
    build_vocab_tiered_soft_constraint_logits_processor,
)

logger = logging.getLogger(__name__)


class DistributedCoordinator:
    """Coordinates generation across torchrun ranks for expert parallelism."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            logger.info(
                f"Rank {self.rank}/{self.world_size} initialized on cuda:{self.local_rank}"
            )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.local_rank}")

    def broadcast_task(self, task=None):
        """Broadcast a generation task from rank 0 to all ranks.

        Rank 0: pass the task dict to broadcast, or None for shutdown.
        Worker ranks: call with no arguments to receive.

        Returns the task dict on all ranks, or None for shutdown signal.
        """
        if self.is_main:
            if task is None:
                size_tensor = torch.tensor(
                    [0], dtype=torch.long, device=self.device
                )
                dist.broadcast(size_tensor, src=0)
                return None

            cpu_task = {}
            for k, v in task.items():
                if isinstance(v, torch.Tensor):
                    cpu_task[k] = v.cpu()
                else:
                    cpu_task[k] = v
            data_bytes = pickle.dumps(cpu_task)

            size_tensor = torch.tensor(
                [len(data_bytes)], dtype=torch.long, device=self.device
            )
            dist.broadcast(size_tensor, src=0)

            data_tensor = torch.frombuffer(
                bytearray(data_bytes), dtype=torch.uint8
            ).to(self.device)
            dist.broadcast(data_tensor, src=0)

            return task
        else:
            size_tensor = torch.tensor(
                [0], dtype=torch.long, device=self.device
            )
            dist.broadcast(size_tensor, src=0)
            size = size_tensor.item()

            if size == 0:
                return None

            data_tensor = torch.empty(
                int(size), dtype=torch.uint8, device=self.device
            )
            dist.broadcast(data_tensor, src=0)

            data_bytes = bytes(data_tensor.cpu().numpy())
            return pickle.loads(data_bytes)

    def shutdown(self):
        """Send shutdown signal to workers and tear down process group."""
        if self.is_main and self.is_distributed:
            self.broadcast_task(None)
        if dist.is_initialized():
            dist.destroy_process_group()


def _build_worker_callables(constraint_config, tokenizer, settings):
    """Build prefix_fn and logits_processors on worker ranks from broadcast config."""
    if not constraint_config:
        return None, None

    cc = constraint_config
    mode = cc.get("vocab_constraint_mode", "hard")
    prefix_fn = None
    logits_processors = []

    if cc.get("vocab_lang") and cc.get("vocab_n_words"):
        prefix_fn_n = build_prefix_fn(
            tokenizer=tokenizer,
            wordlist_dir=settings.WORDLIST_DIR,
            vocab_lang=cc["vocab_lang"],
            vocab_n_words=cc["vocab_n_words"],
        )

        if mode == "hard":
            prefix_fn = prefix_fn_n
        elif mode == "soft":
            k = cc.get("vocab_soft_tier2_max_rank_multiplier", 1.0)
            n_words = cc["vocab_n_words"]
            kn_words = max(n_words, int(math.ceil(float(k) * n_words)))
            prefix_fn_kn = build_prefix_fn(
                tokenizer=tokenizer,
                wordlist_dir=settings.WORDLIST_DIR,
                vocab_lang=cc["vocab_lang"],
                vocab_n_words=kn_words,
            )
            soft_lp = build_vocab_tiered_soft_constraint_logits_processor(
                prefix_fn_n=prefix_fn_n,
                prefix_fn_kn=prefix_fn_kn,
                penalty_m=cc.get("vocab_soft_tier2_penalty", 0.0),
                penalty_n=cc.get("vocab_soft_tier3_penalty", 8.0),
            )
            if soft_lp:
                logits_processors.extend(soft_lp)

    pp = build_presence_penalty_processor(
        presence_penalty=cc.get("presence_penalty"),
        prompt_len=cc.get("prompt_len", 0),
    )
    if pp is not None:
        logits_processors.append(pp)

    return prefix_fn, logits_processors or None


def worker_loop(coordinator, model_service, settings):
    """Main loop for worker ranks (non-0). Blocks waiting for generation tasks."""
    logger.info(f"Worker rank {coordinator.rank} entering generation loop")
    tokenizer = model_service.tokenizer

    while True:
        task = coordinator.broadcast_task()
        if task is None:
            logger.info(f"Worker rank {coordinator.rank} received shutdown signal")
            break

        # Reconstruct gen_kwargs: start with serializable kwargs from rank 0
        gen_kwargs = dict(task["gen_kwargs"])

        # Build callables (prefix_fn, logits_processors) from constraint config
        prefix_fn, logits_processors = _build_worker_callables(
            task.get("constraint_config"), tokenizer, settings
        )
        if prefix_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
        if logits_processors is not None:
            gen_kwargs["logits_processor"] = logits_processors

        # Move inputs to local GPU
        inputs = {
            "input_ids": task["input_ids"].to(coordinator.device),
            "attention_mask": task["attention_mask"].to(coordinator.device),
        }

        with torch.inference_mode():
            model_service.model.generate(**inputs, **gen_kwargs)
        # Output discarded — rank 0 computes the same result
