from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


class GPUConcurrencyGate:
    """Process-local concurrency gate to avoid overlapping GPU-bound generation."""

    def __init__(self, max_concurrency: int = 1):
        import threading

        self._sem = threading.Semaphore(max_concurrency)

    def __enter__(self):
        self._sem.acquire()

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()


class _NoOpGate:
    """No-op context manager replacing GPUConcurrencyGate in distributed mode."""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass


class ModelService:
    def __init__(self, model, tokenizer, settings, coordinator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.coordinator = coordinator

        if coordinator and coordinator.is_distributed:
            # In distributed mode, generation is serialized via the coordinator
            # broadcast mechanism — no need for a thread-level semaphore.
            self.gpu_gate = _NoOpGate()
        else:
            self.gpu_gate = GPUConcurrencyGate(
                max_concurrency=int(
                    getattr(settings, "GENERATION_MAX_CONCURRENCY", 1)
                )
            )

    @classmethod
    def from_settings(cls, s, coordinator=None):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        dtype: Any = "auto"
        ts = str(s.DTYPE).lower()
        if ts in ("bf16", "bfloat16", "torch.bfloat16"):
            dtype = torch.bfloat16
        elif ts in ("fp16", "float16", "torch.float16"):
            dtype = torch.float16

        init_kwargs: Dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
            "attn_implementation": "sdpa",
        }
        if dtype != "auto":
            init_kwargs["dtype"] = dtype

        use_ep = getattr(s, "EXPERT_PARALLEL", False) and coordinator and coordinator.is_distributed

        if use_ep:
            from transformers.distributed.configuration_utils import DistributedConfig

            init_kwargs["distributed_config"] = DistributedConfig(
                enable_expert_parallel=True
            )
            experts_impl = getattr(s, "EXPERTS_IMPLEMENTATION", "eager")
            init_kwargs["experts_implementation"] = experts_impl
            # Expert parallelism assigns devices via torchrun LOCAL_RANK;
            # do not use accelerate device_map.
        else:
            init_kwargs["device_map"] = s.DEVICE_MAP

        try:
            model = AutoModelForCausalLM.from_pretrained(s.MODEL_NAME, **init_kwargs)
        except (ValueError, AttributeError):
            model = AutoModelForImageTextToText.from_pretrained(s.MODEL_NAME, **init_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
        )
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()
        return cls(model=model, tokenizer=tokenizer, settings=s, coordinator=coordinator)
