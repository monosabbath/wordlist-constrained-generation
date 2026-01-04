from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPUConcurrencyGate:
    """Process-local concurrency gate to avoid overlapping GPU-bound generation."""

    def __init__(self, max_concurrency: int = 1):
        import threading

        self._sem = threading.Semaphore(max_concurrency)

    def __enter__(self):
        self._sem.acquire()

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()


class ModelService:
    def __init__(self, model, tokenizer, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.gpu_gate = GPUConcurrencyGate(max_concurrency=int(getattr(settings, "GENERATION_MAX_CONCURRENCY", 1)))

    @classmethod
    def from_settings(cls, s):
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
            "device_map": s.DEVICE_MAP,
            "attn_implementation": "sdpa",
        }
        if dtype != "auto":
            init_kwargs["dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(s.MODEL_NAME, **init_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False)
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()
        return cls(model=model, tokenizer=tokenizer, settings=s)
