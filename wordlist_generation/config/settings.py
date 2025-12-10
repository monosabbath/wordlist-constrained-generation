import os
import tempfile
from typing import Tuple
from dotenv import load_dotenv

# Load .env as early as possible
load_dotenv()


def _parse_int_tuple(values: str) -> Tuple[int, ...]:
    try:
        return tuple(sorted({int(x.strip()) for x in values.split(",") if x.strip()}))
    except Exception:
        return (64, 128, 256, 512)


class Settings:
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Model/runtime config
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "auto")
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
    SECRET_TOKEN: str = os.getenv("SECRET_TOKEN", "changeme")
    TOKENIZER_PADDING_SIDE: str = os.getenv("TOKENIZER_PADDING_SIDE", "left").strip()
    PAD_TO_MULTIPLE_OF: int = int(os.getenv("PAD_TO_MULTIPLE_OF", "64"))
    MAX_INPUT_TOKENS: int = int(os.getenv("MAX_INPUT_TOKENS", "512"))
    ALLOWED_MAX_NEW_TOKENS: Tuple[int, ...] = _parse_int_tuple(
        os.getenv("ALLOWED_MAX_NEW_TOKENS", "64,128,256,512")
    )

    # Constrained vocab
    WORDLIST_DIR: str = os.getenv("WORDLIST_DIR", "wordlists")

    # Torch dtype
    TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "auto")

    # Batch jobs
    BATCH_JOB_TEMP_DIR: str = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
    BATCH_JOB_PIPELINE_SIZE: int = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))

    # In-process GPU generation concurrency (1 = fully serialized)
    GENERATION_MAX_CONCURRENCY: int = int(os.getenv("GENERATION_MAX_CONCURRENCY", "1"))

    # Fused MoE: Use Triton kernels for Qwen3 MoE models
    USE_FUSED_MOE: bool = os.getenv("USE_FUSED_MOE", "false").lower() == "true"
