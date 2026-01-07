from typing import Optional, List, Union
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    
    # OpenAI compatible fields
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

    # Custom fields for this repo
    vocab_lang: Optional[str] = None
    vocab_n_words: Optional[int] = None
    vocab_constraint_mode: Optional[str] = None  # "hard" | "soft" (default: Settings.VOCAB_CONSTRAINT_MODE)
    # Rank-aware (tiered) soft penalties for vocab constraints (only used when mode == "soft"):
    # ranks <= N: no penalty
    # N < ranks <= k*N: penalty (tier2)
    # ranks > k*N: penalty (tier3)
    vocab_soft_tier2_max_rank_multiplier: Optional[float] = None
    vocab_soft_tier2_penalty: Optional[float] = None
    vocab_soft_tier3_penalty: Optional[float] = None
    num_beams: Optional[int] = 1
    length_penalty: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    top_k: Optional[int] = 500
    
    # Batch processing helper
    custom_id: Optional[str] = None
