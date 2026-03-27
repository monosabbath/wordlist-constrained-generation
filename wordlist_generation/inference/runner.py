from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from wordlist_generation.inference.generation import (
    extract_and_reorder_messages,
    normalize_max_new_tokens,
    getgen_kwargs,
    decode_generated_text,
)
from wordlist_generation.inference.vocab_constraints.constraints import (
    build_regexp_prefix_fn,
)
from wordlist_generation.inference.vocab_constraints.logits_processor import (
    PresencePenaltyLogitsProcessor,
    TieredSoftPrefixConstraintLogitsProcessor,
)


def build_chat_inputs(
    *,
    tokenizer,
    messages: List[Any] | List[List[Dict[str, str]]],
    max_input_tokens: int,
    device,
    enable_thinking: bool = False,
) -> Tuple[Any, int]:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        enable_thinking=enable_thinking,
    ).to(device)
    input_len = int(inputs["input_ids"].shape[1])
    return inputs, input_len


def generate_sequences(*, model_service, inputs, gen_kwargs) -> Any:
    with model_service.gpu_gate:
        with torch.inference_mode():
            return model_service.model.generate(**inputs, **gen_kwargs)


def unwrap_generated_sequences(outputs) -> Any:
    return outputs.sequences if hasattr(outputs, "sequences") else outputs


def build_prefix_fn(
    *,
    tokenizer,
    wordlist_dir: str,
    vocab_lang: Optional[str],
    vocab_n_words: Optional[int],
):
    if not vocab_lang or not vocab_n_words:
        return None
    return build_regexp_prefix_fn(
        tokenizer=tokenizer,
        lang=vocab_lang,
        n_words=vocab_n_words,
        wordlist_dir=wordlist_dir,
    )


def build_generation_kwargs(
    *,
    tokenizer,
    allowed_max_new_tokens: Optional[tuple[int, ...]],
    requested_max_tokens: Optional[int],
    stop_ids: List[int],
    num_beams: Optional[int],
    length_penalty: Optional[float],
    prefix_fn,
    logits_processor,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
) -> Tuple[Dict[str, Any], int]:
    max_new_tokens = normalize_max_new_tokens(requested_max_tokens, allowed_max_new_tokens)
    gen_kwargs = getgen_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        num_beams=num_beams,
        length_penalty=length_penalty if length_penalty is not None else 1.0,
        prefix_fn=prefix_fn,
        logits_processor=logits_processor,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return gen_kwargs, max_new_tokens


def build_vocab_tiered_soft_constraint_logits_processor(
    *,
    prefix_fn_n,
    prefix_fn_kn,
    penalty_m: float | None,
    penalty_n: float | None,
):
    if prefix_fn_n is None or prefix_fn_kn is None:
        return None
    if penalty_m is None or penalty_n is None:
        return None
    m = float(penalty_m)
    n = float(penalty_n)
    if m < 0 or n <= 0 or n < m:
        return None
    return [
        TieredSoftPrefixConstraintLogitsProcessor(
            prefix_allowed_tokens_fn_n=prefix_fn_n,
            prefix_allowed_tokens_fn_kn=prefix_fn_kn,
            penalty_m=m,
            penalty_n=n,
        )
    ]


def build_soft_constraint_setup(
    *,
    tokenizer,
    settings,
    vocab_lang: str | None,
    vocab_n_words: int | None,
    prefix_fn_n,
    vocab_constraint_mode: str | None,
    vocab_soft_tier2_max_rank_multiplier: float | None,
    vocab_soft_tier2_penalty: float | None,
    vocab_soft_tier3_penalty: float | None,
) -> Tuple[Any, Any]:
    """Resolve constraint mode and build soft-constraint logits processor if needed.

    Returns (prefix_for_generate, vocab_logits_processor).
    - hard mode: returns (prefix_fn_n, None)
    - soft mode: returns (None, [TieredSoftPrefixConstraintLogitsProcessor])

    Raises ValueError on invalid parameters.
    """
    import math

    mode = str(vocab_constraint_mode or settings.VOCAB_CONSTRAINT_MODE or "hard").strip().lower()
    if mode not in ("hard", "soft"):
        raise ValueError("vocab_constraint_mode must be 'hard' or 'soft'.")

    vocab_logits_processor = None
    prefix_for_generate = prefix_fn_n

    if (vocab_lang and vocab_n_words) and mode == "soft":
        k = (
            vocab_soft_tier2_max_rank_multiplier
            if vocab_soft_tier2_max_rank_multiplier is not None
            else settings.VOCAB_SOFT_TIER2_MAX_RANK_MULTIPLIER
        )
        m = (
            vocab_soft_tier2_penalty
            if vocab_soft_tier2_penalty is not None
            else settings.VOCAB_SOFT_TIER2_PENALTY
        )
        n = (
            vocab_soft_tier3_penalty
            if vocab_soft_tier3_penalty is not None
            else settings.VOCAB_SOFT_TIER3_PENALTY
        )
        if float(k) < 1:
            raise ValueError("vocab_soft_tier2_max_rank_multiplier must be >= 1.")
        if float(m) < 0 or float(n) <= 0 or float(n) < float(m):
            raise ValueError(
                "Require 0 <= vocab_soft_tier2_penalty <= vocab_soft_tier3_penalty, and vocab_soft_tier3_penalty > 0."
            )

        n_words = int(vocab_n_words or 0)
        kn_words = max(n_words, int(math.ceil(float(k) * n_words)))
        prefix_fn_kn = build_prefix_fn(
            tokenizer=tokenizer,
            wordlist_dir=settings.WORDLIST_DIR,
            vocab_lang=vocab_lang,
            vocab_n_words=kn_words,
        )
        if not prefix_fn_kn:
            raise ValueError(
                f"Constrained vocabulary config failed for lang '{vocab_lang}' at kN={kn_words}."
            )
        vocab_logits_processor = build_vocab_tiered_soft_constraint_logits_processor(
            prefix_fn_n=prefix_fn_n,
            prefix_fn_kn=prefix_fn_kn,
            penalty_m=float(m),
            penalty_n=float(n),
        )
        prefix_for_generate = None

    return prefix_for_generate, vocab_logits_processor


def build_presence_penalty_processor(
    *,
    presence_penalty: float | None,
    prompt_len: int,
) -> PresencePenaltyLogitsProcessor | None:
    if presence_penalty is None or presence_penalty == 0.0:
        return None
    return PresencePenaltyLogitsProcessor(penalty=presence_penalty, prompt_len=prompt_len)


def decode_sequences(
    *,
    tokenizer,
    generated_sequences,
    input_len: int,
    stop_ids: List[int],
) -> List[str]:
    texts: List[str] = []
    for seq in generated_sequences:
        texts.append(decode_generated_text(tokenizer, seq[input_len:], stop_ids=stop_ids))
    return texts


def prepare_messages(messages: List[Any]) -> List[Dict[str, str]]:
    return extract_and_reorder_messages(messages)
