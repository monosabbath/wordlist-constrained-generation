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
    TieredSoftPrefixConstraintLogitsProcessor,
)


def build_chat_inputs(
    *,
    tokenizer,
    messages: List[Any] | List[List[Dict[str, str]]],
    max_input_tokens: int,
    device,
) -> Tuple[Any, int]:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
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
