from typing import Any, Dict, List, Optional
import torch


# --- Helper functions for message processing ---

def extract_and_reorder_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Extract system prompt and reorder messages with system first."""
    system_prompt = ""
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role")
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        if role == "system":
            system_prompt = content
            break

    result: List[Dict[str, str]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role")
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        if role != "system":
            result.append({"role": role, "content": content})
    return result


def normalize_max_new_tokens(requested: Optional[int], allowed: Optional[tuple[int, ...]]) -> int:
    if not allowed or len(allowed) == 0:
        cap = 512
        target = int(requested) if requested is not None else cap
        return min(target, cap)
    cap = allowed[-1]
    target = int(requested) if requested is not None else cap
    target = min(target, cap)
    for a in allowed:
        if target <= a:
            return a
    return allowed[-1]


def getgen_kwargs(
    tokenizer,
    max_new_tokens: int,
    stop_ids: Optional[list[int]],
    num_beams: Optional[int] = None,
    length_penalty: float = 1.0,
    prefix_fn=None,
    # Sampling parameters
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
):
    t = float(temperature if temperature is not None else 1.0)
    tp = float(top_p if top_p is not None else 1.0)
    tk = int(top_k if top_k is not None else 50)
    rp = float(repetition_penalty if repetition_penalty is not None else 1.0)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True if t > 0 else False,
        num_beams=int(num_beams or 1),
        length_penalty=float(length_penalty),
        temperature=t if t > 0 else 1.0,
        top_p=tp,
        top_k=tk,
        repetition_penalty=rp,
    )
    if prefix_fn:
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        else:
            gen_kwargs["pad_token_id"] = stop_ids[0] if isinstance(stop_ids, list) and stop_ids else None
    return gen_kwargs
