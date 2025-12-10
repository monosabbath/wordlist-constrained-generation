from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
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


@lru_cache(maxsize=16)
def build_template_kwargs_for_model(model_name: str) -> Dict[str, Any]:
    """Build kwargs for tokenizer.apply_chat_template()."""
    return {"add_generation_prompt": True, "tokenize": False}


# --- Input processing functions ---

def move_inputs_to_correct_device(inputs: Dict[str, torch.Tensor], model):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k: v.to(device) for k, v in inputs.items()}


def tokenizer_encode_for_chat(tokenizer, texts: Any, pad_to_multiple_of: int, max_input_tokens: int) -> Dict[str, torch.Tensor]:
    tok_kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "pad_to_multiple_of": pad_to_multiple_of,
        "max_length": max_input_tokens,
        "return_token_type_ids": False,
    }
    return tokenizer(texts, **tok_kwargs)


def strip_unused_model_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in inputs.items() if k != "token_type_ids"}


def normalize_max_new_tokens(requested: Optional[int], allowed: Optional[tuple[int, ...]]) -> int:
    # Use the configured allowed set to determine the cap; fall back to 512 if not provided.
    if not allowed or len(allowed) == 0:
        cap = 512
        target = int(requested) if requested is not None else cap
        return min(target, cap)
    # allowed is sorted in Settings; use its max as cap
    cap = allowed[-1]
    target = int(requested) if requested is not None else cap
    target = min(target, cap)
    for a in allowed:
        if target <= a:
            return a
    return allowed[-1]


def truncate_inputs_to_max_length(inputs: Dict[str, torch.Tensor], max_length: int) -> Dict[str, torch.Tensor]:
    """Truncate input tensors to max_length (keeps last tokens)."""
    input_ids = inputs["input_ids"]
    seq_len = input_ids.size(-1)
    if seq_len <= max_length:
        return inputs

    # Use [..., -max_length:] which works for both 1D and 2D tensors
    out = {"input_ids": input_ids[..., -max_length:]}
    if "attention_mask" in inputs:
        out["attention_mask"] = inputs["attention_mask"][..., -max_length:]
    # Copy any other keys unchanged
    for k, v in inputs.items():
        if k not in out:
            out[k] = v
    return out


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
    # Stochastic beam search parameter
    beam_pruning_temperature: Optional[float] = None,
):
    # Defaults for sampling params when not provided
    t = float(temperature if temperature is not None else 1.0)
    tp = float(top_p if top_p is not None else 1.0)
    tk = int(top_k if top_k is not None else 50)
    rp = float(repetition_penalty if repetition_penalty is not None else 1.0)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        # Enable sampling together with beam search
        do_sample=True,
        num_beams=int(num_beams or 10),
        length_penalty=float(length_penalty),
        temperature=t,
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
    if beam_pruning_temperature is not None:
        gen_kwargs["beam_pruning_temperature"] = float(beam_pruning_temperature)
    return gen_kwargs
