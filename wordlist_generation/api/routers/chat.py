import time
import uuid

import math

from fastapi import APIRouter, Depends, HTTPException, Request

from wordlist_generation.api.routers.models import ChatCompletionRequest
from wordlist_generation.api.dependencies import verify_token
from wordlist_generation.inference.runner import (
    prepare_messages,
    build_chat_inputs,
    build_prefix_fn,
    build_vocab_tiered_soft_constraint_logits_processor,
    build_generation_kwargs,
    generate_sequences,
    unwrap_generated_sequences,
)
from wordlist_generation.inference.vocab_constraints.constraints import get_stop_ids
from wordlist_generation.inference.generation import decode_generated_text

router = APIRouter(prefix="/v1", tags=["chat"])


@router.get("/models")
def list_models(request: Request, auth_ok: bool = Depends(verify_token)):
    settings = request.app.state.settings
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_NAME,
                "object": "model",
                "owned_by": "owner",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request, auth_ok: bool = Depends(verify_token)):
    settings = request.app.state.settings
    ms = request.app.state.model_service
    tokenizer = ms.tokenizer
    model = ms.model

    messages = prepare_messages(req.messages)
    inputs, input_len = build_chat_inputs(
        tokenizer=tokenizer,
        messages=messages,
        max_input_tokens=settings.MAX_INPUT_TOKENS,
        device=model.device,
    )

    prefix_fn_n = build_prefix_fn(
        tokenizer=tokenizer,
        wordlist_dir=settings.WORDLIST_DIR,
        vocab_lang=req.vocab_lang,
        vocab_n_words=req.vocab_n_words,
    )
    if (req.vocab_lang and req.vocab_n_words) and prefix_fn_n is None:
        raise HTTPException(
            status_code=500,
            detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.",
        )

    mode = (req.vocab_constraint_mode or settings.VOCAB_CONSTRAINT_MODE or "hard").strip().lower()
    if mode not in ("hard", "soft"):
        raise HTTPException(status_code=400, detail="vocab_constraint_mode must be 'hard' or 'soft'.")

    vocab_logits_processor = None
    prefix_for_generate = prefix_fn_n
    if (req.vocab_lang and req.vocab_n_words) and mode == "soft":
        k = (
            req.vocab_soft_tier2_max_rank_multiplier
            if req.vocab_soft_tier2_max_rank_multiplier is not None
            else settings.VOCAB_SOFT_TIER2_MAX_RANK_MULTIPLIER
        )
        m = req.vocab_soft_tier2_penalty if req.vocab_soft_tier2_penalty is not None else settings.VOCAB_SOFT_TIER2_PENALTY
        n = req.vocab_soft_tier3_penalty if req.vocab_soft_tier3_penalty is not None else settings.VOCAB_SOFT_TIER3_PENALTY

        if float(k) < 1:
            raise HTTPException(status_code=400, detail="vocab_soft_tier2_max_rank_multiplier must be >= 1.")
        if float(m) < 0 or float(n) <= 0 or float(n) < float(m):
            raise HTTPException(status_code=400, detail="Require 0 <= vocab_soft_tier2_penalty <= vocab_soft_tier3_penalty, and vocab_soft_tier3_penalty > 0.")

        n_words = int(req.vocab_n_words or 0)
        kn_words = max(n_words, int(math.ceil(float(k) * n_words)))

        prefix_fn_kn = build_prefix_fn(
            tokenizer=tokenizer,
            wordlist_dir=settings.WORDLIST_DIR,
            vocab_lang=req.vocab_lang,
            vocab_n_words=kn_words,
        )
        if prefix_fn_kn is None:
            raise HTTPException(
                status_code=500,
                detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}' at kN={kn_words}.",
            )

        vocab_logits_processor = build_vocab_tiered_soft_constraint_logits_processor(
            prefix_fn_n=prefix_fn_n,
            prefix_fn_kn=prefix_fn_kn,
            penalty_m=float(m),
            penalty_n=float(n),
        )
        prefix_for_generate = None

    stop_ids = get_stop_ids(tokenizer)
    gen_kwargs, max_new_tokens = build_generation_kwargs(
        tokenizer=tokenizer,
        allowed_max_new_tokens=settings.ALLOWED_MAX_NEW_TOKENS,
        requested_max_tokens=req.max_tokens,
        stop_ids=stop_ids,
        num_beams=req.num_beams,
        length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        prefix_fn=prefix_for_generate,
        logits_processor=vocab_logits_processor,
        # Sampling params
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    # Ensure we use high-performance SDPA for generation
    # cache_implementation="paged" is removed because it forces continuous batching
    # which doesn't support beam search or prefix_allowed_tokens_fn in v5 yet.

    outputs = generate_sequences(model_service=ms, inputs=inputs, gen_kwargs=gen_kwargs)
    generated_sequences = unwrap_generated_sequences(outputs)
    text = decode_generated_text(tokenizer, generated_sequences[0][input_len:], stop_ids=stop_ids)
    
    prompt_tokens = int(input_len)
    completion_tokens = int(generated_sequences[0].shape[0] - input_len)
    created = int(time.time())

    # Determine finish reason
    last_token = int(generated_sequences[0][-1].item())
    finish_reason = "stop" if stop_ids and (last_token in set(stop_ids)) else ("length" if completion_tokens >= max_new_tokens else "stop")

    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": req.model or settings.MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp
