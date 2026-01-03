import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from wordlist_generation.api.routers.models import ChatCompletionRequest
from wordlist_generation.api.dependencies import verify_token
from wordlist_generation.inference.runner import (
    prepare_messages,
    build_chat_inputs,
    build_prefix_fn,
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

    prefix_fn = build_prefix_fn(
        tokenizer=tokenizer,
        wordlist_dir=settings.WORDLIST_DIR,
        vocab_lang=req.vocab_lang,
        vocab_n_words=req.vocab_n_words,
    )
    if (req.vocab_lang and req.vocab_n_words) and prefix_fn is None:
        raise HTTPException(
            status_code=500,
            detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.",
        )

    stop_ids = get_stop_ids(tokenizer)
    gen_kwargs, max_new_tokens = build_generation_kwargs(
        tokenizer=tokenizer,
        allowed_max_new_tokens=settings.ALLOWED_MAX_NEW_TOKENS,
        requested_max_tokens=req.max_tokens,
        stop_ids=stop_ids,
        num_beams=req.num_beams,
        length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        prefix_fn=prefix_fn,
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
