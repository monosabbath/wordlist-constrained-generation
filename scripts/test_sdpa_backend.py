"""
Diagnose which SDPA backend PyTorch selects during generation.

Hooks into torch.nn.functional.scaled_dot_product_attention to detect
which kernel (flash, mem_efficient, math) is actually dispatched, under
conditions matching real batch beam-search generation.

Usage:
    uv run python scripts/test_sdpa_backend.py [--num-beams 10] [--batch-size 4]
"""

import argparse
import sys
from collections import Counter
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordlist_generation.settings import Settings


def parse_args():
    p = argparse.ArgumentParser(description="Test which SDPA backend is used")
    p.add_argument("--num-beams", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4, help="Number of prompts in a batch (effective = batch_size * num_beams)")
    p.add_argument("--max-new-tokens", type=int, default=16, help="Short generation to keep test fast")
    return p.parse_args()


def probe_sdpa_backends(model, inputs, gen_kwargs):
    """Run generation while probing which SDPA kernel is dispatched."""

    call_log = Counter()
    original_sdpa = F.scaled_dot_product_attention

    def logging_sdpa(*args, **kwargs):
        # Probe: disable each backend one at a time to see which one is active.
        # We do this on the first call only, then let the rest run normally.
        result = original_sdpa(*args, **kwargs)
        # Log tensor shapes for context
        q = args[0] if args else kwargs.get("query")
        k = args[1] if len(args) > 1 else kwargs.get("key")
        if q is not None and not call_log:
            print(f"  First SDPA call shapes: Q={list(q.shape)}, K={list(k.shape)}, dtype={q.dtype}")
            if "attn_mask" in kwargs and kwargs["attn_mask"] is not None:
                print(f"  attn_mask shape: {list(kwargs['attn_mask'].shape)}, dtype={kwargs['attn_mask'].dtype}")
            elif len(args) > 3 and args[3] is not None:
                print(f"  attn_mask shape: {list(args[3].shape)}, dtype={args[3].dtype}")
            else:
                print("  attn_mask: None (causal)")
            print(f"  is_causal: {kwargs.get('is_causal', 'not set')}")
        call_log["total"] += 1
        return result

    # Patch and run
    with patch("torch.nn.functional.scaled_dot_product_attention", logging_sdpa):
        with torch.inference_mode():
            model.generate(**inputs, **gen_kwargs)

    print(f"  Total SDPA calls: {call_log['total']}")

    # Now test which backend is actually selected by disabling them one at a time
    print("\n--- Backend availability test ---")

    backends = {
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
    }
    for name, enabled in backends.items():
        print(f"  {name}: {'enabled' if enabled else 'DISABLED'}")

    # Run a single forward pass with each backend disabled to see which one is actually used
    # If disabling a backend causes an error, that backend was the one being used
    print("\n--- Backend selection probe (single forward pass) ---")

    # Prepare a single-step input matching generation conditions
    sample_input_ids = inputs["input_ids"][:1]  # single sequence
    sample_mask = inputs["attention_mask"][:1] if "attention_mask" in inputs else None

    # Expand for beam search to match real conditions
    expanded_ids = sample_input_ids.repeat(gen_kwargs.get("num_beams", 1), 1)
    expanded_mask = sample_mask.repeat(gen_kwargs.get("num_beams", 1), 1) if sample_mask is not None else None

    fwd_kwargs = {"input_ids": expanded_ids}
    if expanded_mask is not None:
        fwd_kwargs["attention_mask"] = expanded_mask

    # Toggle functions are setters (not context managers) in this PyTorch version
    toggle_fns = {
        "flash_sdp": torch.backends.cuda.enable_flash_sdp,
        "mem_efficient_sdp": torch.backends.cuda.enable_mem_efficient_sdp,
        "math_sdp": torch.backends.cuda.enable_math_sdp,
    }

    def run_with_only(backend_name):
        """Disable all backends except the named one, run forward, restore."""
        for n, fn in toggle_fns.items():
            fn(n == backend_name)
        try:
            with torch.inference_mode():
                model(**fwd_kwargs)
            return True, None
        except RuntimeError as e:
            return False, str(e)[:150]
        finally:
            for fn in toggle_fns.values():
                fn(True)

    print("\n--- Forced single-backend forward pass ---")
    for backend_name in ["flash_sdp", "mem_efficient_sdp", "math_sdp"]:
        ok, err = run_with_only(backend_name)
        label = backend_name.replace("_sdp", "").upper() + " ONLY"
        if ok:
            print(f"  {label}: OK")
        else:
            print(f"  {label}: FAILED ({err})")


def main():
    args = parse_args()
    s = Settings()

    if not s.MODEL_NAME:
        print("ERROR: MODEL_NAME not set. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {s.MODEL_NAME}")
    print(f"Batch size: {args.batch_size}, num_beams: {args.num_beams}")
    print(f"Effective sequences in attention: {args.batch_size * args.num_beams}")
    print()

    # Load model
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.bfloat16
    ts = str(s.DTYPE).lower()
    if ts in ("fp16", "float16", "torch.float16"):
        dtype = torch.float16

    print(f"Loading model (dtype={dtype}, attn=sdpa)...")
    model = AutoModelForCausalLM.from_pretrained(
        s.MODEL_NAME,
        device_map=s.DEVICE_MAP,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=s.TRUST_REMOTE_CODE,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build batched input (varying lengths to simulate real padding)
    prompts = [
        "Write a short greeting in Spanish.",
        "Translate the following to French: Hello, how are you?",
        "Create a simple dialogue between two people at a café.",
        "List three common Spanish phrases for travelers.",
        "Write a brief introduction in German.",
        "Translate to Italian: The weather is nice today.",
        "Create a short conversation about ordering food.",
        "Write two sentences about learning languages.",
    ][:args.batch_size]

    messages_batch = [[{"role": "user", "content": p}] for p in prompts]

    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=s.MAX_INPUT_TOKENS,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    print(f"Input shape: {list(inputs['input_ids'].shape)} (padded to {input_len} tokens)")
    print(f"Padding ratio: {(inputs['attention_mask'] == 0).float().mean():.1%}")
    print()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
        length_penalty=1.0,
    )

    print("=" * 60)
    print("PHASE 1: Generation with SDPA backend logging")
    print("=" * 60)
    probe_sdpa_backends(model, inputs, gen_kwargs)

    print()
    print("=" * 60)
    print("PHASE 2: Padded attention mask test")
    print("=" * 60)
    print("Testing if padding in batched input causes backend fallback...")

    # Compare: single unpacked prompt (no padding mask) vs padded batch
    single_input = tokenizer.apply_chat_template(
        messages_batch[:1],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=False,
    ).to(model.device)
    print(f"\nSingle prompt (no padding): shape={list(single_input['input_ids'].shape)}")

    single_ids = single_input["input_ids"]
    expanded_ids = single_ids.repeat(args.num_beams, 1)
    fwd_single = {"input_ids": expanded_ids}

    toggle_fns = {
        "flash_sdp": torch.backends.cuda.enable_flash_sdp,
        "mem_efficient_sdp": torch.backends.cuda.enable_mem_efficient_sdp,
        "math_sdp": torch.backends.cuda.enable_math_sdp,
    }

    def try_single_backend(fwd_kwargs, backend_name):
        for n, fn in toggle_fns.items():
            fn(n == backend_name)
        try:
            with torch.inference_mode():
                model(**fwd_kwargs)
            return True
        except RuntimeError:
            return False
        finally:
            for fn in toggle_fns.values():
                fn(True)

    for bk in ["flash_sdp", "mem_efficient_sdp"]:
        label = bk.replace("_sdp", "").upper() + " ONLY"
        ok = try_single_backend(fwd_single, bk)
        print(f"  No-padding {label}: {'OK' if ok else 'FAILED'}")

    # Now with explicit padding mask (simulating batched input)
    padded_ids = torch.nn.functional.pad(single_ids, (16, 0), value=tokenizer.pad_token_id)
    padded_mask = torch.nn.functional.pad(
        torch.ones_like(single_ids), (16, 0), value=0
    )
    padded_ids = padded_ids.repeat(args.num_beams, 1)
    padded_mask = padded_mask.repeat(args.num_beams, 1)
    fwd_padded = {"input_ids": padded_ids, "attention_mask": padded_mask}

    print(f"\nPadded prompt (16 pad tokens): shape={list(padded_ids.shape)}")
    for bk in ["flash_sdp", "mem_efficient_sdp"]:
        label = bk.replace("_sdp", "").upper() + " ONLY"
        ok = try_single_backend(fwd_padded, bk)
        print(f"  With-padding {label}: {'OK' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
