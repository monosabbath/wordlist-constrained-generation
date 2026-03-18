import importlib.util
import sys
import types

# Stub transformers to avoid heavy dependency for unit tests
_transformers = types.ModuleType("transformers")
_gen = types.ModuleType("transformers.generation")
_lp = types.ModuleType("transformers.generation.logits_process")

class _FakeLogitsProcessor:
    pass

_lp.LogitsProcessor = _FakeLogitsProcessor
_transformers.generation = _gen
sys.modules["transformers"] = _transformers
sys.modules["transformers.generation"] = _gen
sys.modules["transformers.generation.logits_process"] = _lp

import torch

# Direct import of the module file to skip wordlist_generation.__init__
spec = importlib.util.spec_from_file_location(
    "logits_processor",
    "wordlist_generation/inference/vocab_constraints/logits_processor.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PresencePenaltyLogitsProcessor = mod.PresencePenaltyLogitsProcessor


def make_scores(batch_size: int, vocab_size: int, fill: float = 0.0) -> torch.FloatTensor:
    return torch.full((batch_size, vocab_size), fill, dtype=torch.float32)


def test_zero_penalty_is_noop():
    proc = PresencePenaltyLogitsProcessor(penalty=0.0, prompt_len=3)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])  # prompt=3, generated=[40, 50]
    scores = make_scores(1, 100, fill=1.0)
    result = proc(input_ids, scores)
    assert torch.equal(result, make_scores(1, 100, fill=1.0))


def test_penalizes_generated_tokens_only():
    """Tokens in the prompt should NOT be penalized; only generated tokens should."""
    proc = PresencePenaltyLogitsProcessor(penalty=2.0, prompt_len=2)
    # prompt = [10, 20], generated = [10, 30]
    # Token 10 appears in both prompt and generated — should be penalized (it was generated)
    # Token 20 appears only in prompt — should NOT be penalized
    # Token 30 appears only in generated — should be penalized
    input_ids = torch.tensor([[10, 20, 10, 30]])
    scores = make_scores(1, 100, fill=5.0)
    result = proc(input_ids, scores)

    assert result[0, 10].item() == 3.0  # 5.0 - 2.0
    assert result[0, 30].item() == 3.0  # 5.0 - 2.0
    assert result[0, 20].item() == 5.0  # untouched (prompt only)
    assert result[0, 0].item() == 5.0   # untouched (never appeared)


def test_no_generated_tokens_yet():
    """When input_ids is just the prompt, nothing should be penalized."""
    proc = PresencePenaltyLogitsProcessor(penalty=2.0, prompt_len=3)
    input_ids = torch.tensor([[10, 20, 30]])
    scores = make_scores(1, 100, fill=5.0)
    result = proc(input_ids, scores)
    assert torch.equal(result, make_scores(1, 100, fill=5.0))


def test_repeated_generated_token_penalized_once():
    """A token appearing multiple times in generated output should still only get one penalty."""
    proc = PresencePenaltyLogitsProcessor(penalty=3.0, prompt_len=1)
    # prompt = [0], generated = [5, 5, 5]
    input_ids = torch.tensor([[0, 5, 5, 5]])
    scores = make_scores(1, 100, fill=10.0)
    result = proc(input_ids, scores)

    assert result[0, 5].item() == 7.0   # 10.0 - 3.0 (once, not thrice)
    assert result[0, 0].item() == 10.0  # prompt token, untouched


def test_batch_independence():
    """Each sequence in the batch should be penalized based on its own generated tokens."""
    proc = PresencePenaltyLogitsProcessor(penalty=1.0, prompt_len=2)
    input_ids = torch.tensor([
        [10, 20, 30, 40],  # generated = [30, 40]
        [10, 20, 50, 60],  # generated = [50, 60]
    ])
    scores = make_scores(2, 100, fill=5.0)
    result = proc(input_ids, scores)

    # Batch 0: tokens 30, 40 penalized
    assert result[0, 30].item() == 4.0
    assert result[0, 40].item() == 4.0
    assert result[0, 50].item() == 5.0  # not in batch 0's generation

    # Batch 1: tokens 50, 60 penalized
    assert result[1, 50].item() == 4.0
    assert result[1, 60].item() == 4.0
    assert result[1, 30].item() == 5.0  # not in batch 1's generation
