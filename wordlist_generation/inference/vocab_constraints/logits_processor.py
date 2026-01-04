from __future__ import annotations

from typing import Callable, Sequence

import torch
from transformers.generation.logits_process import LogitsProcessor


PrefixAllowedTokensFn = Callable[[int, torch.LongTensor], Sequence[int]]


class SoftPrefixConstraintLogitsProcessor(LogitsProcessor):
    """Soft alternative to prefix_allowed_tokens_fn.

    Decreases logits for tokens not in the allowed set by a fixed penalty.
    """

    def __init__(self, *, prefix_allowed_tokens_fn: PrefixAllowedTokensFn, penalty: float):
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty <= 0:
            return scores

        batch_size = int(scores.shape[0])
        for batch_id in range(batch_size):
            allowed = self.prefix_allowed_tokens_fn(batch_id, input_ids[batch_id])
            if not allowed:
                continue

            # Constant shift is softmax-invariant; this effectively penalizes disallowed tokens only.
            scores[batch_id] = scores[batch_id] - self.penalty
            idx = torch.tensor(list(allowed), device=scores.device, dtype=torch.long)
            scores[batch_id, idx] = scores[batch_id, idx] + self.penalty
        return scores
