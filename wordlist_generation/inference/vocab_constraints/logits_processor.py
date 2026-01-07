from __future__ import annotations

from typing import Callable, Sequence

import torch
from transformers.generation.logits_process import LogitsProcessor


PrefixAllowedTokensFn = Callable[[int, torch.LongTensor], Sequence[int]]


class TieredSoftPrefixConstraintLogitsProcessor(LogitsProcessor):
    """Tiered soft alternative to prefix_allowed_tokens_fn.

    Applies three tiers based on two prefix-allowed sets:
      - allowed_n (rank <= N): no penalty
      - allowed_kn minus allowed_n (N < rank <= kN): penalty m
      - everything else (rank > kN): penalty n
    """

    def __init__(
        self,
        *,
        prefix_allowed_tokens_fn_n: PrefixAllowedTokensFn,
        prefix_allowed_tokens_fn_kn: PrefixAllowedTokensFn,
        penalty_m: float,
        penalty_n: float,
    ):
        self.prefix_allowed_tokens_fn_n = prefix_allowed_tokens_fn_n
        self.prefix_allowed_tokens_fn_kn = prefix_allowed_tokens_fn_kn
        self.penalty_m = float(penalty_m)
        self.penalty_n = float(penalty_n)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty_n <= 0:
            return scores

        batch_size = int(scores.shape[0])
        tier1_boost = float(self.penalty_n - self.penalty_m)

        for batch_id in range(batch_size):
            allowed_n = self.prefix_allowed_tokens_fn_n(batch_id, input_ids[batch_id])
            allowed_kn = self.prefix_allowed_tokens_fn_kn(batch_id, input_ids[batch_id])
            if not allowed_n and not allowed_kn:
                continue

            allowed_kn_set = set(allowed_kn) if allowed_kn else set()
            if allowed_n:
                allowed_kn_set.update(allowed_n)

            scores[batch_id] = scores[batch_id] - self.penalty_n

            if allowed_kn_set and tier1_boost != 0.0:
                idx = torch.tensor(list(allowed_kn_set), device=scores.device, dtype=torch.long)
                scores[batch_id, idx] = scores[batch_id, idx] + tier1_boost

            if allowed_n and self.penalty_m != 0.0:
                idx0 = torch.tensor(list(allowed_n), device=scores.device, dtype=torch.long)
                scores[batch_id, idx0] = scores[batch_id, idx0] + self.penalty_m

        return scores
