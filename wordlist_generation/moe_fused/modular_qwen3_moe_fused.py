# Modified from https://github.com/huggingface/transformers/blob/bdf5fb70aa11782cce22027d76879f71f4e41c1e/src/transformers/models/qwen3_moe/modular_qwen3_moe.py

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
)
from transformers.utils.generic import OutputRecorder

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from .functional import moe_fused_linear
from .kernels.indexing import get_expert_counts_and_idx


def moe_fused_kaiming_uniform_(weight: torch.Tensor) -> None:
    # Kaiming uniform on in_features
    # Although Qwen's default activation is silu, we set the gain `a = sqrt(5)` following the original Linear
    in_features = weight.shape[-1]
    bound = math.sqrt(3 * 5 / in_features)
    nn.init.uniform_(weight, -bound, bound)


class MoeFusedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_experts"]
    in_features: int
    out_features: int
    num_experts: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        moe_fused_kaiming_uniform_(self.weight)

    def forward(self, input: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
        return moe_fused_linear(input, self.weight, m_sizes)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, num_experts={self.num_experts}"


# This class follows the implementation in HF Transformers
# patch_Qwen3MoeFusedSparseMoeBlock_forward can make it faster
class Qwen3MoeFusedSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_selected = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate_proj = MoeFusedLinear(self.hidden_size, self.moe_intermediate_size, config.num_experts)
        self.up_proj = MoeFusedLinear(self.hidden_size, self.moe_intermediate_size, config.num_experts)
        self.down_proj = MoeFusedLinear(self.moe_intermediate_size, self.hidden_size, config.num_experts)
        assert config.hidden_act == "silu"

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        M = batch_size * sequence_length
        # Use down_proj's device as the final compute device (where most computation happens)
        compute_device = self.down_proj.weight.device

        hidden_states = hidden_states.view(M, hidden_dim).to(compute_device)
        # router_logits: (M, num_experts)
        router_logits = self.gate(hidden_states.to(self.gate.weight.device))
        router_logits = router_logits.to(compute_device)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        # routing_weights, selected_experts: (M, num_selected)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_selected, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype and ensure correct device
        routing_weights = routing_weights.to(device=compute_device, dtype=hidden_states.dtype)

        hidden_states = hidden_states.unsqueeze(1).expand(M, self.num_selected, hidden_dim)
        # hidden_states must be contiguous
        hidden_states = hidden_states.reshape(M * self.num_selected, hidden_dim)
        selected_experts = selected_experts.view(M * self.num_selected).to(compute_device)

        # Sort selected_experts and hidden_states for better memory coalescence of weight
        # It's possible to fuse a sort and a MoeFusedLinear layer, but for now we separate them for clarity
        m_sizes, sort_idx, inv_sort_idx = get_expert_counts_and_idx(selected_experts, self.num_experts)
        hidden_states = hidden_states[sort_idx]

        # Ensure hidden_states and m_sizes are on each layer's device for the fused linear ops
        hidden_states = hidden_states.to(self.gate_proj.weight.device)
        m_sizes_gate = m_sizes.to(self.gate_proj.weight.device)
        gate_h = self.gate_proj(hidden_states, m_sizes_gate)

        hidden_states = hidden_states.to(self.up_proj.weight.device)
        m_sizes_up = m_sizes.to(self.up_proj.weight.device)
        up_h = self.up_proj(hidden_states, m_sizes_up)

        # Ensure gate_h and up_h are on the same device for the multiplication
        up_h = up_h.to(gate_h.device)
        hidden_states = F.silu(gate_h) * up_h
        del gate_h, up_h

        hidden_states = hidden_states.to(self.down_proj.weight.device)
        m_sizes_down = m_sizes.to(self.down_proj.weight.device)
        hidden_states = self.down_proj(hidden_states, m_sizes_down)

        # Ensure inv_sort_idx is on the same device as hidden_states for indexing
        inv_sort_idx = inv_sort_idx.to(hidden_states.device)
        hidden_states = hidden_states[inv_sort_idx]

        hidden_states = hidden_states.view(M, self.num_selected, hidden_dim)
        routing_weights = routing_weights.to(hidden_states.device)
        hidden_states = torch.einsum("beo,be->bo", hidden_states, routing_weights)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        router_logits = router_logits.to(hidden_states.device)
        return hidden_states, router_logits


class Qwen3MoeFusedDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeFusedSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        """Forward with multi-GPU device handling for residual connections."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        # Ensure hidden_states is on the same device as residual for multi-GPU compatibility
        hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3MoeFusedModel(Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3MoeFusedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Qwen3MoeFusedForCausalLM(Qwen3MoeForCausalLM):
    # Tell accelerate not to split these modules across devices for multi-GPU
    # This keeps MoE blocks intact on a single device for efficient Triton kernel execution
    _no_split_modules = ["Qwen3MoeFusedDecoderLayer", "Qwen3MoeFusedSparseMoeBlock"]

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.model = Qwen3MoeFusedModel(config)
        self._can_record_outputs["router_logits"] = OutputRecorder(Qwen3MoeFusedSparseMoeBlock, index=1)
        self._can_record_outputs["hidden_states"] = Qwen3MoeFusedDecoderLayer
