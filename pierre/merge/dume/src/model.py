"""
DUME MoE model components (MLX).

Implements the MoE block with top-k routing and the full DUME model wrapper.

Paper: arxiv.org/abs/2603.29765
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from pierre.merge.dume.src.router import RidgeRouter


class ExpertMLP(nn.Module):
    """A single MLP expert block (gate-up-down, SiLU activation).

    Standard LLaMA-style MLP: out = down(silu(gate(x)) * up(x))

    Args:
        hidden_dim: H — input/output dimension.
        intermediate_dim: MLP intermediate dimension.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DUMEMoEBlock(nn.Module):
    """MoE block: routes tokens to experts via ridge-regression-initialized router.

    Section 2.2, Eq. 4:
        MoE_l(x) = sum_{d in Top-k} g_d(x; W_l) * E_{d,l}(x)

    where g is the gating weight from Top-k(Softmax(W_l^T x)).

    Args:
        hidden_dim: H.
        intermediate_dim: MLP intermediate size.
        num_experts: D — number of domain experts.
        top_k: experts selected per token (default 1).
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = RidgeRouter(hidden_dim, num_experts, top_k)
        self.experts = [
            ExpertMLP(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with top-k expert routing.

        Args:
            x: (batch, seq, H) hidden states.

        Returns:
            (batch, seq, H) — weighted sum of selected expert outputs.
        """
        orig_shape = x.shape
        # Flatten to (N, H) for routing
        if x.ndim == 3:
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x

        indices, weights = self.router(x_flat)  # (N, top_k), (N, top_k)
        N, H = x_flat.shape

        # Compute expert outputs and combine
        output = mx.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_ids = indices[:, k]  # (N,)
            gate_weights = weights[:, k : k + 1]  # (N, 1)

            for e in range(self.num_experts):
                mask = expert_ids == e  # (N,)
                if not mx.any(mask).item():
                    continue
                # Gather tokens for this expert
                token_indices = mx.argwhere(mask).squeeze(-1)
                expert_input = x_flat[token_indices]
                expert_output = self.experts[e](expert_input)
                expert_gate = gate_weights[token_indices]
                # Scatter back
                output = output.at[token_indices].add(expert_gate * expert_output)

        return output.reshape(orig_shape)

    def deterministic_forward(self, x: mx.array, domain_id: int) -> mx.array:
        """Forward pass routing ALL tokens to a specific domain's expert.

        Used during ridge regression statistics extraction (Algorithm 1):
        instead of using the gating layer, we deterministically forward each
        feature map to the MoE expert corresponding to the domain of the input.

        Args:
            x: (batch, seq, H) hidden states.
            domain_id: integer domain label d in [0, D).

        Returns:
            (batch, seq, H) — output of expert d.
        """
        return self.experts[domain_id](x)


class DUMEModel(nn.Module):
    """Wrapper that holds shared transformer layers + MoE blocks.

    This is a structural skeleton. In practice you would load a base
    architecture (e.g. LLaMA via mlx-lm) and replace MLP blocks with
    DUMEMoEBlock instances after MoErging.

    Args:
        hidden_dim: H.
        intermediate_dim: MLP intermediate size.
        num_layers: L — number of transformer blocks.
        num_experts: D.
        top_k: routing top-k.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_layers: int,
        num_experts: int,
        top_k: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.moe_blocks = [
            DUMEMoEBlock(hidden_dim, intermediate_dim, num_experts, top_k)
            for _ in range(num_layers)
        ]

    def set_router_weights(self, weights: list[mx.array]) -> None:
        """Set router weights from ridge regression solution.

        Args:
            weights: list of L arrays, each (H, D).
        """
        assert len(weights) == self.num_layers
        for l, W in enumerate(weights):
            self.moe_blocks[l].router.weight = W

    def get_hidden_states_for_extraction(
        self,
        forward_fn,
        input_ids: mx.array,
        domain_id: int,
    ) -> list[mx.array]:
        """Extract hidden states before each MoE block for router training.

        During extraction (Algorithm 1), tokens are deterministically routed
        to their domain's expert.

        Args:
            forward_fn: callable that takes input_ids and returns hidden states
                        at each layer boundary. This depends on your base model.
            input_ids: (batch, seq) token ids.
            domain_id: which domain these tokens belong to.

        Returns:
            List of L hidden state arrays, each (batch, seq, H).
        """
        # This is a template — the actual implementation depends on the base
        # model architecture. The key idea: at each MoE block, collect the
        # hidden state BEFORE the MoE, then use deterministic_forward to
        # route through the correct expert for subsequent layers.
        hidden_states = forward_fn(input_ids)
        return hidden_states
