"""
Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks
for Continual LLM Learning — Model Architecture (MLX)

Paper: https://arxiv.org/abs/2604.01152
Implements: MoELoRADelta (§3.1), StackedMoELoRALayer (§3.2),
            NullSpaceProjector (§3.5), MetaRouter (§3.6)

Section references:
  §3.1 — MoE-LoRA building block (experts, routing, load balancing)
  §3.2 — StackedMoELoRALayer (additive composition, CPU offloading)
  §3.5 — Null-space projection (SVD, projector construction)
  §3.6 — Meta-router (outcome-based sigmoid gating)
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BrainstacksConfig:
    """Configuration for Brainstacks components.

    All defaults from paper (2604.01152) unless marked [UNSPECIFIED].
    """

    # §3.1 — MoE-LoRA building block
    num_experts: int = 4           # §3.1 — "N = 4 experts"
    top_k: int = 2                 # §3.1 — "top-K = 2 are active per token"
    rank: int = 16                 # §3.1 — "rank r = 16"
    alpha: int = 16                # §3.1 — "α = r = 16"
    aux_loss_coeff: float = 0.01   # §3.1 — "λ_aux = 0.01"

    # §3.5 — Null-space projection
    ns_n_samples: int = 400        # §3.5 — "n_samples = 400"
    ns_top_k: int = 64             # §3.5 — "top-K = 64 principal directions"

    # §3.6 — Meta-router
    router_hidden_dim: int = 512   # §3.6 — "hidden dim 512"
    mid_layer_weight: float = 0.45 # §3.6 — "0.45 × mid"
    last_layer_weight: float = 0.55  # §3.6 — "0.55 × last"
    chat_floor: float = 0.20      # §3.6 — "chat floor of 0.20"
    gate_threshold: float = 0.12  # §3.6 — "gate threshold (0.12)"
    router_dropout: float = 0.1   # [UNSPECIFIED] — dropout rate not given
    num_domains: int = 5          # §4.3 — "5 domains"


class MoELoRADelta(nn.Module):
    """§3.1 — MoE-LoRA building block with Shazeer-style noisy top-2 routing.

    "The fundamental unit of Brainstacks is the MoE-LoRA delta module
    (MoELoRADelta), which replaces each targeted linear projection in the
    transformer." (§3.1)

    Each expert i: A_i ∈ R^{d_in × r} (Kaiming uniform), B_i ∈ R^{r × d_out} (zeros)
    rsLoRA scaling: s = α / √r
    Router: noisy top-K gating with learned noise projection
    """

    def __init__(self, d_in: int, d_out: int, config: BrainstacksConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_experts = config.num_experts  # §3.1 — N = 4
        self.top_k = config.top_k              # §3.1 — K = 2
        self.rank = config.rank                # §3.1 — r = 16

        # §3.1 — rsLoRA scaling: "α = r = 16 yields effective scale s = 4.0"
        self.scale = config.alpha / math.sqrt(config.rank)

        # §3.1 — Expert weights: "A_i ∈ R^{d_in × r}, B_i ∈ R^{r × d_out}"
        # Stacked for vectorized computation
        # §3.1 — "Kaiming uniform initialization" for A
        std = 1.0 / math.sqrt(d_in)
        self.A = mx.random.uniform(
            low=-std, high=std,
            shape=(self.num_experts, self.rank, d_in),
        )
        # §3.1 — "zero initialization" for B
        self.B = mx.zeros((self.num_experts, d_out, self.rank))

        # §3.1 — Noisy Top-K Router
        # "weight projection W_r ∈ R^{d_in × N} and noise projection W_n ∈ R^{d_in × N}"
        self.W_r = nn.Linear(d_in, self.num_experts, bias=False)
        self.W_n = nn.Linear(d_in, self.num_experts, bias=False)

    def __call__(self, x: mx.array, training: bool = False) -> tuple[mx.array, mx.array]:
        """Forward pass computing gated expert corrections.

        §3.1 — "The final output combines the frozen base with gated expert corrections"
        y = base_output + Σ_i g_i * s * B_i(A_i · x)

        Args:
            x: input tensor — shape: (batch, seq_len, d_in)
            training: whether to add router noise

        Returns:
            delta: expert correction — shape: (batch, seq_len, d_out)
            aux_loss: load balancing loss — scalar
        """
        batch, seq_len, _ = x.shape

        # §3.1 — Router logits with Shazeer-style noisy gating
        logits = self.W_r(x)  # (batch, seq_len, N)

        if training:
            # §3.1 — "noise magnitude is input-dependent and learned"
            noise_std = nn.softplus(self.W_n(x))  # (batch, seq_len, N)
            noise = mx.random.normal(logits.shape) * noise_std  # (batch, seq_len, N)
            logits = logits + noise
        # §3.1 — "At inference, the noise term is disabled"

        # §3.1 — "entries outside the top-K are set to −∞ before softmax"
        top_k_idx = mx.argpartition(-logits, kth=self.top_k, axis=-1)[..., :self.top_k]
        # (batch, seq_len, K)

        # Build sparse gate: set non-top-K to -inf
        mask = mx.full(logits.shape, -1e9)  # (batch, seq_len, N)
        # Scatter top-K logit values back
        top_k_vals = mx.take_along_axis(logits, top_k_idx, axis=-1)  # (batch, seq_len, K)

        # Reconstruct: place top-K values, rest stay -inf
        # Use put_along_axis for scatter
        mask = mask.at[
            mx.arange(batch)[:, None, None],
            mx.arange(seq_len)[None, :, None],
            top_k_idx,
        ].add(top_k_vals + 1e9)

        # §3.1 — "producing exactly K non-zero gates that sum to 1"
        gates = mx.softmax(mask, axis=-1, precise=True)  # (batch, seq_len, N)

        # §3.1 — Vectorized expert computation via einsum
        # "stacked A ∈ R^{N × r × d_in}, B ∈ R^{N × d_out × r}"
        h = mx.einsum("bsd,nrd->bsnr", x, self.A)  # (batch, seq_len, N, r)
        expert_out = mx.einsum("bsnr,nor->bsno", h, self.B)  # (batch, seq_len, N, d_out)

        # Apply gating and rsLoRA scaling
        # §3.1 — y = Σ_i g_i * s * B_i(A_i · x)
        gated = mx.einsum("bsn,bsno->bso", gates, expert_out)  # (batch, seq_len, d_out)
        delta = gated * self.scale  # Python scalar — no type promotion issue

        # §3.1 — Load balancing auxiliary loss
        # "L_aux = Σ_e P(e) · f(e)" where P(e) is mean routing prob, f(e) is dispatch fraction
        P = mx.mean(gates, axis=(0, 1))  # (N,) — mean routing probability per expert
        f = mx.mean((gates > 0).astype(gates.dtype), axis=(0, 1))  # (N,) — dispatch fraction
        aux_loss = mx.sum(P * f) * self.num_experts  # scalar

        return delta, aux_loss


class StackedMoELoRALayer(nn.Module):
    """§3.2 — Manages additive composition of frozen + active MoE-LoRA stacks.

    "Each transformer projection is wrapped in a StackedMoELoRALayer that
    manages additive composition" (§3.2)

    On Apple Silicon unified memory, CPU offloading is less relevant —
    frozen stacks stay in unified memory but are excluded from gradient computation.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        d_in: int,
        d_out: int,
        config: BrainstacksConfig,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.d_in = d_in
        self.d_out = d_out
        self.config = config

        self.frozen_stacks: list[dict] = []  # list of frozen weight dicts
        self.active_stack: Optional[MoELoRADelta] = None

        # §3.5 — Null-space projector (set externally before training new domain)
        self._null_space_P: Optional[mx.array] = None

        # §3.6 — Domain weights from meta-router (set at inference time)
        self._domain_weights: Optional[mx.array] = None

    def add_new_stack(self) -> None:
        """Add a new trainable MoE-LoRA stack.

        §3.3, Algorithm 1, line 3 — "Add new trainable MoELoRADelta stack to each layer"
        §3.1 — "Zero initialization of B ensures that new stacks start as identity"
        """
        self.active_stack = MoELoRADelta(self.d_in, self.d_out, self.config)

    def freeze_active_stack(self) -> None:
        """Freeze the active stack and add to frozen list.

        §3.2 — On Apple Silicon, we freeze by saving weights and removing
        from trainable parameters. No CPU/GPU shuttle needed (unified memory).
        §3.3, Algorithm 1, line 6 — "Freeze active stack → move to frozen stacks"
        """
        if self.active_stack is None:
            return
        # Save weights as frozen (non-trainable) arrays
        frozen_weights = self.active_stack.parameters()
        self.frozen_stacks.append(frozen_weights)
        self.active_stack = None

    def _compute_frozen_delta(self, x: mx.array, weights: dict) -> mx.array:
        """Compute delta from a single frozen stack's weights without gradients.

        Args:
            x: input — shape: (batch, seq_len, d_in)
            weights: frozen stack parameter dict

        Returns:
            delta — shape: (batch, seq_len, d_out)
        """
        A = weights["A"]  # (N, r, d_in)
        B = weights["B"]  # (N, d_out, r)
        W_r_weight = weights["W_r"]["weight"]  # (N, d_in) — nn.Linear stores (out, in)

        # Router (no noise at inference — §3.1)
        logits = x @ W_r_weight.T  # (batch, seq_len, N)

        top_k_idx = mx.argpartition(-logits, kth=self.config.top_k, axis=-1)[..., :self.config.top_k]
        mask = mx.full(logits.shape, -1e9)
        top_k_vals = mx.take_along_axis(logits, top_k_idx, axis=-1)
        mask = mask.at[
            mx.arange(x.shape[0])[:, None, None],
            mx.arange(x.shape[1])[None, :, None],
            top_k_idx,
        ].add(top_k_vals + 1e9)
        gates = mx.softmax(mask, axis=-1, precise=True)  # (batch, seq_len, N)

        # Expert computation
        scale = self.config.alpha / math.sqrt(self.config.rank)
        h = mx.einsum("bsd,nrd->bsnr", x, A)
        expert_out = mx.einsum("bsnr,nor->bsno", h, B)
        gated = mx.einsum("bsn,bsno->bso", gates, expert_out)
        return gated * scale

    def __call__(self, x: mx.array, training: bool = False) -> tuple[mx.array, mx.array]:
        """Forward pass with additive composition.

        §3.2 — "y = base(x) + Σ frozen_delta_i(x) + active_delta(x)"

        Args:
            x: input — shape: (batch, seq_len, d_in)
            training: whether in training mode

        Returns:
            output: base + all deltas — shape: (batch, seq_len, d_out)
            total_aux_loss: sum of aux losses — scalar
        """
        y = self.base_layer(x)  # (batch, seq_len, d_out)
        total_aux_loss = mx.array(0.0)

        # §3.2 — Frozen stacks (no gradient)
        for i, frozen_w in enumerate(self.frozen_stacks):
            delta = mx.stop_gradient(self._compute_frozen_delta(x, frozen_w))
            if self._domain_weights is not None:
                weight = self._domain_weights[min(i, self._domain_weights.shape[0] - 1)]
                delta = delta * weight
            y = y + delta

        # Active stack (trainable)
        if self.active_stack is not None:
            delta, aux_loss = self.active_stack(x, training=training)
            total_aux_loss = total_aux_loss + aux_loss

            # §3.5 — Null-space projection: "δ_projected = δ − δ · P"
            if self._null_space_P is not None:
                delta = delta - (delta @ self._null_space_P)

            y = y + delta

        return y, total_aux_loss


class NullSpaceProjector:
    """§3.5 — Computes null-space projectors for StackedMoELoRALayers.

    "Before training each new domain (from domain 2 onward), Brainstacks
    computes null-space projectors for every StackedMoELoRALayer that has
    frozen stacks." (§3.5)

    Procedure:
    1. Run n_samples validation examples, collect frozen stack output deltas
    2. Stack into matrix D of shape [n_samples, h_dim]
    3. Compute top-K principal directions via SVD
    4. Form P = V · V^T
    """

    def __init__(self, config: BrainstacksConfig):
        self.n_samples = config.ns_n_samples  # §3.5 — 400
        self.top_k = config.ns_top_k          # §3.5 — 64

    def compute_projector(self, deltas: mx.array) -> mx.array:
        """Compute the null-space projection matrix from collected deltas.

        §3.5 — SVD to find top-K principal directions, then P = V · V^T

        Note: MLX has mx.linalg.svd but no svd_lowrank equivalent.
        For n_samples > 2K we use full SVD and truncate to K directions.

        Args:
            deltas: collected frozen stack output deltas — shape: (n_samples, h_dim)

        Returns:
            P: projection matrix — shape: (h_dim, h_dim)
        """
        # Full SVD, truncate to top_k
        # §3.5 — "top-K = 64 principal directions"
        _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)  # SVD on CPU for stability
        V = Vt[:self.top_k].T  # (h_dim, top_k)

        # §3.5 — "Form the projection matrix P = V · V^T"
        P = V @ V.T  # (h_dim, h_dim)
        mx.eval(P)
        return P


class MetaRouter(nn.Module):
    """§3.6 — Outcome-based sigmoid gating meta-router (~2M parameters).

    "A lightweight neural network (~2M parameters) that takes a prompt's deep
    semantic features (weighted average of mid-layer and last-layer hidden states)
    and outputs independent sigmoid probabilities per domain." (§3.6)

    Architecture: Token projection → learned query attention (global context) →
    cross-attention with domain queries → fusion MLP → per-domain sigmoid output
    """

    def __init__(self, d_model: int, config: BrainstacksConfig):
        super().__init__()
        self.config = config
        h = config.router_hidden_dim  # §3.6 — 512
        num_d = config.num_domains    # §4.3 — 5

        # §3.6 — "Token projection to hidden dim 512"
        self.token_proj = nn.Linear(d_model, h)

        # §3.6 — "global context via learned query attention"
        self.global_query = mx.random.normal((1, 1, h)) * 0.02
        self.global_attn = nn.MultiHeadAttention(h, num_heads=4)

        # §3.6 — "per-domain context via cross-attention with learnable domain query vectors"
        self.domain_queries = mx.random.normal((1, num_d, h)) * 0.02
        self.domain_attn = nn.MultiHeadAttention(h, num_heads=4)

        # §3.6 — "fusion MLP with GELU activation and dropout"
        self.fusion_linear1 = nn.Linear(h * 2, h)
        self.fusion_dropout = nn.Dropout(config.router_dropout)  # [UNSPECIFIED] rate
        self.fusion_linear2 = nn.Linear(h, num_d)

        # §3.6 — "learned temperature scaling"
        self.temperature = mx.ones((num_d,))

    def __call__(
        self,
        mid_hidden: mx.array,
        last_hidden: mx.array,
        training: bool = False,
    ) -> mx.array:
        """Compute per-domain gating probabilities.

        §3.6 — "predicts from base-model-only hidden states (all stacks disabled)"

        Args:
            mid_hidden: mid-layer hidden states — shape: (batch, seq_len, d_model)
            last_hidden: last-layer hidden states — shape: (batch, seq_len, d_model)
            training: whether in training mode (for dropout)

        Returns:
            domain_probs: per-domain sigmoid probabilities — shape: (batch, num_domains)
        """
        batch = mid_hidden.shape[0]

        # §3.6 — "0.45 × mid + 0.55 × last"
        h = (0.45 * mid_hidden + 0.55 * last_hidden)  # (batch, seq_len, d_model)

        h = self.token_proj(h)  # (batch, seq_len, hidden_dim)

        # §3.6 — Global context via learned query attention
        global_q = mx.broadcast_to(
            self.global_query, (batch, 1, self.config.router_hidden_dim)
        )  # (batch, 1, hidden_dim)
        global_ctx = self.global_attn(global_q, h, h)  # (batch, 1, hidden_dim)

        # §3.6 — Per-domain context via cross-attention with domain queries
        domain_q = mx.broadcast_to(
            self.domain_queries, (batch, self.config.num_domains, self.config.router_hidden_dim)
        )  # (batch, num_domains, hidden_dim)
        domain_ctx = self.domain_attn(domain_q, h, h)  # (batch, num_domains, hidden_dim)

        # Fusion: concatenate global context (broadcast) with per-domain context
        global_expanded = mx.broadcast_to(
            global_ctx, (batch, self.config.num_domains, self.config.router_hidden_dim)
        )  # (batch, num_domains, hidden_dim)
        fused = mx.concatenate([global_expanded, domain_ctx], axis=-1)  # (batch, num_domains, hidden_dim*2)

        # §3.6 — Fusion MLP → per-domain logits
        fused = self.fusion_linear1(fused)  # (batch, num_domains, hidden_dim)
        fused = nn.gelu(fused)              # §3.6 — "GELU activation"
        if training:
            fused = self.fusion_dropout(fused)
        logits = self.fusion_linear2(fused)  # (batch, num_domains, num_domains)

        # Take diagonal: each domain's logit from its position
        # logits[:, i, i] for each domain i
        logits = mx.take_along_axis(
            logits,
            mx.arange(self.config.num_domains)[None, :, None].astype(mx.int32),
            axis=-1,
        ).squeeze(-1)  # (batch, num_domains)

        # §3.6 — "learned temperature scaling"
        logits = logits / self.temperature  # (batch, num_domains)

        # §3.6 — "independent sigmoid probabilities per domain"
        return mx.sigmoid(logits)  # (batch, num_domains)

    def gate_domains(self, domain_probs: mx.array) -> mx.array:
        """Apply gating thresholds for inference.

        §3.6 — "chat floor of 0.20", "gate threshold (0.12)"

        Args:
            domain_probs: raw sigmoid probabilities — shape: (batch, num_domains)

        Returns:
            gated_weights: thresholded domain weights — shape: (batch, num_domains)
        """
        weights = domain_probs

        # §3.6 — Chat floor: domain 0 = chat
        chat_col = mx.maximum(weights[:, 0:1], self.config.chat_floor)
        weights = mx.concatenate([chat_col, weights[:, 1:]], axis=-1)

        # §3.6 — Gate threshold: "weight below 0.12 are not loaded"
        weights = mx.where(weights >= self.config.gate_threshold, weights, 0.0)

        return weights
