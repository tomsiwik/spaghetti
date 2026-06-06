"""
Ridge regression router for DUME (Section 2.1, 2.3).

Core equations:
  W* = (X^T X + lambda I)^{-1} X^T Y          (Eq. 2)
  A_l += F_{d,<l}(X_i)^T F_{d,<l}(X_i)        (Eq. 7, incremental X^T X)
  b_l += F_{d,<l}(X_i)^T Y_i                   (Eq. 7, incremental X^T Y)
  W*[:,c] = W*[:,c] / ||W*[:,c]||               (column normalization)

Paper: arxiv.org/abs/2603.29765
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RidgeRouter(nn.Module):
    """Top-k router with weights initialized via ridge regression.

    Args:
        hidden_dim: H — hidden size of the transformer.
        num_experts: D — number of domain experts (MoE experts per block).
        top_k: number of experts selected per token (default 1).
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # W_l in R^{H x D} — the router linear projection
        self.weight = mx.zeros((hidden_dim, num_experts))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens to experts.

        Args:
            x: (batch, seq, H) or (seq, H) hidden states.

        Returns:
            expert_indices: (..., top_k) selected expert ids.
            expert_weights: (..., top_k) softmax gating weights.
        """
        # g(x; W_l) = Top-k(Softmax(W_l^T x))  — Eq. 4
        logits = x @ self.weight  # (..., D)
        probs = mx.softmax(logits, axis=-1)

        # Top-k selection
        if self.top_k >= self.num_experts:
            indices = mx.broadcast_to(
                mx.arange(self.num_experts), (*probs.shape[:-1], self.num_experts)
            )
            weights = probs
        else:
            indices = mx.argpartition(-probs, kth=self.top_k, axis=-1)[
                ..., : self.top_k
            ]
            weights = mx.take_along_axis(probs, indices, axis=-1)
            # Re-normalize selected weights
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)

        return indices, weights


class RouterStatistics:
    """Accumulates X^T X and X^T Y matrices incrementally (Eq. 3, 7).

    One instance per transformer block l.

    Args:
        hidden_dim: H — feature dimension.
        num_experts: D — number of domain experts / classes.
        dtype: accumulation dtype.
    """

    def __init__(self, hidden_dim: int, num_experts: int, dtype=mx.float32):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        # A_l = X^T X in R^{H x H}, b_l = X^T Y in R^{H x D}
        self.A = mx.zeros((hidden_dim, hidden_dim), dtype=dtype)
        self.B = mx.zeros((hidden_dim, num_experts), dtype=dtype)
        self.n_tokens = 0

    def update(self, features: mx.array, domain_id: int) -> None:
        """Accumulate one batch of features for a given domain.

        Args:
            features: (N, H) — hidden states before MoE block l.
                      N = batch_size * seq_len (already flattened).
            domain_id: integer domain label d in [0, D).

        Implements Eq. 7:
            A_l <- A_l + F_{d,<l}(X_i)^T F_{d,<l}(X_i)
            b_l <- b_l + F_{d,<l}(X_i)^T Y_i
        """
        # features: (N, H)
        features = features.astype(self.A.dtype)
        n = features.shape[0]

        # A_l += features^T features
        self.A = self.A + features.T @ features

        # Y_i is one-hot: only column `domain_id` gets features^T 1
        # b_l[:,d] += sum of features along N axis
        col_update = mx.sum(features, axis=0)  # (H,)
        # Scatter into the correct column
        self.B = self.B.at[:, domain_id].add(col_update)

        self.n_tokens += n


def solve_ridge(
    stats: RouterStatistics,
    lam: float = 0.1,
    column_normalize: bool = True,
) -> mx.array:
    """Compute optimal router weights via ridge regression closed-form.

    W* = (A + lambda I)^{-1} b          (Eq. 2 / Eq. 6)
    W*[:,c] = W*[:,c] / ||W*[:,c]||      (column normalization, Sec 2.3)

    Args:
        stats: accumulated RouterStatistics for one layer.
        lam: Tikhonov regularization parameter.
        column_normalize: whether to L2-normalize each column of W*.

    Returns:
        W*: (H, D) optimal router weight matrix.
    """
    H = stats.hidden_dim
    # (A + lambda I)
    regularized = stats.A + lam * mx.eye(H, dtype=stats.A.dtype)

    # W* = (A + lambda I)^{-1} b
    # Use mx.linalg.solve for numerical stability: solve (A + lam I) W* = b
    W_star = mx.linalg.solve(regularized, stats.B)

    if column_normalize:
        # W*[:,c] = W*[:,c] / ||W*[:,c]||
        col_norms = mx.linalg.norm(W_star, axis=0, keepdims=True)
        col_norms = mx.maximum(col_norms, 1e-8)  # avoid division by zero
        W_star = W_star / col_norms

    mx.eval(W_star)
    return W_star


def extract_router_weights(
    model_fn,
    domain_datasets: list[list[mx.array]],
    hidden_dim: int,
    num_layers: int,
    num_experts: int,
    lam: float = 0.1,
    column_normalize: bool = True,
) -> list[mx.array]:
    """Full DUME router extraction pipeline (Algorithm 1).

    1. Initialize A_l, b_l for each layer l.
    2. For each domain d, forward each batch through the model,
       collecting hidden states before each MoE block.
    3. Solve ridge regression for each layer.

    Args:
        model_fn: callable(input_ids, domain_id) -> list of (N, H) hidden
                  states, one per MoE layer. During extraction, each domain's
                  tokens are routed deterministically to that domain's expert.
        domain_datasets: list of D datasets, each a list of input_ids batches
                         as mx.array.
        hidden_dim: H.
        num_layers: L — number of MoE blocks (transformer layers).
        num_experts: D.
        lam: ridge regression lambda.
        column_normalize: whether to column-normalize W*.

    Returns:
        List of L weight matrices, each (H, D).
    """
    # Step 1: Initialize statistics for each layer
    stats = [
        RouterStatistics(hidden_dim, num_experts) for _ in range(num_layers)
    ]

    # Step 2: Forward all domain data, accumulate A_l, b_l
    for domain_id, dataset in enumerate(domain_datasets):
        for batch in dataset:
            # model_fn returns hidden states before each MoE block
            hidden_states_per_layer = model_fn(batch, domain_id)
            for l, h in enumerate(hidden_states_per_layer):
                # Flatten to (N, H) if needed
                if h.ndim == 3:
                    h = h.reshape(-1, h.shape[-1])
                stats[l].update(h, domain_id)
            mx.eval(stats[0].A)  # force eval at loop boundary

    # Step 3: Solve for each layer
    weights = []
    for l in range(num_layers):
        W = solve_ridge(stats[l], lam=lam, column_normalize=column_normalize)
        weights.append(W)

    return weights
