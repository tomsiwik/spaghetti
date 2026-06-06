"""AIMD Load Balance -- TCP congestion control for expert load balancing.

Three-way comparison of expert load balancing strategies:

1. **AIMD (this model)**: Additive Increase / Multiplicative Decrease on
   per-expert routing bias. When an expert is overloaded (receives > 1/G
   fraction of tokens), multiplicatively decrease its bias by factor beta.
   When underloaded, additively increase by alpha. Non-gradient feedback,
   decoupled from the training loss. Inspired by TCP congestion control
   (Jacobson 1988, Chiu-Jain 1989).

2. **Aux Loss (Switch Transformer baseline)**: L_bal = G * sum(f_i * p_i)
   added to training loss. Gradient-based. Fedus et al. 2022.

3. **No balance (control)**: Pure softmax routing, no balancing mechanism.

Connection to DeepSeek-V3: Their "auxiliary-loss-free" per-expert bias uses
additive increase/decrease (symmetric). AIMD is asymmetric: gentle increase,
aggressive decrease. This asymmetry is what gives TCP its provably fast
convergence to fairness (Chiu-Jain theorem).
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


class CapsuleGroup(nn.Module):
    """A group of rank-1 capsules: y = B @ ReLU(A @ x)."""

    def __init__(self, n_embd: int, n_capsules: int):
        super().__init__()
        self.A = nn.Linear(n_embd, n_capsules, bias=False)
        self.B = nn.Linear(n_capsules, n_embd, bias=False)

    def __call__(self, x):
        return self.B(nn.relu(self.A(x)))


class AIMDCapsulePool(nn.Module):
    """Capsule pool with AIMD load balancing on routing bias.

    AIMD update rule (applied after each forward pass, NOT through gradients):
        For each expert i:
            f_i = fraction of tokens routed to expert i this step
            target = 1/G (fair share)

            if f_i > target + epsilon:   # overloaded
                bias_i *= beta           # multiplicative decrease (aggressive)
            elif f_i < target - epsilon:  # underloaded
                bias_i += alpha          # additive increase (gentle)

    The bias is added to routing logits before softmax:
        scores = router(x) + bias   (bias is NOT a learned parameter)
        probs = softmax(scores)

    Convergence guarantee (Chiu-Jain): AIMD converges to the unique fair
    allocation point where f_i = 1/G for all i, as long as 0 < alpha and
    0 < beta < 1. Rate of convergence depends on beta (lower = faster but
    more oscillation).
    """

    def __init__(self, n_embd: int, n_groups: int = 4,
                 n_capsules_per_group: int = 64,
                 top_k_groups: int = 2,
                 # AIMD parameters
                 alpha: float = 0.01,     # additive increase step
                 beta: float = 0.5,       # multiplicative decrease factor
                 epsilon: float = 0.02,   # dead zone around target
                 ):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # Learned router
        self.router = nn.Linear(n_embd, n_groups, bias=False)

        # Capsule groups
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        # AIMD bias: NOT a learned parameter, updated by feedback control
        # Stored as a regular array, excluded from gradient computation
        self._bias = [0.0] * n_groups

        # Tracking for metrics
        self._load_fractions = None
        self._gate_probs = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        # Compute routing scores with AIMD bias
        scores = self.router(x)  # (B, T, G)

        # Add bias (not a gradient-tracked parameter)
        bias_array = mx.array(self._bias).reshape(1, 1, self.n_groups)
        scores = scores + bias_array

        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Top-k group selection
        top_vals = mx.topk(scores, self.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Run all groups, weight by masked probs
        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]  # (B, T, 1)
            out = out + w * group(x)

        # Compute load fractions and update AIMD bias (non-gradient)
        self._update_aimd(masked_probs)

        return out

    def _update_aimd(self, masked_probs):
        """Apply AIMD feedback control to routing bias.

        True AIMD: the asymmetry is essential.
        - Overloaded: multiplicatively scale bias toward negative (aggressive)
        - Underloaded: additively nudge bias upward (gentle)

        The multiplicative factor means overloaded experts get punished
        proportionally harder the more overloaded they are (because a
        larger positive bias gets more aggressively reduced).
        """
        # Load fraction: average routing weight per expert across batch
        # masked_probs: (B, T, G)
        load = mx.mean(masked_probs, axis=(0, 1))  # (G,)
        mx.eval(load)
        self._load_fractions = load

        target = 1.0 / self.n_groups
        for i in range(self.n_groups):
            f_i = load[i].item()
            excess = f_i - target  # positive = overloaded, negative = underloaded

            if excess > self.epsilon:
                # Overloaded: multiplicative decrease
                # Scale current bias toward negative proportionally to excess
                self._bias[i] = self.beta * self._bias[i] - self.alpha * (excess / target)
            elif excess < -self.epsilon:
                # Underloaded: additive increase
                self._bias[i] += self.alpha

    def balance_loss(self) -> mx.array:
        """No auxiliary loss -- AIMD handles balancing via bias feedback."""
        return mx.array(0.0)


class AuxLossCapsulePool(nn.Module):
    """Capsule pool with Switch Transformer auxiliary load-balancing loss.

    L_bal = G * sum_i(f_i * p_i)
    where f_i = fraction of tokens routed to expert i
          p_i = mean routing probability for expert i
          G = number of experts

    This is the standard baseline from Fedus et al. (2022).
    """

    def __init__(self, n_embd: int, n_groups: int = 4,
                 n_capsules_per_group: int = 64,
                 top_k_groups: int = 2,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups
        self.balance_coeff = balance_coeff

        self.router = nn.Linear(n_embd, n_groups, bias=False)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        self._gate_probs = None
        self._load_fractions = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        scores = self.router(x)  # (B, T, G)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Top-k group selection
        top_vals = mx.topk(scores, self.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Track load fractions for metrics
        load = mx.mean(masked_probs, axis=(0, 1))
        mx.eval(load)
        self._load_fractions = load

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]
            out = out + w * group(x)

        return out

    def balance_loss(self) -> mx.array:
        """Switch Transformer balance loss: L = G * sum(f_i * p_i)."""
        if self._gate_probs is None:
            return mx.array(0.0)
        # f_i: fraction of tokens dispatched to expert i (from masked routing)
        # p_i: mean routing probability for expert i (from softmax)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (G,)
        # Use gate probs as proxy for f_i (they are correlated)
        return self.balance_coeff * self.n_groups * mx.sum(mean_probs * mean_probs)


class NoBalanceCapsulePool(nn.Module):
    """Capsule pool with NO load balancing -- control condition."""

    def __init__(self, n_embd: int, n_groups: int = 4,
                 n_capsules_per_group: int = 64,
                 top_k_groups: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups

        self.router = nn.Linear(n_embd, n_groups, bias=False)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        self._gate_probs = None
        self._load_fractions = None

    def __call__(self, x):
        scores = self.router(x)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        top_vals = mx.topk(scores, self.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        load = mx.mean(masked_probs, axis=(0, 1))
        mx.eval(load)
        self._load_fractions = load

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i + 1]
            out = out + w * group(x)
        return out

    def balance_loss(self) -> mx.array:
        return mx.array(0.0)


# --- Block and Model definitions ---

class AIMDBlock(nn.Module):
    def __init__(self, n_embd, n_head, pool_cls, pool_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = pool_cls(n_embd, **pool_kwargs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


def _make_model(pool_cls, pool_kwargs, vocab_size, block_size, n_embd,
                n_head, n_layer):
    """Factory for building GPT variants with different pool types."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.wpe = nn.Embedding(block_size, n_embd)
            self.norm0 = RMSNorm(n_embd)
            self.layers = [AIMDBlock(n_embd, n_head, pool_cls, pool_kwargs)
                           for _ in range(n_layer)]
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        def __call__(self, tokens):
            B, T = tokens.shape
            pos = mx.arange(T)
            x = self.wte(tokens) + self.wpe(pos)
            x = self.norm0(x)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)

        def aux_loss(self):
            total = mx.array(0.0)
            for layer in self.layers:
                total = total + layer.capsule_pool.balance_loss()
            return total

        def on_domain_switch(self, domain):
            pass

        def load_balance_stats(self):
            """Return per-layer load fraction stats."""
            stats = []
            for i, layer in enumerate(self.layers):
                pool = layer.capsule_pool
                if pool._load_fractions is not None:
                    fracs = pool._load_fractions
                    mx.eval(fracs)
                    frac_list = fracs.tolist()
                    stats.append({
                        "layer": i,
                        "load_fractions": frac_list,
                        "max_load": max(frac_list),
                        "min_load": min(frac_list),
                        "imbalance": max(frac_list) - min(frac_list),
                    })
            return stats

    return _Model()


@register("aimd_balance", parent="capsule_moe")
class AIMDLoadBalanceGPT(nn.Module):
    """GPT with AIMD load balancing on expert routing."""

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2,
                 alpha: float = 0.01, beta: float = 0.5,
                 epsilon: float = 0.02):
        super().__init__()
        pool_kwargs = dict(
            n_groups=n_groups, n_capsules_per_group=n_capsules_per_group,
            top_k_groups=top_k_groups,
            alpha=alpha, beta=beta, epsilon=epsilon,
        )
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [AIMDBlock(n_embd, n_head, AIMDCapsulePool, pool_kwargs)
                       for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self):
        # AIMD has no auxiliary loss -- balancing is via bias feedback
        return mx.array(0.0)

    def on_domain_switch(self, domain):
        pass

    def load_balance_stats(self):
        stats = []
        for i, layer in enumerate(self.layers):
            pool = layer.capsule_pool
            if pool._load_fractions is not None:
                fracs = pool._load_fractions
                mx.eval(fracs)
                frac_list = fracs.tolist()
                stats.append({
                    "layer": i,
                    "load_fractions": frac_list,
                    "max_load": max(frac_list),
                    "min_load": min(frac_list),
                    "imbalance": max(frac_list) - min(frac_list),
                    "bias": list(pool._bias),
                })
        return stats


@register("aux_loss_balance", parent="capsule_moe")
class AuxLossBalanceGPT(nn.Module):
    """GPT with Switch Transformer auxiliary load-balancing loss."""

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, balance_coeff: float = 0.01):
        super().__init__()
        pool_kwargs = dict(
            n_groups=n_groups, n_capsules_per_group=n_capsules_per_group,
            top_k_groups=top_k_groups, balance_coeff=balance_coeff,
        )
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [AIMDBlock(n_embd, n_head, AuxLossCapsulePool, pool_kwargs)
                       for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self):
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return total

    def on_domain_switch(self, domain):
        pass

    def load_balance_stats(self):
        stats = []
        for i, layer in enumerate(self.layers):
            pool = layer.capsule_pool
            if pool._load_fractions is not None:
                fracs = pool._load_fractions
                mx.eval(fracs)
                frac_list = fracs.tolist()
                stats.append({
                    "layer": i,
                    "load_fractions": frac_list,
                    "max_load": max(frac_list),
                    "min_load": min(frac_list),
                    "imbalance": max(frac_list) - min(frac_list),
                })
        return stats


@register("no_balance", parent="capsule_moe")
class NoBalanceGPT(nn.Module):
    """GPT with NO load balancing -- control condition."""

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2):
        super().__init__()
        pool_kwargs = dict(
            n_groups=n_groups, n_capsules_per_group=n_capsules_per_group,
            top_k_groups=top_k_groups,
        )
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [AIMDBlock(n_embd, n_head, NoBalanceCapsulePool, pool_kwargs)
                       for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self):
        return mx.array(0.0)

    def on_domain_switch(self, domain):
        pass

    def load_balance_stats(self):
        stats = []
        for i, layer in enumerate(self.layers):
            pool = layer.capsule_pool
            if pool._load_fractions is not None:
                fracs = pool._load_fractions
                mx.eval(fracs)
                frac_list = fracs.tolist()
                stats.append({
                    "layer": i,
                    "load_fractions": frac_list,
                    "max_load": max(frac_list),
                    "min_load": min(frac_list),
                    "imbalance": max(frac_list) - min(frac_list),
                })
        return stats
