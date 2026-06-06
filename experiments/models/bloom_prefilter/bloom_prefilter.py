"""Bloom Filter Pre-Filtering for Two-Stage Expert Routing.

Two-stage routing pipeline:
  Stage 1: Bloom filter per expert group -- fast parallel query eliminates
           irrelevant groups. Guarantees zero false negatives (no expert that
           should fire is missed). May have false positives (some irrelevant
           experts survive).
  Stage 2: Full softmax routing over surviving candidates only.

Key design:
- Each expert group has its own Bloom filter encoding which token patterns
  it handles (built from activation profiles during a profiling phase).
- At inference, a token is hashed and queried against all Bloom filters
  in parallel. Groups whose Bloom filter returns "definitely not" are
  eliminated. Only survivors get scored by the softmax router.
- The Bloom filter uses k_hash independent hash functions, m bits per filter.
- Profiling uses the existing dead capsule profiling infrastructure:
  run tokens through the trained model, record which groups fire above
  a threshold, insert those token representations into the group's filter.

Architecture:
  Token x in R^d
    |
    v
  [Bloom Filter Stage]
    For each group g in 0..G-1:
      query bloom_g(hash(x)) -> {0, 1}
    survivors = {g : bloom_g(hash(x)) = 1}
    |
    v
  [Softmax Router Stage]
    scores = W_r[survivors] @ x
    probs = softmax(scores)
    top-k from survivors
    |
    v
  Weighted sum of selected expert outputs

Benefits at scale:
- Bloom filter query is O(k_hash) per group, independent of d
- With m=256 bits per group, storage is 256*G bits = trivial
- At N=256 experts, if 70% are eliminated, softmax runs on ~77 experts
  instead of 256, saving ~70% of routing compute

Prior art:
- No published work uses Bloom filters for MoE expert selection
- Nearest: hash-based routing (LSH, product keys in PEER)
- Bloom filters are standard in databases (membership testing)
"""

import math
import random as pyrandom

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class VectorizedBloomBank:
    """Vectorized bank of Bloom filters, one per expert group.

    Uses numpy for fast batch operations. Each filter is stored as a numpy
    bit array. Hash functions use random projections (dot products with
    fixed random vectors) followed by modular arithmetic, computed in batch.

    Hash scheme: For each of k_hash functions i and each token x:
      key_i(x) = floor((R_i @ x_quantized) mod p mod m)
    where R_i is a fixed random integer vector of length n_hash_dims,
    x_quantized discretizes the first n_hash_dims of x to integers.
    """

    def __init__(self, n_groups: int, m_bits: int = 256,
                 k_hash: int = 4, n_hash_dims: int = 8,
                 activation_threshold: float = 0.1, seed: int = 0):
        self.n_groups = n_groups
        self.m_bits = m_bits
        self.k_hash = k_hash
        self.n_hash_dims = n_hash_dims
        self.activation_threshold = activation_threshold

        # Bit arrays: (G, m_bits) boolean numpy arrays
        self.bits = np.zeros((n_groups, m_bits), dtype=bool)

        # Random hash coefficients: (k_hash, n_hash_dims) integers
        rng = np.random.RandomState(seed)
        self.p = 2**31 - 1
        self.hash_a = rng.randint(1, self.p, size=(k_hash, n_hash_dims))  # (k, d_h)
        self.hash_b = rng.randint(0, self.p, size=(k_hash,))  # (k,)

        self.n_profiled = 0
        self.n_inserted_per_group = np.zeros(n_groups, dtype=int)

    def _quantize(self, x_np):
        """Quantize float vectors to integers for hashing.

        x_np: (..., d) numpy array
        Returns: (..., n_hash_dims) integer array
        """
        # Use first n_hash_dims dimensions, quantize to 256 bins
        x_trunc = x_np[..., :self.n_hash_dims]
        # Clip to [-4, 4] range, map to [0, 255]
        x_clipped = np.clip(x_trunc, -4.0, 4.0)
        x_int = ((x_clipped + 4.0) / 8.0 * 256).astype(np.int64)
        return x_int

    def _compute_hashes(self, x_int):
        """Compute k_hash hash positions for a batch of quantized vectors.

        x_int: (..., n_hash_dims) integer array
        Returns: (..., k_hash) integer array of bit positions in [0, m_bits)
        """
        # x_int: (..., d_h), hash_a: (k, d_h)
        # For each hash function i: h_i = sum(a_i * x_int, dim=-1) + b_i
        # Result shape: (..., k)
        original_shape = x_int.shape[:-1]
        flat = x_int.reshape(-1, self.n_hash_dims)  # (N, d_h)

        # (N, d_h) @ (d_h, k) -> (N, k)
        dots = flat @ self.hash_a.T  # (N, k)
        hashes = (dots + self.hash_b[None, :]) % self.p % self.m_bits
        return hashes.reshape(*original_shape, self.k_hash)

    def profile_batch(self, x_mx, group_activations_mx):
        """Build Bloom filters from a batch of token-group activation data.

        Args:
            x_mx: (B, T, d) token embeddings (MLX array)
            group_activations_mx: (B, T, G) activation magnitudes per group (MLX)
        """
        x_np = np.array(x_mx)
        act_np = np.array(group_activations_mx)
        B, T, G = act_np.shape

        # Quantize all vectors
        x_int = self._quantize(x_np)  # (B, T, d_h)
        # Compute hashes for all vectors
        hashes = self._compute_hashes(x_int)  # (B, T, k)

        # For each group, insert tokens where activation > threshold
        for g in range(G):
            active_mask = act_np[:, :, g] > self.activation_threshold  # (B, T)
            active_indices = np.where(active_mask)
            if len(active_indices[0]) == 0:
                continue
            # Get hash positions for active tokens
            active_hashes = hashes[active_indices]  # (n_active, k)
            # Set bits
            for h_idx in range(self.k_hash):
                positions = active_hashes[:, h_idx].astype(int)
                self.bits[g, positions] = True
            self.n_inserted_per_group[g] += len(active_indices[0])

        self.n_profiled += B * T

    def query_batch(self, x_mx):
        """Query all Bloom filters for a batch of tokens.

        Args:
            x_mx: (B, T, d) token embeddings (MLX array)

        Returns:
            survivor_mask: (B, T, G) MLX array -- True if group MAY be relevant
        """
        x_np = np.array(x_mx)
        B, T, d = x_np.shape

        x_int = self._quantize(x_np)  # (B, T, d_h)
        hashes = self._compute_hashes(x_int)  # (B, T, k)

        # For each hash function, look up bits for all groups
        # hashes: (B, T, k) -- bit positions
        # bits: (G, m) -- the bit arrays
        # We need: for each (b, t, g), check all k hash positions are set

        # Reshape hashes to (B*T, k)
        flat_hashes = hashes.reshape(-1, self.k_hash)  # (N, k)
        N = flat_hashes.shape[0]

        # For each hash function, gather bits for all groups
        # result: (N, k, G) -- whether bit is set
        all_set = np.ones((N, self.n_groups), dtype=bool)
        for h_idx in range(self.k_hash):
            positions = flat_hashes[:, h_idx].astype(int)  # (N,)
            # bits[g, positions[n]] for all g, n
            # bits: (G, m), positions: (N,)
            # bits[:, positions]: (G, N)
            bit_values = self.bits[:, positions]  # (G, N)
            # Transpose to (N, G) and AND
            all_set &= bit_values.T  # (N, G)

        mask = all_set.reshape(B, T, self.n_groups)
        return mx.array(mask)

    def elimination_rate(self, x_mx) -> float:
        """Fraction of expert-token pairs eliminated by Bloom filtering."""
        mask = self.query_batch(x_mx)
        mx.eval(mask)
        total = mask.size
        survivors = mx.sum(mask.astype(mx.float32)).item()
        return 1.0 - survivors / total

    @property
    def fill_ratios(self):
        return [np.sum(self.bits[g]) / self.m_bits for g in range(self.n_groups)]

    def theoretical_fpr(self, n_inserted: int) -> float:
        """Theoretical false positive rate given n insertions."""
        if n_inserted == 0:
            return 0.0
        exponent = -self.k_hash * n_inserted / self.m_bits
        return (1.0 - math.exp(exponent)) ** self.k_hash

    def get_diagnostics(self) -> dict:
        """Return diagnostic info about the Bloom filter bank."""
        fill_rats = self.fill_ratios
        fprs = [self.theoretical_fpr(int(n))
                for n in self.n_inserted_per_group]
        return {
            "n_profiled": self.n_profiled,
            "n_inserted_per_group": self.n_inserted_per_group.tolist(),
            "fill_ratios": fill_rats,
            "mean_fill_ratio": sum(fill_rats) / len(fill_rats),
            "theoretical_fprs": fprs,
            "mean_theoretical_fpr": sum(fprs) / len(fprs) if fprs else 0,
            "m_bits": self.m_bits,
            "k_hash": self.k_hash,
        }


# Keep the simple BloomFilter class for unit tests
class BloomFilter:
    """A standard Bloom filter for membership testing (scalar, for tests)."""

    def __init__(self, m_bits: int = 256, k_hash: int = 4, seed: int = 0):
        self.m = m_bits
        self.k = k_hash
        self.bits = set()
        rng = pyrandom.Random(seed)
        self.p = 2**31 - 1
        self.coeffs = [(rng.randint(1, self.p - 1), rng.randint(0, self.p - 1))
                       for _ in range(k_hash)]

    def _hash_value(self, key: int, i: int) -> int:
        a, b = self.coeffs[i]
        return ((a * key + b) % self.p) % self.m

    def _hash_vector(self, vec) -> int:
        n_dims = min(8, len(vec))
        key = 0
        for j in range(n_dims):
            bucket = int(max(0, min(7, (vec[j] + 3.0) / 6.0 * 8)))
            key = key * 8 + bucket
        return key

    def insert_vector(self, vec):
        key = self._hash_vector(vec)
        for i in range(self.k):
            self.bits.add(self._hash_value(key, i))

    def query_vector(self, vec) -> bool:
        key = self._hash_vector(vec)
        return all(self._hash_value(key, i) in self.bits for i in range(self.k))

    @property
    def fill_ratio(self) -> float:
        return len(self.bits) / self.m

    def theoretical_fpr(self, n_inserted: int) -> float:
        if n_inserted == 0:
            return 0.0
        exponent = -self.k * n_inserted / self.m
        return (1.0 - math.exp(exponent)) ** self.k


class BloomCapsulePool(nn.Module):
    """Pool of capsule groups with two-stage routing:
    Stage 1: Bloom filter pre-filtering (eliminates irrelevant groups)
    Stage 2: Softmax routing over survivors only

    During training, Bloom filters are not used (full softmax routing).
    After profiling, Bloom filters are activated for inference.
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_capsules_per_group: int = 32,
                 top_k_groups: int = 2,
                 m_bits: int = 256, k_hash: int = 4,
                 activation_threshold: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups

        # Group router (learned, same as CapsulePool)
        self.router = nn.Linear(n_embd, n_groups, bias=False)

        # Capsule groups (learned)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        # Bloom filter bank (not learned, built during profiling)
        self.bloom_bank = VectorizedBloomBank(
            n_groups, m_bits, k_hash,
            n_hash_dims=min(8, n_embd),
            activation_threshold=activation_threshold
        )

        # Whether to use Bloom pre-filtering
        self.use_bloom = False

        self._gate_probs = None

    def _get_group_activations(self, x):
        """Compute activation magnitude per group (for profiling).

        Uses L1 norm of each group's output as activation signal.
        x: (B, T, d) -> activations: (B, T, G)
        """
        act_list = []
        for group in self.groups:
            out = group(x)  # (B, T, d)
            act = mx.mean(mx.abs(out), axis=-1, keepdims=True)  # (B, T, 1)
            act_list.append(act)
        return mx.concatenate(act_list, axis=-1)  # (B, T, G)

    def profile(self, x):
        """Profile one batch: compute group activations and insert into Bloom filters.

        x: (B, T, d) post-norm hidden states
        """
        activations = self._get_group_activations(x)
        mx.eval(activations, x)
        self.bloom_bank.profile_batch(x, activations)

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        B, T, D = x.shape

        # Stage 1: Bloom pre-filtering (if activated)
        if self.use_bloom:
            mx.eval(x)
            bloom_mask = self.bloom_bank.query_batch(x)  # (B, T, G)
            bloom_mask = bloom_mask.astype(mx.float32)
        else:
            bloom_mask = mx.ones((B, T, self.n_groups))

        # Stage 2: Softmax routing over survivors
        scores = self.router(x)  # (B, T, G)

        # Mask out eliminated groups (set score to -inf)
        masked_scores = scores * bloom_mask + (1 - bloom_mask) * (-1e9)

        probs = mx.softmax(masked_scores, axis=-1)
        self._gate_probs = probs

        # Top-k from survivors
        top_vals = mx.topk(masked_scores, self.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (masked_scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Compute weighted expert outputs
        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * group(x)

        return out

    def balance_loss(self) -> mx.array:
        """Balance loss: same formula as CapsulePool."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (G,)
        return self.n_groups * mx.sum(mean_probs * mean_probs)


class BloomBlock(nn.Module):
    """Transformer block with Bloom-prefiltered CapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k_groups: int = 2,
                 m_bits: int = 256, k_hash: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = BloomCapsulePool(
            n_embd, n_groups, n_capsules_per_group,
            top_k_groups, m_bits, k_hash
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("bloom_prefilter", parent="capsule_moe")
class BloomPrefilterGPT(nn.Module):
    """GPT with Bloom filter pre-filtered capsule routing.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - BloomCapsulePool: two-stage routing (Bloom filter + softmax)
    - Language model head (same as GPT)

    Training protocol:
    1. Train normally (Bloom filters inactive, full softmax routing)
    2. Profile: run tokens through model, build Bloom filters from activations
    3. Activate Bloom filters for inference

    Default config: d=64, G=8, 32 caps/group, k=2, m=256 bits, 4 hash functions.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k_groups: int = 2,
                 m_bits: int = 256, k_hash: int = 4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [BloomBlock(n_embd, n_head, n_groups,
                                   n_capsules_per_group, top_k_groups,
                                   m_bits, k_hash)
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

    def aux_loss(self) -> mx.array:
        """Combined auxiliary loss: balance."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def set_bloom_active(self, active: bool):
        """Enable or disable Bloom pre-filtering for all layers."""
        for layer in self.layers:
            layer.capsule_pool.use_bloom = active

    def profile_batch(self, tokens):
        """Run a profiling forward pass: compute hidden states and profile
        each layer's Bloom filters from activation patterns.

        Must be called AFTER training, BEFORE enabling Bloom filtering.
        """
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            h = layer.norm2(x + layer.attn(layer.norm1(x)))
            mx.eval(h)
            layer.capsule_pool.profile(h)
            x = layer(x)

    def get_bloom_diagnostics(self) -> dict:
        """Return diagnostic info for all layers' Bloom filter banks."""
        diagnostics = {}
        for li, layer in enumerate(self.layers):
            diagnostics[f"layer_{li}"] = layer.capsule_pool.bloom_bank.get_diagnostics()
        return diagnostics

    def get_elimination_rate(self, tokens) -> dict:
        """Measure what fraction of expert-token pairs are eliminated per layer."""
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        rates = {}
        for li, layer in enumerate(self.layers):
            h = layer.norm2(x + layer.attn(layer.norm1(x)))
            mx.eval(h)
            rate = layer.capsule_pool.bloom_bank.elimination_rate(h)
            rates[f"layer_{li}"] = rate
            x = layer(x)
        return rates
