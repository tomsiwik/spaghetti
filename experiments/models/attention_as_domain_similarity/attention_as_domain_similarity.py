#!/usr/bin/env python3
"""
Attention LoRA Cosine as Domain Similarity Predictor

Hypothesis: Attention LoRA cosine similarity reliably predicts semantic domain
overlap across many domain pairs. The attention-specific signal is STRONGER
than FFN cosine as a domain similarity predictor.

Prior finding (ffn_only_vs_all_modules): math-medical attention cos=0.85 vs
FFN cos=0.59. Attention amplifies domain correlation. But that was just ONE
pair out of 10. This experiment tests whether the pattern holds systematically.

Design:
  1. Build a micro transformer (not MLP -- needs attention!) with LoRA on
     both attention AND FFN layers.
  2. Generate 12 synthetic domains in 4 clusters with GRADUATED similarity:
     - Within-cluster domains share prototype transition matrices
     - Cross-cluster similarity varies (some clusters more related than others)
  3. Train 12 domain experts (LoRA on all modules)
  4. Compute three cosine matrices: attention-only, FFN-only, all-modules
  5. Define ground-truth semantic similarity from cluster structure
  6. Measure Spearman correlation between cosine matrices and ground truth
  7. Compare: which module type best predicts domain similarity?

Kill criteria:
  K1: Attention cosine does not correlate with semantic domain similarity
      (Spearman rho < 0.3 or p > 0.05)
  K2: Signal vanishes with more data (correlation drops as we add more domains)

Architecture: 2-layer transformer with single-head self-attention + FFN,
d=64, d_ff=256, rank=8 LoRA on Q, K, V, O projections AND FFN fc1/fc2.
Pure numpy. Runs in ~3 minutes.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ===========================================================================
# Constants
# ===========================================================================

VOCAB_SIZE = 32
CONTEXT_LEN = 16
D_MODEL = 64
D_FF = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 16
LORA_RANK = 8
LORA_ALPHA = 8

# 4 clusters of 3 domains each = 12 domains, 66 pairs
CLUSTERS = {
    'code':      ['python', 'javascript', 'rust'],
    'reasoning': ['math', 'logic', 'physics'],
    'knowledge': ['medical', 'law', 'history'],
    'creative':  ['poetry', 'fiction', 'comedy'],
}

# Graduated inter-cluster similarity (0=unrelated, 1=identical)
# This is the ground truth for the correlation test
CLUSTER_SIMILARITY = {
    ('code', 'code'):           1.0,
    ('reasoning', 'reasoning'): 1.0,
    ('knowledge', 'knowledge'): 1.0,
    ('creative', 'creative'):   1.0,
    ('code', 'reasoning'):      0.5,   # both use logical structures
    ('reasoning', 'code'):      0.5,
    ('code', 'knowledge'):      0.15,  # largely unrelated
    ('knowledge', 'code'):      0.15,
    ('code', 'creative'):       0.1,   # very different
    ('creative', 'code'):       0.1,
    ('reasoning', 'knowledge'): 0.4,   # stats/medical overlap
    ('knowledge', 'reasoning'): 0.4,
    ('reasoning', 'creative'):  0.1,   # largely unrelated
    ('creative', 'reasoning'):  0.1,
    ('knowledge', 'creative'):  0.2,   # some narrative overlap
    ('creative', 'knowledge'):  0.2,
}

DOMAIN_TO_CLUSTER = {}
for cluster, domains in CLUSTERS.items():
    for domain in domains:
        DOMAIN_TO_CLUSTER[domain] = cluster

ALL_DOMAINS = []
for cluster in ['code', 'reasoning', 'knowledge', 'creative']:
    ALL_DOMAINS.extend(CLUSTERS[cluster])


# ===========================================================================
# Ground truth similarity matrix
# ===========================================================================

def build_ground_truth_matrix():
    """Build N x N ground truth semantic similarity matrix.

    Within-cluster pairs get similarity 0.7 (same cluster but different domains).
    Self-similarity is 1.0 (excluded from correlation).
    Cross-cluster pairs use the CLUSTER_SIMILARITY table.

    Returns: (N, N) matrix, domain list
    """
    N = len(ALL_DOMAINS)
    sim = np.zeros((N, N))
    for i, d1 in enumerate(ALL_DOMAINS):
        for j, d2 in enumerate(ALL_DOMAINS):
            c1, c2 = DOMAIN_TO_CLUSTER[d1], DOMAIN_TO_CLUSTER[d2]
            if i == j:
                sim[i, j] = 1.0
            elif c1 == c2:
                sim[i, j] = 0.7  # within-cluster, different domains
            else:
                sim[i, j] = CLUSTER_SIMILARITY[(c1, c2)]
    return sim


# ===========================================================================
# Utilities
# ===========================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    """Approximate GeLU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_grad(x):
    """Numerical gradient of GeLU."""
    eps = 1e-5
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)


# ===========================================================================
# Model: Micro Transformer with LoRA on Attention + FFN
# ===========================================================================

class MicroTransformer:
    """2-layer transformer with single-head attention + FFN, LoRA on all modules.

    Architecture per layer:
      h' = h + Attention(h)   [Q, K, V, O projections with LoRA]
      h'' = h' + FFN(h')      [fc1, fc2 with LoRA]

    Attention is causal (lower-triangular mask).
    """

    def __init__(self, rng):
        scale = 0.02
        self.wte = rng.randn(VOCAB_SIZE, D_MODEL) * scale
        # Positional encoding (learned)
        self.wpe = rng.randn(CONTEXT_LEN, D_MODEL) * scale

        self.layers = []
        for _ in range(N_LAYERS):
            layer = {
                # Attention projections
                'Wq': rng.randn(D_MODEL, D_MODEL) * scale,
                'Wk': rng.randn(D_MODEL, D_MODEL) * scale,
                'Wv': rng.randn(D_MODEL, D_MODEL) * scale,
                'Wo': rng.randn(D_MODEL, D_MODEL) * scale,
                # FFN
                'W1': rng.randn(D_MODEL, D_FF) * scale,
                'b1': np.zeros(D_FF),
                'W2': rng.randn(D_FF, D_MODEL) * scale,
                'b2': np.zeros(D_MODEL),
            }
            self.layers.append(layer)
        self.W_out = rng.randn(D_MODEL, VOCAB_SIZE) * scale

    def forward(self, x_ids, lora=None):
        """Forward pass.
        x_ids: (B, T) integer token ids
        lora: dict with attention and FFN LoRA params per layer
        Returns: logits (B, V), intermediates for backprop
        """
        B, T = x_ids.shape
        sc = LORA_ALPHA / LORA_RANK

        # Embed
        emb = self.wte[x_ids] + self.wpe[:T]  # (B, T, d)

        intermediates = []
        h = emb

        for l, layer in enumerate(self.layers):
            inter = {'h_in': h.copy()}

            # --- Self-Attention ---
            Q = h @ layer['Wq']  # (B, T, d)
            K = h @ layer['Wk']
            V = h @ layer['Wv']
            if lora is not None:
                Q = Q + sc * (h @ lora['Aq'][l] @ lora['Bq'][l])
                K = K + sc * (h @ lora['Ak'][l] @ lora['Bk'][l])
                V = V + sc * (h @ lora['Av'][l] @ lora['Bv'][l])

            # Reshape for multi-head: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
            Q = Q.reshape(B, T, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            K = K.reshape(B, T, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            V = V.reshape(B, T, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)

            # Attention scores
            scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D_HEAD)  # (B, nh, T, T)
            # Causal mask
            mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
            scores = scores + mask
            attn = softmax(scores, axis=-1)  # (B, nh, T, T)

            # Attend
            out = attn @ V  # (B, nh, T, d_head)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)  # (B, T, d)

            # Output projection
            attn_out = out @ layer['Wo']  # (B, T, d)
            if lora is not None:
                attn_out = attn_out + sc * (out @ lora['Ao'][l] @ lora['Bo'][l])

            inter['Q'] = Q
            inter['K'] = K
            inter['V'] = V
            inter['attn'] = attn
            inter['attn_out_pre'] = out  # before Wo

            h = h + attn_out  # residual
            inter['h_after_attn'] = h.copy()

            # --- FFN ---
            z1 = h @ layer['W1'] + layer['b1']
            if lora is not None:
                z1 = z1 + sc * (h @ lora['A1'][l] @ lora['B1'][l])
            a1 = gelu(z1)

            z2 = a1 @ layer['W2'] + layer['b2']
            if lora is not None:
                z2 = z2 + sc * (a1 @ lora['A2'][l] @ lora['B2'][l])

            inter['z1'] = z1
            inter['a1'] = a1
            h = h + z2  # residual

            intermediates.append(inter)

        # Use last token for prediction (causal LM style)
        h_last = h[:, -1, :]  # (B, d)
        logits = h_last @ self.W_out  # (B, V)
        return logits, intermediates, h


def init_lora(rng):
    """Initialize LoRA parameters for attention (Q,K,V,O) and FFN (fc1,fc2).
    A: Kaiming init, B: zero init.
    """
    lora = {
        # Attention LoRA
        'Aq': [], 'Bq': [],
        'Ak': [], 'Bk': [],
        'Av': [], 'Bv': [],
        'Ao': [], 'Bo': [],
        # FFN LoRA
        'A1': [], 'B1': [],
        'A2': [], 'B2': [],
    }
    for _ in range(N_LAYERS):
        a_scale = np.sqrt(2.0 / D_MODEL)
        a_ff_scale = np.sqrt(2.0 / D_FF)
        # Attention
        lora['Aq'].append(rng.randn(D_MODEL, LORA_RANK) * a_scale)
        lora['Bq'].append(np.zeros((LORA_RANK, D_MODEL)))
        lora['Ak'].append(rng.randn(D_MODEL, LORA_RANK) * a_scale)
        lora['Bk'].append(np.zeros((LORA_RANK, D_MODEL)))
        lora['Av'].append(rng.randn(D_MODEL, LORA_RANK) * a_scale)
        lora['Bv'].append(np.zeros((LORA_RANK, D_MODEL)))
        lora['Ao'].append(rng.randn(D_MODEL, LORA_RANK) * a_scale)
        lora['Bo'].append(np.zeros((LORA_RANK, D_MODEL)))
        # FFN
        lora['A1'].append(rng.randn(D_MODEL, LORA_RANK) * a_scale)
        lora['B1'].append(np.zeros((LORA_RANK, D_FF)))
        lora['A2'].append(rng.randn(D_FF, LORA_RANK) * a_ff_scale)
        lora['B2'].append(np.zeros((LORA_RANK, D_MODEL)))
    return lora


def lora_to_delta_vectors(lora):
    """Extract separate attention, FFN, and combined delta vectors.

    Returns:
        attn_delta: flattened attention LoRA deltas (A@B for Q,K,V,O)
        ffn_delta: flattened FFN LoRA deltas (A@B for fc1, fc2)
        full_delta: concatenation of both
    """
    attn_parts = []
    ffn_parts = []
    for l in range(N_LAYERS):
        # Attention deltas
        attn_parts.append((lora['Aq'][l] @ lora['Bq'][l]).flatten())
        attn_parts.append((lora['Ak'][l] @ lora['Bk'][l]).flatten())
        attn_parts.append((lora['Av'][l] @ lora['Bv'][l]).flatten())
        attn_parts.append((lora['Ao'][l] @ lora['Bo'][l]).flatten())
        # FFN deltas
        ffn_parts.append((lora['A1'][l] @ lora['B1'][l]).flatten())
        ffn_parts.append((lora['A2'][l] @ lora['B2'][l]).flatten())

    attn_delta = np.concatenate(attn_parts)
    ffn_delta = np.concatenate(ffn_parts)
    full_delta = np.concatenate([attn_delta, ffn_delta])
    return attn_delta, ffn_delta, full_delta


# ===========================================================================
# Training
# ===========================================================================

def train_lora(model, data_x, data_y, rng, steps=300, lr=0.005,
               batch_size=32, log_every=100):
    """Train LoRA via SPSA (Simultaneous Perturbation Stochastic Approximation).

    Uses a single 2-forward-pass gradient estimate per step, perturbing ALL
    B matrices simultaneously. This is much faster than per-parameter
    perturbation while still producing domain-specific LoRA deltas.

    SPSA (Spall 1992): perturb all params with random signs, measure
    loss(+delta) - loss(-delta), divide by perturbation to get gradient estimate.
    Converges to true gradient in expectation.
    """
    lora = init_lora(rng)
    n_data = data_x.shape[0]
    losses = []

    # All B-matrix keys
    b_keys = ['Bq', 'Bk', 'Bv', 'Bo', 'B1', 'B2']
    eps = 5e-3  # perturbation size

    for step in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
        bx, by = data_x[idx], data_y[idx]
        B_sz = bx.shape[0]

        # Generate random perturbation directions for all B matrices
        perturbations = {}
        for key in b_keys:
            perturbations[key] = []
            for l in range(N_LAYERS):
                # Rademacher random variables (+1/-1)
                delta = rng.choice([-1.0, 1.0], size=lora[key][l].shape)
                perturbations[key].append(delta)

        # Perturb +
        for key in b_keys:
            for l in range(N_LAYERS):
                lora[key][l] = lora[key][l] + eps * perturbations[key][l]

        logits_p, _, _ = model.forward(bx, lora)
        probs_p = softmax(logits_p)
        loss_p = -np.mean(np.log(probs_p[np.arange(B_sz), by] + 1e-10))

        # Perturb - (double delta from +)
        for key in b_keys:
            for l in range(N_LAYERS):
                lora[key][l] = lora[key][l] - 2 * eps * perturbations[key][l]

        logits_m, _, _ = model.forward(bx, lora)
        probs_m = softmax(logits_m)
        loss_m = -np.mean(np.log(probs_m[np.arange(B_sz), by] + 1e-10))

        # Restore to original
        for key in b_keys:
            for l in range(N_LAYERS):
                lora[key][l] = lora[key][l] + eps * perturbations[key][l]

        # SPSA gradient estimate and update
        grad_scalar = (loss_p - loss_m) / (2 * eps)
        for key in b_keys:
            for l in range(N_LAYERS):
                # SPSA: g_i = (loss+ - loss-) / (2 * eps * delta_i)
                # For Rademacher: 1/delta_i = delta_i (since delta_i in {+1,-1})
                lora[key][l] -= lr * grad_scalar * perturbations[key][l]

        # Record loss at midpoint
        loss = (loss_p + loss_m) / 2
        losses.append(loss)

        if step % log_every == 0:
            print(f"    step {step:4d}: loss={loss:.4f}")

    if losses:
        print(f"    final loss: {losses[-1]:.4f}")
    return lora, losses[-1] if losses else float('inf')


# ===========================================================================
# Data Generation
# ===========================================================================

def generate_cluster_prototypes(rng, cross_cluster_distance=2.0):
    """Generate transition matrices for each cluster.

    Clusters with higher inter-cluster similarity in CLUSTER_SIMILARITY
    share more of their prototype structure.
    """
    # Start with a shared base
    base_logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * 0.5

    prototypes = {}
    cluster_names = ['code', 'reasoning', 'knowledge', 'creative']

    # Each cluster gets a unique perturbation
    cluster_perturbations = {}
    for i, name in enumerate(cluster_names):
        pert = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * cross_cluster_distance
        # Cluster-specific character group bias
        group_start = (i * VOCAB_SIZE) // len(cluster_names)
        group_end = ((i + 1) * VOCAB_SIZE) // len(cluster_names)
        pert[:, group_start:group_end] += 2.0
        pert += np.eye(VOCAB_SIZE) * 0.5
        cluster_perturbations[name] = pert

    # Build prototypes with inter-cluster blending based on similarity
    for name in cluster_names:
        logits = base_logits.copy() + cluster_perturbations[name]
        # Blend with similar clusters
        for other_name in cluster_names:
            if other_name != name:
                sim = CLUSTER_SIMILARITY.get((name, other_name), 0.0)
                logits += sim * 0.3 * cluster_perturbations[other_name]
        prototypes[name] = softmax(logits, axis=-1)

    return prototypes


def generate_domain_data(rng, prototypes, domain_name, cluster_name,
                          n_sequences=300, noise_scale=0.15):
    """Generate (x, y) pairs for next-token prediction."""
    prototype = prototypes[cluster_name]
    noise = rng.randn(*prototype.shape) * noise_scale

    # Domain-specific perturbation
    domain_seed = hash(domain_name) % (2**31)
    domain_rng = np.random.RandomState(domain_seed)
    domain_bias = domain_rng.randn(*prototype.shape) * noise_scale * 0.5

    # Domain signature characters
    domain_idx = ALL_DOMAINS.index(domain_name)
    for j in range(3):
        c = (domain_idx * 2 + j) % VOCAB_SIZE
        domain_bias[:, c] += 0.3

    transition = softmax(np.log(prototype + 1e-10) + noise + domain_bias, axis=-1)

    # Generate sequences
    sequences = []
    for _ in range(n_sequences):
        seq = [rng.randint(VOCAB_SIZE)]
        for t in range(CONTEXT_LEN):
            prev = seq[-1]
            next_tok = rng.choice(VOCAB_SIZE, p=transition[prev])
            seq.append(next_tok)
        sequences.append(seq)

    seqs = np.array(sequences)
    x = seqs[:, :-1]  # (n, T)
    y = seqs[:, -1]   # (n,) -- predict last token
    return x, y


# ===========================================================================
# Analysis
# ===========================================================================

def compute_cosine_matrices(all_deltas):
    """Compute pairwise cosine similarity matrices for attn, FFN, and full deltas.

    all_deltas: list of (attn_delta, ffn_delta, full_delta) tuples
    Returns: (attn_cos_matrix, ffn_cos_matrix, full_cos_matrix)
    """
    N = len(all_deltas)
    attn_mat = np.zeros((N, N))
    ffn_mat = np.zeros((N, N))
    full_mat = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            a_i, f_i, c_i = all_deltas[i]
            a_j, f_j, c_j = all_deltas[j]

            # Cosine similarity
            def cos(u, v):
                n1, n2 = np.linalg.norm(u), np.linalg.norm(v)
                if n1 < 1e-12 or n2 < 1e-12:
                    return 0.0
                return float(np.dot(u, v) / (n1 * n2))

            attn_mat[i, j] = cos(a_i, a_j)
            ffn_mat[i, j] = cos(f_i, f_j)
            full_mat[i, j] = cos(c_i, c_j)

    return attn_mat, ffn_mat, full_mat


def extract_upper_triangle(matrix):
    """Extract upper triangle (i < j) as a flat vector, excluding diagonal."""
    N = matrix.shape[0]
    indices = np.triu_indices(N, k=1)
    return matrix[indices]


def analyze_correlations(attn_cos, ffn_cos, full_cos, gt_sim):
    """Compute Spearman correlations between cosine matrices and ground truth.

    Uses absolute cosine values (|cos|) since we care about similarity magnitude.
    """
    # Extract upper triangles
    gt_vec = extract_upper_triangle(gt_sim)
    attn_vec = extract_upper_triangle(np.abs(attn_cos))
    ffn_vec = extract_upper_triangle(np.abs(ffn_cos))
    full_vec = extract_upper_triangle(np.abs(full_cos))

    # Spearman correlation
    attn_rho, attn_p = stats.spearmanr(gt_vec, attn_vec)
    ffn_rho, ffn_p = stats.spearmanr(gt_vec, ffn_vec)
    full_rho, full_p = stats.spearmanr(gt_vec, full_vec)

    # Also Pearson for comparison
    attn_r, attn_pr = stats.pearsonr(gt_vec, attn_vec)
    ffn_r, ffn_pr = stats.pearsonr(gt_vec, ffn_vec)
    full_r, full_pr = stats.pearsonr(gt_vec, full_vec)

    return {
        'spearman': {
            'attn': {'rho': float(attn_rho), 'p': float(attn_p)},
            'ffn':  {'rho': float(ffn_rho),  'p': float(ffn_p)},
            'full': {'rho': float(full_rho), 'p': float(full_p)},
        },
        'pearson': {
            'attn': {'r': float(attn_r), 'p': float(attn_pr)},
            'ffn':  {'r': float(ffn_r),  'p': float(ffn_pr)},
            'full': {'r': float(full_r), 'p': float(full_pr)},
        },
        'n_pairs': len(gt_vec),
        'gt_unique_values': len(np.unique(gt_vec)),
    }


def analyze_within_vs_cross(attn_cos, ffn_cos, full_cos):
    """Analyze within-cluster vs cross-cluster cosine, broken down by module type."""
    N = len(ALL_DOMAINS)
    within_attn, cross_attn = [], []
    within_ffn, cross_ffn = [], []
    within_full, cross_full = [], []

    for i in range(N):
        for j in range(i + 1, N):
            c_i = DOMAIN_TO_CLUSTER[ALL_DOMAINS[i]]
            c_j = DOMAIN_TO_CLUSTER[ALL_DOMAINS[j]]
            if c_i == c_j:
                within_attn.append(abs(attn_cos[i, j]))
                within_ffn.append(abs(ffn_cos[i, j]))
                within_full.append(abs(full_cos[i, j]))
            else:
                cross_attn.append(abs(attn_cos[i, j]))
                cross_ffn.append(abs(ffn_cos[i, j]))
                cross_full.append(abs(full_cos[i, j]))

    return {
        'within_cluster': {
            'attn_mean': float(np.mean(within_attn)),
            'ffn_mean': float(np.mean(within_ffn)),
            'full_mean': float(np.mean(within_full)),
            'n': len(within_attn),
        },
        'cross_cluster': {
            'attn_mean': float(np.mean(cross_attn)),
            'ffn_mean': float(np.mean(cross_ffn)),
            'full_mean': float(np.mean(cross_full)),
            'n': len(cross_attn),
        },
        'attn_ratio': float(np.mean(within_attn) / (np.mean(cross_attn) + 1e-10)),
        'ffn_ratio': float(np.mean(within_ffn) / (np.mean(cross_ffn) + 1e-10)),
        'full_ratio': float(np.mean(within_full) / (np.mean(cross_full) + 1e-10)),
    }


def ablation_domain_count(all_deltas, gt_sim, domain_counts=[6, 8, 10, 12]):
    """Test K2: does the correlation hold as we add more domains?

    Subsample domains and recompute correlation to check for signal stability.
    """
    rng = np.random.RandomState(999)
    results = []
    N = len(ALL_DOMAINS)

    for n_dom in domain_counts:
        if n_dom > N:
            continue
        # Sample domains, keeping at least 2 per cluster
        trials = []
        for trial in range(5):
            indices = []
            for cluster, domains in CLUSTERS.items():
                # Take at least 1 domain per cluster (up to min(n_per, available))
                n_per = max(1, n_dom // len(CLUSTERS))
                cluster_indices = [ALL_DOMAINS.index(d) for d in domains]
                chosen = rng.choice(cluster_indices, size=min(n_per, len(cluster_indices)),
                                    replace=False).tolist()
                indices.extend(chosen)
            # Fill remaining slots
            remaining = [i for i in range(N) if i not in indices]
            n_extra = n_dom - len(indices)
            if n_extra > 0:
                indices.extend(rng.choice(remaining, size=n_extra, replace=False).tolist())
            indices = sorted(indices[:n_dom])

            # Extract sub-matrices
            sub_deltas = [all_deltas[i] for i in indices]
            sub_gt = gt_sim[np.ix_(indices, indices)]

            attn_cos, ffn_cos, full_cos = compute_cosine_matrices(sub_deltas)
            gt_vec = extract_upper_triangle(sub_gt)
            attn_vec = extract_upper_triangle(np.abs(attn_cos))

            if len(np.unique(gt_vec)) < 2:
                continue
            rho, p = stats.spearmanr(gt_vec, attn_vec)
            trials.append({'rho': float(rho), 'p': float(p)})

        if trials:
            results.append({
                'n_domains': n_dom,
                'n_pairs': n_dom * (n_dom - 1) // 2,
                'mean_rho': float(np.mean([t['rho'] for t in trials])),
                'std_rho': float(np.std([t['rho'] for t in trials])),
                'mean_p': float(np.mean([t['p'] for t in trials])),
            })

    return results


# ===========================================================================
# Main
# ===========================================================================

def run_experiment(seed=42):
    """Run the full experiment for one seed."""
    print(f"\n{'='*70}")
    print(f"  Attention as Domain Similarity Predictor (seed={seed})")
    print(f"{'='*70}")

    rng = np.random.RandomState(seed)
    t0 = time.time()

    # Build model
    print("\n[1] Building micro transformer...")
    model = MicroTransformer(rng)

    # Generate data
    print("\n[2] Generating synthetic domain data...")
    prototypes = generate_cluster_prototypes(rng)
    domain_data = {}
    for domain in ALL_DOMAINS:
        cluster = DOMAIN_TO_CLUSTER[domain]
        x, y = generate_domain_data(rng, prototypes, domain, cluster)
        domain_data[domain] = (x, y)
        print(f"    {domain:12s} ({cluster:10s}): {x.shape[0]} sequences")

    # Train LoRA for each domain
    print("\n[3] Training domain experts...")
    all_loras = {}
    all_deltas = []
    all_losses = {}
    for domain in ALL_DOMAINS:
        print(f"\n  Training {domain}...")
        x, y = domain_data[domain]
        lora, final_loss = train_lora(model, x, y, np.random.RandomState(seed + hash(domain) % 1000),
                                       steps=500, lr=0.005, batch_size=32, log_every=200)
        all_loras[domain] = lora
        attn_d, ffn_d, full_d = lora_to_delta_vectors(lora)
        all_deltas.append((attn_d, ffn_d, full_d))
        all_losses[domain] = final_loss
        print(f"    delta norms: attn={np.linalg.norm(attn_d):.4f}, "
              f"ffn={np.linalg.norm(ffn_d):.4f}, "
              f"full={np.linalg.norm(full_d):.4f}")

    # Compute cosine matrices
    print("\n[4] Computing cosine similarity matrices...")
    attn_cos, ffn_cos, full_cos = compute_cosine_matrices(all_deltas)
    gt_sim = build_ground_truth_matrix()

    # Correlations
    print("\n[5] Computing correlations...")
    correlations = analyze_correlations(attn_cos, ffn_cos, full_cos, gt_sim)
    print(f"    Spearman correlations with ground truth:")
    print(f"      Attention: rho={correlations['spearman']['attn']['rho']:.4f} "
          f"(p={correlations['spearman']['attn']['p']:.4e})")
    print(f"      FFN:       rho={correlations['spearman']['ffn']['rho']:.4f} "
          f"(p={correlations['spearman']['ffn']['p']:.4e})")
    print(f"      Full:      rho={correlations['spearman']['full']['rho']:.4f} "
          f"(p={correlations['spearman']['full']['p']:.4e})")

    # Within vs cross cluster analysis
    print("\n[6] Within-cluster vs cross-cluster analysis...")
    cluster_analysis = analyze_within_vs_cross(attn_cos, ffn_cos, full_cos)
    print(f"    Within-cluster mean |cos|:")
    print(f"      Attention: {cluster_analysis['within_cluster']['attn_mean']:.6f}")
    print(f"      FFN:       {cluster_analysis['within_cluster']['ffn_mean']:.6f}")
    print(f"      Full:      {cluster_analysis['within_cluster']['full_mean']:.6f}")
    print(f"    Cross-cluster mean |cos|:")
    print(f"      Attention: {cluster_analysis['cross_cluster']['attn_mean']:.6f}")
    print(f"      FFN:       {cluster_analysis['cross_cluster']['ffn_mean']:.6f}")
    print(f"      Full:      {cluster_analysis['cross_cluster']['full_mean']:.6f}")
    print(f"    Within/Cross ratio:")
    print(f"      Attention: {cluster_analysis['attn_ratio']:.2f}x")
    print(f"      FFN:       {cluster_analysis['ffn_ratio']:.2f}x")
    print(f"      Full:      {cluster_analysis['full_ratio']:.2f}x")

    # K2 ablation: stability across domain counts
    print("\n[7] K2 ablation: correlation stability across domain counts...")
    k2_ablation = ablation_domain_count(all_deltas, gt_sim)
    for ab in k2_ablation:
        print(f"    N={ab['n_domains']:2d} ({ab['n_pairs']:2d} pairs): "
              f"rho={ab['mean_rho']:.4f} +/- {ab['std_rho']:.4f} "
              f"(p={ab['mean_p']:.4e})")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        'seed': seed,
        'correlations': correlations,
        'cluster_analysis': cluster_analysis,
        'k2_ablation': k2_ablation,
        'losses': {d: float(v) for d, v in all_losses.items()},
        'attn_cos_matrix': attn_cos.tolist(),
        'ffn_cos_matrix': ffn_cos.tolist(),
        'full_cos_matrix': full_cos.tolist(),
        'gt_sim_matrix': gt_sim.tolist(),
        'elapsed_seconds': elapsed,
    }


def run_multi_seed(seeds=[42, 123, 7]):
    """Run experiment across multiple seeds and aggregate."""
    all_results = []
    for seed in seeds:
        result = run_experiment(seed)
        all_results.append(result)

    # Aggregate correlations
    attn_rhos = [r['correlations']['spearman']['attn']['rho'] for r in all_results]
    ffn_rhos = [r['correlations']['spearman']['ffn']['rho'] for r in all_results]
    full_rhos = [r['correlations']['spearman']['full']['rho'] for r in all_results]

    # Aggregate within/cross ratios
    attn_ratios = [r['cluster_analysis']['attn_ratio'] for r in all_results]
    ffn_ratios = [r['cluster_analysis']['ffn_ratio'] for r in all_results]
    full_ratios = [r['cluster_analysis']['full_ratio'] for r in all_results]

    summary = {
        'n_seeds': len(seeds),
        'seeds': seeds,
        'spearman_rho': {
            'attn': {'mean': float(np.mean(attn_rhos)), 'std': float(np.std(attn_rhos)),
                     'values': attn_rhos},
            'ffn':  {'mean': float(np.mean(ffn_rhos)),  'std': float(np.std(ffn_rhos)),
                     'values': ffn_rhos},
            'full': {'mean': float(np.mean(full_rhos)), 'std': float(np.std(full_rhos)),
                     'values': full_rhos},
        },
        'within_cross_ratio': {
            'attn': {'mean': float(np.mean(attn_ratios)), 'std': float(np.std(attn_ratios)),
                     'values': attn_ratios},
            'ffn':  {'mean': float(np.mean(ffn_ratios)),  'std': float(np.std(ffn_ratios)),
                     'values': ffn_ratios},
            'full': {'mean': float(np.mean(full_ratios)), 'std': float(np.std(full_ratios)),
                     'values': full_ratios},
        },
        'per_seed': all_results,
    }

    # Kill criteria assessment
    mean_attn_rho = np.mean(attn_rhos)
    all_attn_p = [r['correlations']['spearman']['attn']['p'] for r in all_results]

    k1_pass = mean_attn_rho >= 0.3 and max(all_attn_p) < 0.05
    k1_detail = (f"mean attn rho={mean_attn_rho:.4f} (threshold: >=0.3), "
                 f"max p={max(all_attn_p):.4e} (threshold: <0.05)")

    # K2: check if rho increases or stays stable as N grows
    k2_ablations = all_results[0]['k2_ablation']  # use first seed
    if len(k2_ablations) >= 2:
        rho_at_small = k2_ablations[0]['mean_rho']
        rho_at_large = k2_ablations[-1]['mean_rho']
        k2_pass = rho_at_large >= rho_at_small * 0.7  # not more than 30% drop
        k2_detail = f"rho at N={k2_ablations[0]['n_domains']}={rho_at_small:.4f}, at N={k2_ablations[-1]['n_domains']}={rho_at_large:.4f}"
    else:
        k2_pass = True
        k2_detail = "insufficient data for ablation"

    # Is attention a BETTER predictor than FFN?
    attn_better = mean_attn_rho > np.mean(ffn_rhos)

    summary['kill_criteria'] = {
        'K1': {'pass': bool(k1_pass), 'detail': k1_detail},
        'K2': {'pass': bool(k2_pass), 'detail': k2_detail},
        'overall_kill': not (k1_pass and k2_pass),
        'attention_better_than_ffn': bool(attn_better),
        'attn_ffn_rho_diff': float(mean_attn_rho - np.mean(ffn_rhos)),
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"\n  Spearman rho (cosine vs ground-truth similarity):")
    print(f"    Attention: {np.mean(attn_rhos):.4f} +/- {np.std(attn_rhos):.4f}")
    print(f"    FFN:       {np.mean(ffn_rhos):.4f} +/- {np.std(ffn_rhos):.4f}")
    print(f"    Full:      {np.mean(full_rhos):.4f} +/- {np.std(full_rhos):.4f}")
    print(f"\n  Within/Cross cluster ratio:")
    print(f"    Attention: {np.mean(attn_ratios):.2f}x +/- {np.std(attn_ratios):.2f}")
    print(f"    FFN:       {np.mean(ffn_ratios):.2f}x +/- {np.std(ffn_ratios):.2f}")
    print(f"    Full:      {np.mean(full_ratios):.2f}x +/- {np.std(full_ratios):.2f}")
    print(f"\n  Kill criteria:")
    print(f"    K1 (attn correlates with similarity): {'PASS' if k1_pass else 'KILL'}")
    print(f"       {k1_detail}")
    print(f"    K2 (signal stable with more domains): {'PASS' if k2_pass else 'KILL'}")
    print(f"       {k2_detail}")
    print(f"    Attention better than FFN: {attn_better}")
    print(f"       diff = {mean_attn_rho - np.mean(ffn_rhos):.4f}")
    print(f"\n  Overall: {'SUPPORTED' if not summary['kill_criteria']['overall_kill'] else 'KILLED'}")

    return summary


if __name__ == '__main__':
    results = run_multi_seed(seeds=[42, 123, 7])

    # Save results
    out_path = Path(__file__).parent / 'results.json'
    # Convert non-serializable types
    def sanitize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"\n  Results saved to {out_path}")
