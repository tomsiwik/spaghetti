#!/usr/bin/env python3
"""
EigenLoRAx Principal Subspace Extraction from N=25 Trained Adapters

Tests whether SVD of stacked ternary LoRA adapter weights reveals a shared
principal subspace that can accelerate new adapter training.

Algorithm (from arXiv 2502.04700):
  1. For each (layer, module, matrix_type), stack N=25 adapter matrices
  2. Center and SVD -> extract top K principal components
  3. Train new adapter using only coefficients on PCs (freeze PCs)
  4. Compare quality to from-scratch LoRA training

Kill criteria:
  K1: principal subspace explains <50% variance (no shared structure)
  K2: new adapter trained on subspace coefficients >20% worse PPL than from-scratch
  K3: subspace extraction takes >10 min for N=25

Platform: Apple Silicon, numpy for SVD, MLX for training. $0.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SUBSPACE_DIR = EXPERIMENT_DIR / "subspace"

# All 25 adapter paths
ADAPTER_PATHS = {
    # 15 domain adapters from n15
    "chemistry": Path("experiments/models/bitnet_scale_n15/adapters/chemistry/adapter.npz"),
    "code": Path("experiments/models/bitnet_scale_n15/adapters/code/adapter.npz"),
    "cooking": Path("experiments/models/bitnet_scale_n15/adapters/cooking/adapter.npz"),
    "creative": Path("experiments/models/bitnet_scale_n15/adapters/creative/adapter.npz"),
    "dialogue": Path("experiments/models/bitnet_scale_n15/adapters/dialogue/adapter.npz"),
    "finance": Path("experiments/models/bitnet_scale_n15/adapters/finance/adapter.npz"),
    "health": Path("experiments/models/bitnet_scale_n15/adapters/health/adapter.npz"),
    "javascript": Path("experiments/models/bitnet_scale_n15/adapters/javascript/adapter.npz"),
    "legal": Path("experiments/models/bitnet_scale_n15/adapters/legal/adapter.npz"),
    "math": Path("experiments/models/bitnet_scale_n15/adapters/math/adapter.npz"),
    "medical": Path("experiments/models/bitnet_scale_n15/adapters/medical/adapter.npz"),
    "physics": Path("experiments/models/bitnet_scale_n15/adapters/physics/adapter.npz"),
    "science": Path("experiments/models/bitnet_scale_n15/adapters/science/adapter.npz"),
    "sql": Path("experiments/models/bitnet_scale_n15/adapters/sql/adapter.npz"),
    "wikitext": Path("experiments/models/bitnet_scale_n15/adapters/wikitext/adapter.npz"),
    # 4 capability adapters from taxonomy
    "conciseness": Path("experiments/models/capability_expert_taxonomy/adapters/conciseness/adapter.npz"),
    "instruction": Path("experiments/models/capability_expert_taxonomy/adapters/instruction/adapter.npz"),
    "reasoning": Path("experiments/models/capability_expert_taxonomy/adapters/reasoning/adapter.npz"),
    "safety": Path("experiments/models/capability_expert_taxonomy/adapters/safety/adapter.npz"),
    # 6 new capability adapters from n25
    "coding_style": Path("experiments/models/bitnet_scale_n25/adapters/coding_style/adapter.npz"),
    "debate": Path("experiments/models/bitnet_scale_n25/adapters/debate/adapter.npz"),
    "formal_writing": Path("experiments/models/bitnet_scale_n25/adapters/formal_writing/adapter.npz"),
    "multilingual": Path("experiments/models/bitnet_scale_n25/adapters/multilingual/adapter.npz"),
    "summarization": Path("experiments/models/bitnet_scale_n25/adapters/summarization/adapter.npz"),
    "translation": Path("experiments/models/bitnet_scale_n25/adapters/translation/adapter.npz"),
}


def log(msg):
    print(msg, flush=True)


# ===========================================================================
# Phase 1: Load all adapters and organize by (layer, module, A/B)
# ===========================================================================
def load_all_adapters():
    """Load all 25 adapters and return dict[adapter_name] -> dict[param_key] -> np.array"""
    adapters = {}
    for name, path in sorted(ADAPTER_PATHS.items()):
        full_path = Path(__file__).parent.parent.parent.parent / path
        if not full_path.exists():
            log(f"  WARNING: Missing adapter {name} at {full_path}")
            continue
        data = dict(np.load(str(full_path)))
        adapters[name] = {k: v.astype(np.float32) for k, v in data.items()}
        log(f"  Loaded {name}: {len(data)} params")
    return adapters


def get_module_keys(adapters):
    """Get unique module keys (e.g., 'model.layers.0.mlp.gate_proj.lora_a')"""
    first = next(iter(adapters.values()))
    return sorted(first.keys())


# ===========================================================================
# Phase 2: EigenLoRAx subspace extraction (per module key)
# ===========================================================================
def extract_subspace_per_key(adapters, key, adapter_names):
    """
    For a single parameter key (e.g., model.layers.0.mlp.gate_proj.lora_a):
    1. Stack N matrices along columns: W_hat = [W_1 | W_2 | ... | W_N]
       where each W_i has shape (m, n), stacked gives (m, N*n)
    2. Center: W_c = W_hat - mean(W_hat)
    3. SVD: U, S, Vt = svd(W_c)
    4. Return singular values and explained variance

    Actually, following the paper more precisely:
    For LoRA-A of shape (in_features, rank), we stack N adapters:
      W_hat shape = (in_features, N * rank)
    For LoRA-B of shape (rank, out_features), we stack N adapters:
      W_hat shape = (rank, N * out_features)

    Returns: (singular_values, total_variance, cumulative_variance_ratio)
    """
    matrices = []
    for name in adapter_names:
        matrices.append(adapters[name][key])

    # Stack along columns (axis=1)
    W_hat = np.concatenate(matrices, axis=1)  # (m, N*n)

    # Center
    # Mean per-adapter: reshape to (m, N, n), mean over N axis
    m = matrices[0].shape[0]
    n = matrices[0].shape[1]
    N = len(adapter_names)
    W_reshaped = W_hat.reshape(m, N, n)
    W_mean = W_reshaped.mean(axis=1, keepdims=True)  # (m, 1, n)
    W_centered = (W_reshaped - W_mean).reshape(m, N * n)

    # SVD (economy)
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    # Explained variance
    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / (total_var + 1e-30)

    return S, total_var, cumvar, U, Vt, W_mean.squeeze(1)


def extract_full_subspace(adapters):
    """Extract subspace for ALL module keys. Returns analysis results."""
    adapter_names = sorted(adapters.keys())
    keys = get_module_keys(adapters)
    N = len(adapter_names)

    log(f"\n{'='*60}")
    log(f"Phase 2: EigenLoRAx Subspace Extraction (N={N})")
    log(f"{'='*60}")

    results = {}
    all_variances = {"lora_a": [], "lora_b": []}

    t0 = time.time()

    for key in keys:
        S, total_var, cumvar, U, Vt, mean = extract_subspace_per_key(
            adapters, key, adapter_names
        )

        matrix_type = "lora_a" if "lora_a" in key else "lora_b"

        # How many PCs needed for 50%, 80%, 95% variance
        k50 = int(np.searchsorted(cumvar, 0.50) + 1) if cumvar[-1] >= 0.50 else len(cumvar)
        k80 = int(np.searchsorted(cumvar, 0.80) + 1) if cumvar[-1] >= 0.80 else len(cumvar)
        k95 = int(np.searchsorted(cumvar, 0.95) + 1) if cumvar[-1] >= 0.95 else len(cumvar)

        results[key] = {
            "shape": list(next(iter(adapters.values()))[key].shape),
            "matrix_type": matrix_type,
            "total_variance": float(total_var),
            "top_singular_values": S[:10].tolist(),
            "k_for_50pct": int(k50),
            "k_for_80pct": int(k80),
            "k_for_95pct": int(k95),
            "cumvar_at_k8": float(cumvar[min(7, len(cumvar)-1)]),
            "cumvar_at_k16": float(cumvar[min(15, len(cumvar)-1)]),
            "cumvar_at_k25": float(cumvar[min(24, len(cumvar)-1)]),
            "n_components": len(S),
        }

        all_variances[matrix_type].append({
            "key": key,
            "cumvar_at_k8": float(cumvar[min(7, len(cumvar)-1)]),
            "cumvar_at_k16": float(cumvar[min(15, len(cumvar)-1)]),
            "k_for_50pct": int(k50),
        })

    extraction_time = time.time() - t0

    # Aggregate statistics
    for mat_type in ["lora_a", "lora_b"]:
        entries = all_variances[mat_type]
        avg_k8 = np.mean([e["cumvar_at_k8"] for e in entries])
        avg_k16 = np.mean([e["cumvar_at_k16"] for e in entries])
        avg_k50 = np.mean([e["k_for_50pct"] for e in entries])
        log(f"\n  {mat_type} ({len(entries)} keys):")
        log(f"    Avg variance explained at K=8:  {avg_k8:.4f}")
        log(f"    Avg variance explained at K=16: {avg_k16:.4f}")
        log(f"    Avg K needed for 50% variance:  {avg_k50:.1f}")

    log(f"\n  Extraction time: {extraction_time:.1f}s")

    return results, extraction_time, all_variances


# ===========================================================================
# Phase 3: K1 Assessment -- does a shared subspace exist?
# ===========================================================================
def assess_k1(all_variances, extraction_time):
    """
    K1: principal subspace explains <50% variance -> KILL
    We check: with K=16 PCs (same as our LoRA rank), what fraction
    of variance is explained across all layers?
    """
    log(f"\n{'='*60}")
    log(f"Phase 3: K1 Assessment -- Shared Subspace?")
    log(f"{'='*60}")

    # For K1, use K=16 (matching our adapter rank)
    k16_a = [e["cumvar_at_k16"] for e in all_variances["lora_a"]]
    k16_b = [e["cumvar_at_k16"] for e in all_variances["lora_b"]]

    avg_a = np.mean(k16_a)
    avg_b = np.mean(k16_b)
    avg_overall = np.mean(k16_a + k16_b)
    min_a = np.min(k16_a)
    min_b = np.min(k16_b)
    min_overall = min(min_a, min_b)

    log(f"  Variance explained at K=16 PCs:")
    log(f"    LoRA-A: avg={avg_a:.4f}, min={min_a:.4f}")
    log(f"    LoRA-B: avg={avg_b:.4f}, min={min_b:.4f}")
    log(f"    Overall: avg={avg_overall:.4f}, min={min_overall:.4f}")

    # Also check K=8 (paper's sweet spot)
    k8_a = [e["cumvar_at_k8"] for e in all_variances["lora_a"]]
    k8_b = [e["cumvar_at_k8"] for e in all_variances["lora_b"]]
    avg_k8 = np.mean(k8_a + k8_b)
    log(f"  Variance explained at K=8 PCs:")
    log(f"    Overall: avg={np.mean(k8_a + k8_b):.4f}")

    # K1 verdict
    k1_pass = avg_overall >= 0.50
    log(f"\n  K1 VERDICT: {'PASS' if k1_pass else 'FAIL'} "
        f"(avg variance at K=16 = {avg_overall:.4f}, threshold 0.50)")

    # K3 verdict (extraction time)
    k3_pass = extraction_time < 600  # 10 min
    log(f"  K3 VERDICT: {'PASS' if k3_pass else 'FAIL'} "
        f"(extraction time = {extraction_time:.1f}s, threshold 600s)")

    return {
        "k1_pass": k1_pass,
        "k1_avg_variance_k16": float(avg_overall),
        "k1_min_variance_k16": float(min_overall),
        "k1_avg_variance_k8": float(avg_k8),
        "k1_lora_a_avg": float(avg_a),
        "k1_lora_b_avg": float(avg_b),
        "k3_pass": k3_pass,
        "k3_extraction_time_s": float(extraction_time),
    }


# ===========================================================================
# Phase 4: If K1 passes, train a new adapter using subspace coefficients
# ===========================================================================
def train_subspace_adapter(adapters, K=16):
    """
    Train a 26th adapter (on a held-out domain) using only subspace coefficients.
    Compare to from-scratch LoRA training on the same data.

    Uses the 'wikitext' domain as test (hold it out from subspace extraction,
    train a wikitext adapter with coefficients only).
    """
    import mlx.core as mx
    import mlx.nn as nn_mlx
    import mlx.optimizers as opt
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear

    HOLDOUT_DOMAIN = "wikitext"
    MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
    LORA_RANK = 16
    LORA_SCALE = 20.0
    TRAIN_STEPS = 400
    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 128
    LEARNING_RATE = 1e-4
    VAL_BATCHES = 25

    log(f"\n{'='*60}")
    log(f"Phase 4: Subspace Adapter Training (holdout={HOLDOUT_DOMAIN}, K={K})")
    log(f"{'='*60}")

    # Extract subspace WITHOUT holdout adapter
    train_adapters = {k: v for k, v in adapters.items() if k != HOLDOUT_DOMAIN}
    train_names = sorted(train_adapters.keys())
    N_train = len(train_names)
    log(f"  Extracting subspace from {N_train} adapters (excluding {HOLDOUT_DOMAIN})")

    keys = get_module_keys(adapters)

    # Extract and store PCs for each key
    subspace = {}  # key -> (V_K, mean)  where V_K has shape (K, total_cols)
    for key in keys:
        S, total_var, cumvar, U, Vt, mean = extract_subspace_per_key(
            train_adapters, key, train_names
        )
        # Top K right singular vectors
        V_K = Vt[:K]  # (K, N*n) -- but we need per-adapter PCs
        subspace[key] = {
            "U_K": U[:, :K],  # (m, K) -- left singular vectors
            "S_K": S[:K],     # (K,)
            "V_K": V_K,       # (K, N*n)
            "mean": mean,     # (m, n)
        }

    log(f"  Subspace extracted. Loading model...")

    # Load model
    model, tokenizer = load(MODEL_ID)

    # Replace BitLinear with nn.Linear (same as N=25 experiment)
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                w = _unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn_mlx.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")

    # Load validation data
    data_dirs = [
        Path(__file__).parent.parent / "bitnet_scale_n15" / "data" / HOLDOUT_DOMAIN,
        Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data" / HOLDOUT_DOMAIN,
    ]
    val_path = None
    train_path = None
    for dd in data_dirs:
        vp = dd / "valid.jsonl"
        tp = dd / "train.jsonl"
        if vp.exists() and tp.exists():
            val_path = vp
            train_path = tp
            break
    if val_path is None:
        log(f"  ERROR: Cannot find data for {HOLDOUT_DOMAIN}")
        return None

    log(f"  Data: {train_path}")

    # Load data
    def load_texts(path, max_n=500):
        texts = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
                if len(texts) >= max_n:
                    break
        return texts

    train_texts = load_texts(train_path, 500)
    val_texts = load_texts(val_path, 50)

    def tokenize_batch(texts, idx):
        text = texts[idx % len(texts)]
        tokens = tokenizer.encode(text, add_special_tokens=True)[:MAX_SEQ_LENGTH + 1]
        if len(tokens) < 4:
            tokens = tokenizer.encode("The quick brown fox", add_special_tokens=True)[:MAX_SEQ_LENGTH + 1]
        return mx.array(tokens[:-1])[None], mx.array(tokens[1:])[None]

    def eval_ppl(model_fn, n_batches):
        total_loss = 0.0
        total_tokens = 0
        for i in range(n_batches):
            inp, tgt = tokenize_batch(val_texts, i)
            logits = model_fn(inp)
            loss = nn_mlx.losses.cross_entropy(logits, tgt, reduction="sum")
            mx.eval(loss)
            total_loss += float(loss)
            total_tokens += tgt.size
        return float(np.exp(total_loss / max(total_tokens, 1)))

    # -----------------------------------------------------------------------
    # Method A: From-scratch LoRA training (baseline)
    # -----------------------------------------------------------------------
    log(f"\n  --- Method A: From-Scratch LoRA (baseline) ---")

    # Apply LoRA
    target_keys_set = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }

    class TernaryLoRALinear(nn_mlx.Module):
        def __init__(self, base_linear, r=16, scale=20.0):
            super().__init__()
            self.linear = base_linear
            self.r = r
            self.scale = scale
            in_f = base_linear.weight.shape[1]
            out_f = base_linear.weight.shape[0]
            s = 1.0 / math.sqrt(in_f)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_f, r))
            self.lora_b = mx.zeros((r, out_f))

        def _ste_ternary(self, W):
            alpha = mx.mean(mx.abs(W)) + 1e-10
            W_scaled = W / alpha
            W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
            return W + mx.stop_gradient(W_q - W)

        def __call__(self, x):
            base_out = self.linear(x)
            A = self._ste_ternary(self.lora_a)
            B = self._ste_ternary(self.lora_b)
            return base_out + (x @ A) @ B * self.scale

    def apply_lora(model_obj):
        c = 0
        for layer in model_obj.model.layers:
            updates = []
            for key, module in layer.named_modules():
                if key in target_keys_set and isinstance(module, nn_mlx.Linear):
                    lora = TernaryLoRALinear(module, r=LORA_RANK, scale=LORA_SCALE)
                    updates.append((key, lora))
                    c += 1
            if updates:
                layer.update_modules(tree_unflatten(updates))
        mx.eval(model_obj.parameters())
        return c

    def get_lora_params(model_obj):
        params = {}
        for name, p in tree_flatten(model_obj.trainable_parameters()):
            if "lora_a" in name or "lora_b" in name:
                params[name] = p
        return params

    def reset_lora(model_obj, seed=42):
        mx.random.seed(seed)
        for layer in model_obj.model.layers:
            for key, module in layer.named_modules():
                if isinstance(module, TernaryLoRALinear):
                    in_d = module.lora_a.shape[0]
                    s = 1.0 / math.sqrt(in_d)
                    module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                    module.lora_b = mx.zeros_like(module.lora_b)
        mx.eval(model_obj.parameters())

    n_applied = apply_lora(model)
    log(f"  Applied LoRA to {n_applied} layers")

    # Measure base PPL (no adapter)
    reset_lora(model, seed=42)
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())
    base_ppl = eval_ppl(model, VAL_BATCHES)
    log(f"  Base PPL (no adapter): {base_ppl:.4f}")

    # Train from scratch
    reset_lora(model, seed=42)
    lora_params = get_lora_params(model)
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)
    n_trainable = sum(p.size for p in lora_params.values())
    log(f"  Trainable params (from-scratch): {n_trainable:,}")

    def loss_fn_scratch(model_obj, inp, tgt):
        logits = model_obj(inp)
        return nn_mlx.losses.cross_entropy(logits, tgt, reduction="mean")

    loss_and_grad = nn_mlx.value_and_grad(model, loss_fn_scratch)

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        inp, tgt = tokenize_batch(train_texts, step)
        loss_val, grads = loss_and_grad(model, inp, tgt)
        # Only update LoRA params
        lora_grads = {}
        for name, g in tree_flatten(grads):
            if "lora_a" in name or "lora_b" in name:
                lora_grads[name] = g
        optimizer.update(model, tree_unflatten(list(lora_grads.items())))
        mx.eval(model.parameters(), optimizer.state)
        if (step + 1) % 100 == 0:
            log(f"    Step {step+1}: loss={float(loss_val):.4f}")

    scratch_time = time.time() - t0
    scratch_ppl = eval_ppl(model, VAL_BATCHES)
    log(f"  From-scratch PPL: {scratch_ppl:.4f} ({scratch_time:.1f}s)")

    # -----------------------------------------------------------------------
    # Method B: Subspace coefficient training (EigenLoRAx-style)
    # -----------------------------------------------------------------------
    log(f"\n  --- Method B: Subspace Coefficients (K={K}) ---")

    # Reset LoRA params
    reset_lora(model, seed=42)

    # For each LoRA module, freeze A/B and instead parameterize as:
    #   A = mean_A + U_K @ diag(coeff_a) @ S_K  (projected back to original shape)
    # Actually, the simpler approach from the paper:
    #   A_new = mean_A + sum_k(alpha_k * PC_k)  where PC_k are the left singular vectors
    #
    # More precisely: for each module key, the trained adapter A matrix of shape (m, r)
    # is expressed as: A = mean + sum_{k=1}^{K} alpha_k * U_k * S_k * V_k^T (reshaped)
    #
    # But we need to think about this more carefully for our use case.
    # The PCs from SVD of the centered stacked matrix give us directions in the
    # flattened adapter space. Each adapter W_i of shape (m, n) when centered and
    # stacked gives SVD: U @ S @ V^T. The columns of U span the principal directions
    # in the row space. But we need full reconstruction.
    #
    # Simplest approach: treat each adapter matrix as a flattened vector of size m*n,
    # stack N such vectors into (N, m*n), SVD that, get K principal directions.
    # New adapter = mean + sum_k alpha_k * PC_k (reshaped to m x n).
    #
    # This is cleaner and directly matches the variance explanation.

    # Re-extract subspace using FLATTENED approach for training
    log(f"  Re-extracting subspace (flattened approach, K={K})...")

    # Build subspace per module key using flattened vectors
    flat_subspace = {}
    for key in keys:
        matrices = []
        for name in train_names:
            W = train_adapters[name][key]
            matrices.append(W.flatten())

        # Stack: (N_train, m*n)
        M = np.stack(matrices, axis=0)
        mean_vec = M.mean(axis=0)  # (m*n,)
        M_centered = M - mean_vec[None, :]

        # SVD of centered matrix
        U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        # U: (N_train, min(N_train, m*n))
        # S: (min(N_train, m*n),)
        # Vt: (min(N_train, m*n), m*n)

        # Top K PCs in the original param space
        PCs = Vt[:K]  # (K, m*n) -- principal directions
        shape = next(iter(adapters.values()))[key].shape

        flat_subspace[key] = {
            "mean": mean_vec,
            "PCs": PCs,  # (K, m*n)
            "S": S[:K],
            "shape": shape,
        }

    # Count subspace parameters: K coefficients per key
    n_subspace_params = K * len(keys)
    log(f"  Subspace params: K={K} x {len(keys)} keys = {n_subspace_params:,}")
    log(f"  From-scratch params: {n_trainable:,}")
    log(f"  Compression ratio: {n_trainable / n_subspace_params:.1f}x")

    # Verify reconstruction: how well can we reconstruct the holdout adapter
    # using K PCs from the other 24?
    holdout_params = adapters[HOLDOUT_DOMAIN]
    recon_errors = []
    for key in keys:
        fs = flat_subspace[key]
        holdout_vec = holdout_params[key].flatten()
        centered = holdout_vec - fs["mean"]

        # Project onto K PCs
        coeffs = centered @ fs["PCs"].T  # (K,)
        recon = fs["mean"] + coeffs @ fs["PCs"]

        # Reconstruction error
        err = np.linalg.norm(holdout_vec - recon) / (np.linalg.norm(holdout_vec) + 1e-10)
        recon_errors.append(err)

    avg_recon_err = np.mean(recon_errors)
    log(f"  Holdout reconstruction error (K={K}): {avg_recon_err:.4f} "
        f"(0=perfect, 1=no reconstruction)")

    # Now train using subspace: initialize adapter as mean + random coefficients,
    # then optimize coefficients only.
    # We parameterize each module's adapter as:
    #   W = reshape(mean + alpha @ PCs, shape)
    # where alpha is (K,) learned coefficients.

    class SubspaceLoRALinear(nn_mlx.Module):
        def __init__(self, base_linear, pc_a, mean_a, pc_b, mean_b,
                     shape_a, shape_b, K, scale=20.0):
            super().__init__()
            self.linear = base_linear
            self.scale = scale
            self.shape_a = shape_a
            self.shape_b = shape_b

            # Frozen PCs and means (as MLX arrays)
            self.pc_a = mx.array(pc_a)    # (K, m_a*n_a)
            self.mean_a = mx.array(mean_a)  # (m_a*n_a,)
            self.pc_b = mx.array(pc_b)    # (K, m_b*n_b)
            self.mean_b = mx.array(mean_b)  # (m_b*n_b,)

            # Learnable coefficients
            self.alpha_a = mx.zeros((K,))
            self.alpha_b = mx.zeros((K,))

        def _ste_ternary(self, W):
            alpha = mx.mean(mx.abs(W)) + 1e-10
            W_scaled = W / alpha
            W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
            return W + mx.stop_gradient(W_q - W)

        def __call__(self, x):
            base_out = self.linear(x)
            # Reconstruct A and B from subspace
            A_flat = self.mean_a + self.alpha_a @ self.pc_a
            B_flat = self.mean_b + self.alpha_b @ self.pc_b
            A = mx.reshape(A_flat, self.shape_a)
            B = mx.reshape(B_flat, self.shape_b)
            A = self._ste_ternary(A)
            B = self._ste_ternary(B)
            return base_out + (x @ A) @ B * self.scale

    # Apply subspace LoRA
    target_modules_per_layer = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    sub_count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        updates = []
        for mod_key in target_modules_per_layer:
            # Find the current module
            parts = mod_key.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p, None)
                if module is None:
                    break

            if module is None:
                continue

            # Get the base linear (unwrap if it's a TernaryLoRALinear)
            if isinstance(module, TernaryLoRALinear):
                base_lin = module.linear
            elif isinstance(module, nn_mlx.Linear):
                base_lin = module
            else:
                continue

            # Get PCs for this layer's A and B
            key_a = f"model.layers.{layer_idx}.{mod_key}.lora_a"
            key_b = f"model.layers.{layer_idx}.{mod_key}.lora_b"

            if key_a not in flat_subspace or key_b not in flat_subspace:
                continue

            fs_a = flat_subspace[key_a]
            fs_b = flat_subspace[key_b]

            sub_lora = SubspaceLoRALinear(
                base_lin,
                pc_a=fs_a["PCs"], mean_a=fs_a["mean"],
                pc_b=fs_b["PCs"], mean_b=fs_b["mean"],
                shape_a=fs_a["shape"], shape_b=fs_b["shape"],
                K=K, scale=LORA_SCALE,
            )
            updates.append((mod_key, sub_lora))
            sub_count += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    log(f"  Applied SubspaceLoRA to {sub_count} modules")

    # Count trainable params
    sub_trainable = 0
    for name, p in tree_flatten(model.trainable_parameters()):
        if "alpha_a" in name or "alpha_b" in name:
            sub_trainable += p.size
    log(f"  Subspace trainable params: {sub_trainable:,}")

    # Train subspace coefficients
    def loss_fn_sub(model_obj, inp, tgt):
        logits = model_obj(inp)
        return nn_mlx.losses.cross_entropy(logits, tgt, reduction="mean")

    loss_and_grad_sub = nn_mlx.value_and_grad(model, loss_fn_sub)
    optimizer_sub = opt.Adam(learning_rate=LEARNING_RATE)

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        inp, tgt = tokenize_batch(train_texts, step)
        loss_val, grads = loss_and_grad_sub(model, inp, tgt)
        # Only update alpha params
        alpha_grads = {}
        for name, g in tree_flatten(grads):
            if "alpha_a" in name or "alpha_b" in name:
                alpha_grads[name] = g
        if alpha_grads:
            optimizer_sub.update(model, tree_unflatten(list(alpha_grads.items())))
        mx.eval(model.parameters(), optimizer_sub.state)
        if (step + 1) % 100 == 0:
            log(f"    Step {step+1}: loss={float(loss_val):.4f}")

    subspace_time = time.time() - t0
    subspace_ppl = eval_ppl(model, VAL_BATCHES)
    log(f"  Subspace PPL: {subspace_ppl:.4f} ({subspace_time:.1f}s)")

    # K2 Assessment
    ppl_gap = (subspace_ppl - scratch_ppl) / scratch_ppl * 100
    k2_pass = ppl_gap <= 20.0
    log(f"\n  K2 VERDICT: {'PASS' if k2_pass else 'FAIL'} "
        f"(subspace PPL gap = {ppl_gap:+.1f}%, threshold 20%)")

    return {
        "holdout_domain": HOLDOUT_DOMAIN,
        "K": K,
        "base_ppl": float(base_ppl),
        "scratch_ppl": float(scratch_ppl),
        "subspace_ppl": float(subspace_ppl),
        "ppl_gap_pct": float(ppl_gap),
        "scratch_params": n_trainable,
        "subspace_params": n_subspace_params,
        "actual_subspace_trainable": sub_trainable,
        "compression_ratio": float(n_trainable / n_subspace_params),
        "scratch_time_s": float(scratch_time),
        "subspace_time_s": float(subspace_time),
        "speedup": float(scratch_time / max(subspace_time, 1)),
        "holdout_recon_error": float(avg_recon_err),
        "k2_pass": k2_pass,
    }


def _unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack ternary weights from BitLinear format."""
    import mlx.core as mx
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


# ===========================================================================
# Main
# ===========================================================================
def main():
    log("="*60)
    log("EigenLoRAx Subspace Extraction Experiment")
    log("="*60)

    total_t0 = time.time()

    # Phase 1: Load adapters
    log(f"\nPhase 1: Loading {len(ADAPTER_PATHS)} adapters...")
    adapters = load_all_adapters()
    log(f"  Loaded {len(adapters)} adapters successfully")

    if len(adapters) < 25:
        log(f"  WARNING: Only {len(adapters)}/25 adapters found!")

    # Phase 2: Extract subspace
    key_results, extraction_time, all_variances = extract_full_subspace(adapters)

    # Phase 3: K1 assessment
    k1_results = assess_k1(all_variances, extraction_time)

    # Phase 4: If K1 and K3 pass, do training comparison
    k2_results = None
    if k1_results["k1_pass"] and k1_results["k3_pass"]:
        log(f"\n  K1 and K3 PASS -- proceeding to K2 (training comparison)")
        k2_results = train_subspace_adapter(adapters, K=16)
    else:
        log(f"\n  K1={'PASS' if k1_results['k1_pass'] else 'FAIL'}, "
            f"K3={'PASS' if k1_results['k3_pass'] else 'FAIL'} "
            f"-- skipping K2 training")
        # Still useful to know if K1 failed -- this is the key finding

    # Compile results
    total_time = time.time() - total_t0
    results = {
        "experiment": "eigenlorax_subspace",
        "n_adapters": len(adapters),
        "adapter_params_each": sum(v.size for v in next(iter(adapters.values())).values()),
        "n_module_keys": len(get_module_keys(adapters)),
        "extraction_time_s": extraction_time,
        "total_time_s": total_time,
        "k1": k1_results,
        "k2": k2_results,
        # Per-layer variance statistics (summary)
        "variance_summary": {
            mat_type: {
                "avg_cumvar_k8": float(np.mean([e["cumvar_at_k8"] for e in entries])),
                "avg_cumvar_k16": float(np.mean([e["cumvar_at_k16"] for e in entries])),
                "avg_k_for_50pct": float(np.mean([e["k_for_50pct"] for e in entries])),
                "min_cumvar_k16": float(np.min([e["cumvar_at_k16"] for e in entries])),
            }
            for mat_type, entries in all_variances.items()
        },
    }

    # Final verdict
    k1_pass = k1_results["k1_pass"]
    k3_pass = k1_results["k3_pass"]
    k2_pass = k2_results["k2_pass"] if k2_results else None

    all_pass = k1_pass and k3_pass and (k2_pass is True)
    any_kill = (not k1_pass) or (not k3_pass) or (k2_pass is False)

    if any_kill:
        verdict = "KILLED"
        reasons = []
        if not k1_pass:
            reasons.append(f"K1 FAIL (variance {k1_results['k1_avg_variance_k16']:.4f} < 0.50)")
        if not k3_pass:
            reasons.append(f"K3 FAIL (time {extraction_time:.0f}s > 600s)")
        if k2_pass is False:
            reasons.append(f"K2 FAIL (PPL gap {k2_results['ppl_gap_pct']:+.1f}% > 20%)")
        results["verdict"] = verdict
        results["kill_reasons"] = reasons
    elif all_pass:
        results["verdict"] = "SUPPORTED"
        results["kill_reasons"] = []
    else:
        results["verdict"] = "PARTIAL"
        results["kill_reasons"] = []

    log(f"\n{'='*60}")
    log(f"VERDICT: {results['verdict']}")
    if results.get("kill_reasons"):
        for r in results["kill_reasons"]:
            log(f"  - {r}")
    log(f"Total time: {total_time:.1f}s")
    log(f"{'='*60}")

    # Save (convert numpy bools to Python bools)
    def sanitize(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(RESULTS_FILE, "w") as f:
        json.dump(sanitize(results), f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
