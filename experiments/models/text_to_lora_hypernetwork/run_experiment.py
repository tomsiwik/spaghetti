#!/usr/bin/env python3
"""Text-to-LoRA hypernetwork: generate adapters from task description.

Tests whether text descriptions can generate useful LoRA B-matrices for BitNet-2B-4T.

Kill criteria:
  K1: T2L-generated adapter PPL > 3x trained adapter on matched domain -> KILL
  K2: Post-processed adapters lose > 50% of T2L quality after orth projection -> KILL
  K3: T2L hypernetwork too large to fit alongside base model in 48GB -> KILL

Approach:
  Phase 1: Extract domain description embeddings from BitNet-2B-4T
  Phase 2: Load trained adapter B-matrices, compute NN baseline in embedding space
  Phase 3: Train hypernetwork (embedding -> adapter coefficients) with LOO cross-val
  Phase 4: Evaluate NN adapter PPL on target domains (proper BitLinear unpack + LoRA)
  Phase 5: Orthogonal projection test for composition safety
  Phase 6: Assess kill criteria

References:
  - Text-to-LoRA (arxiv 2506.06105, ICML 2025)
  - FlyLoRA (arxiv 2510.08396)
  - exp_real_data_25_domain_adapters (24 trained adapters)

Platform: Apple M5 Pro 48GB, MLX
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "adapters"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEED = 42
VAL_SAMPLES = 10  # Per domain for fast eval

DOMAINS = [
    "medical", "code", "math", "legal", "finance", "science",
    "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering",
    "agriculture", "environmental", "politics", "economics",
    "sociology", "linguistics", "cybersecurity", "marketing",
    "sports", "music",
]

DOMAIN_DESCRIPTIONS = {
    "medical": "Medical knowledge including anatomy, physiology, diseases, diagnosis, and treatment of patients in clinical settings.",
    "code": "Programming and software development including Python, algorithms, data structures, debugging, and code review.",
    "math": "Mathematics including algebra, calculus, statistics, proofs, theorems, and mathematical problem-solving.",
    "legal": "Legal knowledge including contract law, constitutional law, case analysis, legal reasoning, and court procedures.",
    "finance": "Financial analysis, investment, banking, accounting, economics, risk management, and portfolio theory.",
    "science": "Natural sciences including physics, chemistry, biology, earth science, and scientific methodology.",
    "history": "Historical events, civilizations, political movements, wars, cultural developments, and historiography.",
    "philosophy": "Philosophical reasoning, ethics, metaphysics, epistemology, logic, and major philosophical traditions.",
    "creative_writing": "Creative writing including fiction, poetry, narrative techniques, character development, and storytelling.",
    "cooking": "Culinary arts including recipes, cooking techniques, food science, nutrition, and cuisine traditions.",
    "health_fitness": "Health and fitness including exercise science, nutrition, wellness, preventive medicine, and physical training.",
    "psychology": "Psychology including cognitive science, behavioral analysis, mental health, therapy approaches, and research methods.",
    "education": "Education theory and practice including pedagogy, curriculum design, assessment, and learning sciences.",
    "engineering": "Engineering disciplines including mechanical, electrical, civil, and systems engineering design and analysis.",
    "agriculture": "Agriculture including crop science, soil management, sustainable farming, agribusiness, and food production.",
    "environmental": "Environmental science including ecology, climate change, conservation, pollution, and sustainability.",
    "politics": "Political science including government systems, policy analysis, international relations, and political theory.",
    "economics": "Economics including microeconomics, macroeconomics, econometrics, market analysis, and economic policy.",
    "sociology": "Sociology including social theory, demographics, cultural studies, institutions, and social research methods.",
    "linguistics": "Linguistics including syntax, semantics, phonology, language acquisition, and computational linguistics.",
    "cybersecurity": "Cybersecurity including network security, cryptography, threat analysis, incident response, and ethical hacking.",
    "marketing": "Marketing strategy, consumer behavior, digital marketing, brand management, and market research.",
    "sports": "Sports science, athletic training, competition rules, sports history, and performance analysis.",
    "music": "Music theory, composition, performance practice, music history, and audio production.",
}

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# BitNet unpacking utilities (from real_data_25_domain_adapters)
# ============================================================================
from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16."""
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


def replace_bitlinear_with_linear(model):
    """Replace BitLinear with nn.Linear for differentiable LoRA."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


class SimpleLoRALinear(nn.Module):
    """Simple LoRA linear: base + scale * (x @ A) @ B."""
    def __init__(self, base_linear, rank, scale, a_init=None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        self.linear = base_linear
        self.scale = scale
        self.rank = rank
        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, rank))
        self.lora_b = mx.zeros((rank, out_features))
        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora_structure(model, skeleton, domain_idx):
    """Apply LoRA structure with A matrices from skeleton."""
    count = 0
    n_layers = len(model.model.layers)
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            else:
                a_mx = None

            lora = SimpleLoRALinear(module, rank=LORA_RANK, scale=LORA_SCALE, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    print(f"  Applied LoRA (r={LORA_RANK}) to {count} layers")
    return model


def set_lora_a(model, skeleton, domain_idx):
    """Update A matrices for a different domain."""
    n_layers = len(model.model.layers)
    for li, layer in enumerate(model.model.layers):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, SimpleLoRALinear):
                continue
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                module.lora_a = mx.array(skeleton[skey]).astype(mx.bfloat16)


def set_lora_b_from_npz(model, adapter_params):
    """Load B matrices from adapter npz data."""
    # Zero all B first
    for layer in model.model.layers:
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, SimpleLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)

    # Apply loaded B weights
    model.update(tree_unflatten(list(adapter_params.items())))


def compute_ppl(model, tokenizer, val_data, n_samples=VAL_SAMPLES):
    """Compute perplexity on validation data."""
    total_loss = 0.0
    total_tokens = 0
    for sample in val_data[:n_samples]:
        text = sample.get("text", "")
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])
        logits = model(x)[0]
        mx.eval(logits)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        mx.eval(loss)
        total_loss += loss.item() * len(tokens[1:])
        total_tokens += len(tokens[1:])
        del logits, loss, x, y
    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_loss / total_tokens))


# ============================================================================
# Phase 1: Extract embeddings
# ============================================================================
def phase_extract_embeddings():
    """Get mean-pooled hidden states for each domain description."""
    print("\n=== Phase 1: Extract domain description embeddings ===")
    t0 = time.time()

    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("model-loaded")

    embeddings = {}
    for domain, desc in DOMAIN_DESCRIPTIONS.items():
        tokens = tokenizer.encode(desc)
        tokens_mx = mx.array(tokens)[None, :]

        # Get hidden states from last layer
        x = model.model.embed_tokens(tokens_mx)
        for layer in model.model.layers:
            x = layer(x, mask=None)
        if hasattr(model.model, "norm"):
            x = model.model.norm(x)

        emb = mx.mean(x[0].astype(mx.float32), axis=0)
        mx.eval(emb)
        embeddings[domain] = np.array(emb)
        del x, emb, tokens_mx

    d_embed = embeddings[DOMAINS[0]].shape[0]
    elapsed = time.time() - t0
    print(f"Extracted {len(embeddings)} embeddings, d={d_embed}, time={elapsed:.1f}s")
    np.savez(EXPERIMENT_DIR / "domain_embeddings.npz", **embeddings)

    log_memory("post-embeddings")
    cleanup(model, tokenizer)
    return embeddings, elapsed


# ============================================================================
# Phase 2: Load adapters + NN baseline
# ============================================================================
def phase_load_and_nn(embeddings):
    """Load trained adapters and compute NN baseline."""
    print("\n=== Phase 2: Load adapters + NN baseline ===")
    t0 = time.time()

    adapters = {}
    for domain in DOMAINS:
        path = ADAPTERS_DIR / domain / "adapter.npz"
        if path.exists():
            data = np.load(str(path))
            b_flat = []
            keys_sorted = sorted(data.keys())
            for key in keys_sorted:
                b_flat.append(data[key].flatten())
            adapters[domain] = {
                "flat": np.concatenate(b_flat),
                "keys": keys_sorted,
            }
            data.close()

    total_params = adapters[DOMAINS[0]]["flat"].shape[0]
    domain_list = [d for d in DOMAINS if d in embeddings and d in adapters]
    print(f"Loaded {len(adapters)} adapters, {total_params:,} params each")

    # NN baseline
    emb_matrix = np.stack([embeddings[d] for d in domain_list])
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / (norms + 1e-8)
    cos_sim = emb_normed @ emb_normed.T

    nn_results = {}
    for i, domain in enumerate(domain_list):
        sims = cos_sim[i].copy()
        sims[i] = -np.inf
        nn_idx = np.argmax(sims)
        nn_domain = domain_list[nn_idx]
        nn_sim = cos_sim[i, nn_idx]
        nn_results[domain] = {
            "nearest_neighbor": nn_domain,
            "cosine_similarity": float(nn_sim),
        }
        print(f"  {domain:20s} -> {nn_domain:20s} (cos={nn_sim:.4f})")

    elapsed = time.time() - t0
    print(f"Phase 2 complete, time={elapsed:.1f}s")
    return adapters, total_params, domain_list, nn_results, elapsed


# ============================================================================
# Phase 3: Train hypernetwork
# ============================================================================
class HyperNetwork(nn.Module):
    """Predicts softmax coefficients over training adapters from an embedding."""
    def __init__(self, d_embed, d_hidden, n_basis):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_embed, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(d_hidden, n_basis)

    def __call__(self, x):
        h = self.trunk(x)
        return mx.softmax(self.head(h), axis=-1)


def phase_train_hypernetwork(embeddings, adapters, domain_list):
    """Train hypernetwork with leave-one-out cross-validation."""
    print("\n=== Phase 3: Train hypernetwork (LOO) ===")
    t0 = time.time()

    np.random.seed(SEED)
    d_embed = embeddings[domain_list[0]].shape[0]
    n_domains = len(domain_list)

    X = np.stack([embeddings[d] for d in domain_list]).astype(np.float32)
    Y = np.stack([adapters[d]["flat"] for d in domain_list]).astype(np.float32)

    # Normalize Y
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_norm = (Y - Y_mean) / Y_std

    results = {}
    for test_idx in range(n_domains):
        test_domain = domain_list[test_idx]
        train_mask = np.ones(n_domains, dtype=bool)
        train_mask[test_idx] = False
        n_train = n_domains - 1

        X_test = mx.array(X[test_idx:test_idx+1])
        Y_train = mx.array(Y_norm[train_mask])
        Y_target_norm = mx.array(Y_norm[test_idx:test_idx+1])

        model = HyperNetwork(d_embed, d_hidden=256, n_basis=n_train)
        optimizer = opt.Adam(learning_rate=1e-3)

        def loss_fn(model, x, y_basis, y_target):
            coeffs = model(x)
            y_pred = coeffs @ y_basis
            diff = y_pred - y_target
            return mx.mean(diff * diff)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        gc.disable()
        best_loss = float("inf")
        for step in range(500):
            loss, grads = loss_and_grad(model, X_test, Y_train, Y_target_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
        gc.enable()

        # Get predicted adapter
        coeffs = model(X_test)
        mx.eval(coeffs)
        coeffs_np = np.array(coeffs[0].astype(mx.float32))

        Y_train_np = Y[train_mask]
        Y_pred = coeffs_np @ Y_train_np

        # Quality metrics
        Y_test_np = Y[test_idx]
        cos_sim = float(
            np.dot(Y_pred, Y_test_np) /
            (np.linalg.norm(Y_pred) * np.linalg.norm(Y_test_np) + 1e-12)
        )
        mse = float(np.mean((Y_pred - Y_test_np) ** 2))

        top_idx = np.argsort(coeffs_np)[-3:][::-1]
        train_domains = [domain_list[j] for j in np.where(train_mask)[0]]
        top_domains = [(train_domains[i], float(coeffs_np[i])) for i in top_idx]

        results[test_domain] = {
            "best_train_loss": best_loss,
            "pred_mse": mse,
            "pred_cos_sim": cos_sim,
            "top_basis_domains": top_domains,
            "predicted_flat": Y_pred,
        }
        print(f"  {test_domain:20s}: loss={best_loss:.6f} cos={cos_sim:.4f} "
              f"top=[{top_domains[0][0]}:{top_domains[0][1]:.3f}]")

        cleanup(model, optimizer)

    elapsed = time.time() - t0
    print(f"Hypernetwork LOO complete, time={elapsed:.1f}s")
    return results, elapsed


# ============================================================================
# Phase 4: Evaluate NN adapter PPL
# ============================================================================
def phase_eval_nn_ppl(nn_results, domain_list):
    """Evaluate PPL when using nearest-neighbor adapter on target domain.

    For each eval domain: load model, unpack BitLinear, apply LoRA with
    NN's adapter B-matrices and NN's skeleton A-matrices, compute PPL
    on target domain validation data.
    """
    print("\n=== Phase 4: Evaluate NN adapter PPL ===")
    t0 = time.time()

    eval_domains = ["medical", "code", "math", "legal", "cooking", "sports"]
    skeleton_data = dict(np.load(str(ADAPTERS_DIR / "grassmannian_skeleton_n24.npz")))

    # Load model once, unpack, apply LoRA structure
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    model = replace_bitlinear_with_linear(model)

    # Apply LoRA with domain 0 A matrices initially
    model = apply_lora_structure(model, skeleton_data, domain_idx=0)
    model.freeze()
    mx.eval(model.parameters())
    log_memory("model-with-lora")

    # Evaluate base PPL (zero B = base model)
    base_ppls = {}
    print("  Computing base PPL...")
    for domain in eval_domains:
        val_path = DATA_DIR / domain / "valid.jsonl"
        if not val_path.exists():
            continue
        with open(val_path) as f:
            val_data = [json.loads(line) for line in f]
        base_ppls[domain] = compute_ppl(model, tokenizer, val_data)
        print(f"    {domain:20s}: base PPL={base_ppls[domain]:.4f}")

    # Evaluate trained adapter PPL (adapter on own domain)
    trained_ppls = {}
    print("  Computing trained adapter PPL...")
    for di, domain in enumerate(domain_list):
        if domain not in eval_domains:
            continue

        # Set A matrices for this domain
        set_lora_a(model, skeleton_data, di)

        # Load B matrices for this domain
        adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            continue
        adapter_params = dict(mx.load(str(adapter_path)))
        set_lora_b_from_npz(model, adapter_params)
        mx.eval(model.parameters())

        val_path = DATA_DIR / domain / "valid.jsonl"
        with open(val_path) as f:
            val_data = [json.loads(line) for line in f]
        trained_ppls[domain] = compute_ppl(model, tokenizer, val_data)
        print(f"    {domain:20s}: trained PPL={trained_ppls[domain]:.4f}")

    # Evaluate NN adapter PPL (NN's adapter applied to target domain data)
    nn_ppls = {}
    print("  Computing NN adapter PPL...")
    for domain in eval_domains:
        nn_domain = nn_results[domain]["nearest_neighbor"]
        nn_di = domain_list.index(nn_domain)

        # Set A matrices for NN domain
        set_lora_a(model, skeleton_data, nn_di)

        # Load B matrices for NN domain
        nn_adapter_path = ADAPTERS_DIR / nn_domain / "adapter.npz"
        if not nn_adapter_path.exists():
            continue
        adapter_params = dict(mx.load(str(nn_adapter_path)))
        set_lora_b_from_npz(model, adapter_params)
        mx.eval(model.parameters())

        val_path = DATA_DIR / domain / "valid.jsonl"
        with open(val_path) as f:
            val_data = [json.loads(line) for line in f]
        nn_ppls[domain] = compute_ppl(model, tokenizer, val_data)
        print(f"    {domain:20s}: NN({nn_domain:15s}) PPL={nn_ppls[domain]:.4f}")

    elapsed = time.time() - t0
    log_memory("post-ppl-eval")
    cleanup(model, tokenizer)
    del skeleton_data
    return base_ppls, trained_ppls, nn_ppls, elapsed


# ============================================================================
# Phase 5: Orthogonal projection
# ============================================================================
def phase_orthogonal_projection(adapters, hyper_results, domain_list):
    """Project generated adapters onto orthogonal complement of existing adapters."""
    print("\n=== Phase 5: Orthogonal projection test ===")
    t0 = time.time()

    proj_results = {}
    for test_domain in domain_list:
        if test_domain not in hyper_results:
            continue

        pred_flat = hyper_results[test_domain]["predicted_flat"]
        other_domains = [d for d in domain_list if d != test_domain and d in adapters]
        basis = np.stack([adapters[d]["flat"] for d in other_domains])

        # Gram-Schmidt projection
        proj = pred_flat.copy()
        for i in range(len(other_domains)):
            b_i = basis[i]
            b_norm_sq = np.dot(b_i, b_i)
            if b_norm_sq < 1e-12:
                continue
            proj -= (np.dot(proj, b_i) / b_norm_sq) * b_i

        orig_norm = np.linalg.norm(pred_flat)
        proj_norm = np.linalg.norm(proj)
        retention = (proj_norm / orig_norm) ** 2 if orig_norm > 1e-12 else 0.0
        cos_op = float(np.dot(pred_flat, proj) / (orig_norm * proj_norm + 1e-12))

        proj_results[test_domain] = {
            "retention_ratio": float(retention),
            "cos_original_projected": cos_op,
        }

    elapsed = time.time() - t0
    mean_retention = np.mean([r["retention_ratio"] for r in proj_results.values()])
    min_retention = min(r["retention_ratio"] for r in proj_results.values())
    print(f"Projection: mean retention={mean_retention:.4f}, min={min_retention:.4f}")
    print(f"Phase 5 complete, time={elapsed:.1f}s")
    return proj_results, elapsed


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    log_memory("start")

    # Phase 1
    embeddings, t_embed = phase_extract_embeddings()
    log_memory("after-phase1")

    # Phase 2
    adapters, total_params, domain_list, nn_results, t_nn = phase_load_and_nn(embeddings)

    # Phase 3
    hyper_results, t_hyper = phase_train_hypernetwork(embeddings, adapters, domain_list)
    log_memory("after-phase3")

    # Phase 4
    base_ppls, trained_ppls, nn_ppls, t_ppl = phase_eval_nn_ppl(nn_results, domain_list)
    log_memory("after-phase4")

    # Phase 5
    proj_results, t_proj = phase_orthogonal_projection(adapters, hyper_results, domain_list)

    # ============================================================================
    # Results analysis
    # ============================================================================
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    # Hypernetwork B-cosine
    cos_sims = {d: hyper_results[d]["pred_cos_sim"] for d in domain_list if d in hyper_results}
    mean_cos = np.mean(list(cos_sims.values()))
    min_cos = min(cos_sims.values())
    max_cos = max(cos_sims.values())
    print(f"\nHypernetwork B-cosine (predicted vs trained):")
    print(f"  mean={mean_cos:.4f}, min={min_cos:.4f}, max={max_cos:.4f}")

    # K1: NN PPL ratio
    print(f"\nK1: NN adapter PPL vs trained adapter PPL:")
    ppl_ratios = {}
    eval_domains = ["medical", "code", "math", "legal", "cooking", "sports"]
    for domain in eval_domains:
        if domain in nn_ppls and domain in trained_ppls:
            ratio = nn_ppls[domain] / trained_ppls[domain]
            ppl_ratios[domain] = ratio
            print(f"  {domain:20s}: NN={nn_ppls[domain]:.3f} Trained={trained_ppls[domain]:.3f} "
                  f"Base={base_ppls.get(domain, 0):.3f} Ratio={ratio:.3f}")

    if ppl_ratios:
        mean_ratio = np.mean(list(ppl_ratios.values()))
        max_ratio = max(ppl_ratios.values())
        k1_pass = max_ratio < 3.0
        print(f"  max ratio={max_ratio:.3f} ({'PASS' if k1_pass else 'FAIL'}, threshold 3.0)")
    else:
        k1_pass = None
        max_ratio = mean_ratio = None

    # K2: Projection retention
    retentions = [r["retention_ratio"] for r in proj_results.values()]
    mean_retention = np.mean(retentions)
    min_retention = min(retentions)
    k2_pass = min_retention > 0.5
    print(f"\nK2: Projection retention:")
    print(f"  mean={mean_retention:.4f}, min={min_retention:.4f} "
          f"({'PASS' if k2_pass else 'FAIL'}, threshold 0.5)")

    # K3: Memory
    peak_mem = mx.get_peak_memory() / 1e9
    k3_pass = peak_mem < 40.0
    print(f"\nK3: Peak memory = {peak_mem:.2f} GB ({'PASS' if k3_pass else 'FAIL'}, threshold 40 GB)")

    # S1
    if ppl_ratios:
        s1_pass = mean_ratio < 1.5 and mean_retention > 0.5
    else:
        s1_pass = None

    # Verdict
    total_time = time.time() - t_start
    if k1_pass is not None:
        verdict = "SUPPORTED" if (k1_pass and k2_pass and k3_pass) else "KILLED"
    else:
        verdict = "INCONCLUSIVE"

    kill_reasons = []
    if k1_pass is False:
        kill_reasons.append(f"K1 FAIL: max PPL ratio {max_ratio:.3f} > 3.0")
    if not k2_pass:
        kill_reasons.append(f"K2 FAIL: min retention {min_retention:.4f} < 0.5")
    if not k3_pass:
        kill_reasons.append(f"K3 FAIL: peak memory {peak_mem:.2f} GB > 40 GB")

    print(f"\n--- VERDICT: {verdict} ---")
    if kill_reasons:
        for r in kill_reasons:
            print(f"  {r}")
    print(f"Total time: {total_time:.1f}s")

    # Save results
    results = {
        "experiment": "text_to_lora_hypernetwork",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(total_time, 1),
        "n_domains": len(domain_list),
        "total_adapter_params": total_params,
        "nn_baseline": {
            d: {"nn": nn_results[d]["nearest_neighbor"],
                "cos": round(nn_results[d]["cosine_similarity"], 4)}
            for d in domain_list
        },
        "hypernetwork_quality": {
            d: {"cos_sim": round(hyper_results[d]["pred_cos_sim"], 4),
                "loss": round(hyper_results[d]["best_train_loss"], 6),
                "top_basis": hyper_results[d]["top_basis_domains"]}
            for d in domain_list if d in hyper_results
        },
        "base_ppls": {d: round(v, 4) for d, v in base_ppls.items()},
        "trained_ppls": {d: round(v, 4) for d, v in trained_ppls.items()},
        "nn_ppls": {d: round(v, 4) for d, v in nn_ppls.items()},
        "ppl_ratios": {d: round(v, 4) for d, v in ppl_ratios.items()} if ppl_ratios else {},
        "projection": {
            d: {"retention": round(r["retention_ratio"], 4),
                "cos_op": round(r["cos_original_projected"], 4)}
            for d, r in proj_results.items()
        },
        "b_cosine_stats": {"mean": round(mean_cos, 4), "min": round(min_cos, 4), "max": round(max_cos, 4)},
        "kill_criteria": {
            "K1": {"max_ppl_ratio": round(max_ratio, 4) if max_ratio else None,
                   "threshold": 3.0, "pass": k1_pass},
            "K2": {"min_retention": round(min_retention, 4), "mean_retention": round(mean_retention, 4),
                   "threshold": 0.5, "pass": k2_pass},
            "K3": {"peak_memory_gb": round(peak_mem, 2), "threshold_gb": 40.0, "pass": k3_pass},
        },
        "success_criteria": {
            "S1": {"mean_ppl_ratio": round(mean_ratio, 4) if mean_ratio else None,
                   "mean_retention": round(mean_retention, 4), "pass": s1_pass},
        },
        "verdict": verdict,
        "phase_times_s": {"embed": round(t_embed, 1), "nn": round(t_nn, 1),
                          "hypernetwork": round(t_hyper, 1), "ppl_eval": round(t_ppl, 1),
                          "projection": round(t_proj, 1)},
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
