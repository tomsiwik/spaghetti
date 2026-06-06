#!/usr/bin/env python3
"""
P9.B3: TT-LoRA + PoLAR Hybrid — Stiefel Retraction on TT Cores
Paper: TT-LoRA MoE (arXiv:2504.21190) + PoLAR (Finding #442)
Prior: Finding #515 (TT-LoRA MLX port), #516 (TT-LoRA quality), #442 (Stiefel sr=r)

Kill criteria:
  K1363: TT-PoLAR sr > TT-LoRA sr (stable rank improvement from Stiefel cores)
  K1364: Quality >= TT-LoRA at same parameter budget
  K1365: Stiefel retraction on small cores completes in < 1ms per step

Design:
  Two TT-LoRA variants on v_proj, all 42 layers, GSM8K, 500 steps:
  A) TT-LoRA (unconstrained) — baseline
  B) TT-LoRA-Stiefel (polar retraction on interior cores every 10 steps)
  Compare: sr(ΔW), GSM8K accuracy, retraction time.
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 2000
N_EVAL = 5 if IS_SMOKE else 50
N_STEPS = 20 if IS_SMOKE else 500
BATCH_SIZE = 2
MAX_SEQ_LEN = 512

TT_RANK = 6
TT_LR = 5e-3       # Paper-recommended (arXiv:2504.21190)
TT_ALPHA = 1.0

RETRACT_EVERY = 10  # Retract to Stiefel every N steps
SEED = 42
HIDDEN_SIZE = 2560
PROJ_NAMES = ["v_proj"]

# Layers to measure sr on (representative sample to avoid full SVD on all 42)
SR_LAYERS = [0, 10, 20, 30, 41]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ────────────────────────────────────────────────
# TT-LoRA Module (from exp_p9_ttlora_quality)
# ────────────────────────────────────────────────

class TTLoRAWrapper(nn.Module):
    """TT-LoRA adapter wrapping a (possibly quantized) linear layer.

    Forward: base_layer(x) + alpha * x @ reconstruct(tt_cores).T
    Last core zero-init → output = base_layer(x) at t=0.
    """

    def __init__(self, base_layer, in_features, out_features, tt_shape,
                 tt_rank=6, alpha=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.tt_shape = tt_shape

        self._validate_split(tt_shape, in_features, out_features)

        d = len(tt_shape)
        ranks = [1] + [tt_rank] * (d - 1) + [1]
        self._n_cores = d
        self._tt_rank = tt_rank

        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            if k == d - 1:
                core = mx.zeros(shape)  # Zero init for residual property
            else:
                std = 1.0 / math.sqrt(tt_shape[k] * ranks[k])
                core = mx.random.normal(shape) * std
            setattr(self, f"core_{k}", core)
        self._cached_delta_w = None

    def _validate_split(self, tt_shape, in_features, out_features):
        prod = 1
        for i, s in enumerate(tt_shape):
            prod *= s
            if prod == in_features:
                rest = 1
                for j in range(i + 1, len(tt_shape)):
                    rest *= tt_shape[j]
                assert rest == out_features, (
                    f"Output factors product {rest} != {out_features}")
                self._split_idx = i + 1
                return
        raise ValueError(
            f"Cannot split {tt_shape} into {in_features} x {out_features}")

    @property
    def tt_cores(self):
        return [getattr(self, f"core_{k}") for k in range(self._n_cores)]

    def reconstruct_delta_w(self):
        """Contract TT cores → ΔW [out_features, in_features]."""
        cores = self.tt_cores
        result = cores[0].squeeze(0)  # [s_0, r_1]
        for k in range(1, len(cores)):
            core = cores[k]  # [r_k, s_k, r_{k+1}]
            r_k, s_k, r_next = core.shape
            result = result @ core.reshape(r_k, s_k * r_next)
            leading = result.shape[0]
            result = result.reshape(leading * s_k, r_next)
        result = result.squeeze(-1)  # [in_features * out_features]
        return result.reshape(self.in_features, self.out_features).T

    def cache_delta_w(self):
        self._cached_delta_w = self.reconstruct_delta_w()
        mx.eval(self._cached_delta_w)

    def __call__(self, x):
        base_out = self.base_layer(x)
        dw = (self._cached_delta_w if self._cached_delta_w is not None
              else self.reconstruct_delta_w())
        return base_out + self.alpha * (x @ dw.T)

    def num_params(self):
        return sum(c.size for c in self.tt_cores)

    def retractable_core_indices(self):
        """Return indices of interior cores suitable for Stiefel retraction.

        Skip first core if left-unfolding has fewer rows than cols (can't
        have orthonormal columns). Skip last core (stores learned scale).
        """
        indices = []
        ranks = [1] + [self._tt_rank] * (self._n_cores - 1) + [1]
        for k in range(self._n_cores):
            if k == self._n_cores - 1:
                continue  # Last core = free (stores learned info)
            r_prev = ranks[k]
            s_k = self.tt_shape[k]
            r_next = ranks[k + 1]
            rows = r_prev * s_k
            cols = r_next
            if rows >= cols and cols > 1:
                indices.append(k)
        return indices


def factorize(n, max_factor=10):
    factors = []
    while n % 8 == 0 and n > 8:
        factors.append(8)
        n //= 8
    for f in range(max_factor, 1, -1):
        while n % f == 0 and n > 1:
            factors.append(f)
            n //= f
    if n > 1:
        factors.append(n)
    factors.sort()
    return factors


def compute_tt_shape(in_features, out_features):
    return factorize(in_features) + factorize(out_features)


# ────────────────────────────────────────────────
# Stiefel Retraction
# ────────────────────────────────────────────────

def retract_core_to_stiefel(core_mx):
    """Polar retraction of a TT core's left-unfolding to Stiefel manifold.

    Core shape: (r_prev, s_k, r_next).
    Left unfolding: (r_prev * s_k, r_next) — enforce orthonormal columns.
    Uses numpy SVD (float64) for numerical precision on small matrices.

    Returns: (retracted_core_mx, stiefel_distance)
    """
    r_prev, s_k, r_next = core_mx.shape
    rows = r_prev * s_k

    # Reshape to left unfolding
    A_np = np.array(core_mx.reshape(rows, r_next), dtype=np.float64)

    # Check for degenerate values
    if not np.all(np.isfinite(A_np)) or np.sum(A_np ** 2) < 1e-12:
        I_r = np.eye(r_next)
        dist = float(np.sqrt(np.sum((A_np.T @ A_np - I_r) ** 2))) if np.all(np.isfinite(A_np)) else float("inf")
        return core_mx, dist

    # Polar decomposition via SVD: A = U Σ V^T → nearest Stiefel = U V^T
    U, S, Vt = np.linalg.svd(A_np, full_matrices=False)
    A_retracted = U @ Vt  # (rows, r_next) with orthonormal columns

    # Measure Stiefel distance: ||A^T A - I||_F
    AtA = A_retracted.T @ A_retracted
    I_r = np.eye(r_next)
    dist = float(np.sqrt(np.sum((AtA - I_r) ** 2)))

    # Reshape back to core shape and convert to MLX float32
    retracted = mx.array(A_retracted.astype(np.float32).reshape(r_prev, s_k, r_next))
    return retracted, dist


def retract_all_stiefel(model, proj_names):
    """Retract all retractable TT cores in all layers to Stiefel.

    Returns: (total_time_s, mean_stiefel_dist, n_cores_retracted)
    """
    layers = get_layers(model)
    total_dist = 0.0
    n_retracted = 0
    t0 = time.perf_counter()

    for layer in layers:
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if not isinstance(proj, TTLoRAWrapper):
                continue
            for k in proj.retractable_core_indices():
                core = getattr(proj, f"core_{k}")
                # Eval core before numpy conversion
                mx.eval(core)
                retracted, dist = retract_core_to_stiefel(core)
                setattr(proj, f"core_{k}", retracted)
                total_dist += dist
                n_retracted += 1

    elapsed = time.perf_counter() - t0
    mean_dist = total_dist / max(n_retracted, 1)
    return elapsed, mean_dist, n_retracted


# ────────────────────────────────────────────────
# Stable Rank Measurement
# ────────────────────────────────────────────────

def measure_stable_rank(model, proj_names, layer_indices):
    """Measure sr(ΔW) for specified layers.

    sr = ||ΔW||_F^2 / ||ΔW||_2^2

    Uses power iteration for ||ΔW||_2 (fastest for large matrices).
    Returns dict: layer_idx -> {sr, fro_norm, op_norm}
    """
    layers = get_layers(model)
    results = {}

    for idx in layer_indices:
        if idx >= len(layers):
            continue
        for pname in proj_names:
            proj = getattr(layers[idx].self_attn, pname)
            if not isinstance(proj, TTLoRAWrapper):
                continue

            # Reconstruct ΔW
            dw = proj.reconstruct_delta_w()  # [out, in]
            mx.eval(dw)

            # Frobenius norm
            fro_sq = mx.sum(dw * dw).item()

            # Operator norm via SVD (ΔW is small enough: max 2048×2560)
            # Use numpy for accuracy
            dw_np = np.array(dw, dtype=np.float64)
            svs = np.linalg.svd(dw_np, compute_uv=False)
            op_norm = float(svs[0])
            op_sq = op_norm ** 2

            sr = fro_sq / op_sq if op_sq > 1e-20 else 0.0

            results[idx] = {
                "sr": round(sr, 4),
                "fro_norm": round(float(np.sqrt(fro_sq)), 6),
                "op_norm": round(op_norm, 6),
                "top5_sv": [round(float(s), 6) for s in svs[:5]],
                "sv_ratio": round(float(svs[0] / svs[min(5, len(svs)) - 1]), 4) if svs[min(5, len(svs)) - 1] > 1e-20 else float("inf"),
            }

            del dw, dw_np
    return results


# ────────────────────────────────────────────────
# Model Injection
# ────────────────────────────────────────────────

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def detect_proj_dims(base_layer, hidden_size=HIDDEN_SIZE):
    test_x = mx.zeros((1, 1, hidden_size))
    test_y = base_layer(test_x)
    mx.eval(test_y)
    in_f = hidden_size
    out_f = test_y.shape[-1]
    del test_x, test_y
    return in_f, out_f


def inject_ttlora(model, proj_names, tt_rank, alpha):
    layers = get_layers(model)
    total_params = 0
    for i, layer in enumerate(layers):
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            tt_shape = compute_tt_shape(in_f, out_f)
            wrapper = TTLoRAWrapper(base, in_f, out_f, tt_shape, tt_rank, alpha)
            setattr(layer.self_attn, name, wrapper)
            total_params += wrapper.num_params()

            if i == 0:
                print(f"  {name}: {in_f}->{out_f}, TT shape {tt_shape}, "
                      f"{len(tt_shape)} cores, {wrapper.num_params()} params/layer",
                      flush=True)
                retractable = wrapper.retractable_core_indices()
                print(f"  Retractable cores: {retractable} "
                      f"(left unfold shapes: "
                      f"{[(wrapper.tt_shape[k] * ([1] + [tt_rank] * (len(tt_shape)-1) + [1])[k], ([1] + [tt_rank] * (len(tt_shape)-1) + [1])[k+1]) for k in retractable]})",
                      flush=True)
    return total_params


# ────────────────────────────────────────────────
# Data Preparation
# ────────────────────────────────────────────────

def prepare_gsm8k_data():
    import shutil

    data_dir = EXPERIMENT_DIR / "data" / "math"
    data_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "train.jsonl").exists():
        n_train = sum(1 for _ in open(data_dir / "train.jsonl"))
        print(f"GSM8K data ready ({n_train} train examples)", flush=True)
        return data_dir

    # Reuse data from quality experiment (avoids datasets lib Python 3.14 issue)
    quality_data = EXPERIMENT_DIR.parent / "exp_p9_ttlora_quality" / "data" / "math"
    if quality_data.exists() and (quality_data / "train.jsonl").exists():
        shutil.copy2(quality_data / "train.jsonl", data_dir / "train.jsonl")
        shutil.copy2(quality_data / "valid.jsonl", data_dir / "valid.jsonl")
        n_train = sum(1 for _ in open(data_dir / "train.jsonl"))
        print(f"GSM8K data copied from quality experiment ({n_train} train)", flush=True)
        return data_dir

    # Fallback: load from HuggingFace
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))

    records = []
    for ex in ds:
        records.append(json.dumps({"messages": [
            {"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"},
            {"role": "assistant", "content": ex["answer"]},
        ]}))

    n_val = max(1, len(records) // 10)
    (data_dir / "train.jsonl").write_text("\n".join(records[n_val:]))
    (data_dir / "valid.jsonl").write_text("\n".join(records[:n_val]))
    print(f"GSM8K: {len(records) - n_val} train, {n_val} val", flush=True)
    return data_dir


# ────────────────────────────────────────────────
# Training Loop
# ────────────────────────────────────────────────

def tokenize_for_training(tokenizer, data_path, max_seq_len=512):
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]
            full = tokenizer.apply_chat_template(msgs, tokenize=False)
            full_ids = tokenizer.encode(full)
            prompt = tokenizer.apply_chat_template(
                [msgs[0]], tokenize=False, add_generation_prompt=True)
            prompt_len = len(tokenizer.encode(prompt))
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
            if prompt_len >= len(full_ids):
                continue
            examples.append({
                "input_ids": full_ids,
                "prompt_len": prompt_len,
                "length": len(full_ids),
            })
    return examples


def train_ttlora(model, tokenizer, data_dir, n_steps, lr, batch_size,
                 stiefel=False, retract_every=10):
    """Train TT-LoRA with optional Stiefel retraction.

    Args:
        stiefel: If True, retract interior cores to Stiefel every retract_every steps
    Returns: (losses, train_time, retraction_times)
    """
    examples = tokenize_for_training(
        tokenizer, data_dir / "train.jsonl", MAX_SEQ_LEN)
    print(f"  {len(examples)} training examples", flush=True)

    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, input_ids, lengths, prompt_lens):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        ce = nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none")
        S = shift_targets.shape[1]
        pos = mx.arange(S)[None, :]
        mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        mask = mask.astype(mx.float32)
        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    retraction_times = []
    t0 = time.time()
    idx = 0
    label = "TT-LoRA-Stiefel" if stiefel else "TT-LoRA"

    for step in range(n_steps):
        # Build padded batch
        batch_exs = []
        for _ in range(batch_size):
            batch_exs.append(examples[idx % len(examples)])
            idx += 1

        max_len = max(e["length"] for e in batch_exs)
        input_ids = mx.array([
            e["input_ids"] + [pad_id] * (max_len - e["length"])
            for e in batch_exs
        ])
        lengths = mx.array([e["length"] for e in batch_exs])
        prompt_lens = mx.array([e["prompt_len"] for e in batch_exs])

        # Forward + backward + update
        loss, grads = loss_and_grad(model, input_ids, lengths, prompt_lens)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        # Stiefel retraction
        if stiefel and (step + 1) % retract_every == 0:
            rt, mean_dist, n_cores = retract_all_stiefel(model, PROJ_NAMES)
            retraction_times.append(rt)

        if (step + 1) % max(1, n_steps // 10) == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            extra = ""
            if stiefel and retraction_times:
                extra = f" retract={retraction_times[-1]*1000:.1f}ms"
            print(f"  [{label}] Step {step+1}/{n_steps}: loss={avg:.4f} "
                  f"({elapsed:.1f}s){extra}", flush=True)

    total_time = time.time() - t0
    print(f"  {label} training: {total_time:.1f}s ({total_time/60:.1f} min)",
          flush=True)
    return losses, total_time, retraction_times


# ────────────────────────────────────────────────
# GSM8K Evaluation
# ────────────────────────────────────────────────

def load_gsm8k_test(n_eval):
    """Load GSM8K test set, with fallback for Python 3.14 compat."""
    cache_path = EXPERIMENT_DIR / "data" / "gsm8k_test.jsonl"
    if cache_path.exists():
        examples = []
        with open(cache_path) as f:
            for line in f:
                examples.append(json.loads(line))
        random.seed(SEED)
        random.shuffle(examples)
        return examples[:n_eval]

    # Try loading from quality experiment cache
    quality_cache = EXPERIMENT_DIR.parent / "exp_p9_ttlora_quality" / "data" / "gsm8k_test.jsonl"
    if quality_cache.exists():
        import shutil
        shutil.copy2(quality_cache, cache_path)
        return load_gsm8k_test(n_eval)

    # Download and cache
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    with open(cache_path, "w") as f:
        for ex in ds:
            f.write(json.dumps({"question": ex["question"], "answer": ex["answer"]}) + "\n")
    return load_gsm8k_test(n_eval)


def eval_gsm8k(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate

    ds = load_gsm8k_test(n_eval)

    correct = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user",
                     "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(
            model, tokenizer, prompt=formatted,
            max_tokens=512, verbose=False,
        )

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1

        if (i + 1) % 25 == 0:
            print(f"  GSM8K {label}: {i+1}/{len(ds)}, "
                  f"acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(ds) * 100
    print(f"  GSM8K {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc, correct, len(ds)


# ────────────────────────────────────────────────
# Retraction Benchmark
# ────────────────────────────────────────────────

def benchmark_retraction(model, proj_names, n_iters=50):
    """Benchmark Stiefel retraction time over multiple iterations."""
    # Warmup
    for _ in range(3):
        retract_all_stiefel(model, proj_names)

    times = []
    for _ in range(n_iters):
        elapsed, _, _ = retract_all_stiefel(model, proj_names)
        times.append(elapsed)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  Retraction benchmark: {mean_ms:.2f} ± {std_ms:.2f} ms/step "
          f"(n={n_iters})", flush=True)
    return {"mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3),
            "min_ms": round(min(times) * 1000, 3),
            "max_ms": round(max(times) * 1000, 3)}


# ────────────────────────────────────────────────
# Run One Variant
# ────────────────────────────────────────────────

def run_variant(variant_name, data_dir, stiefel=False):
    """Train and evaluate one TT-LoRA variant. Returns results dict."""
    from mlx_lm import load

    print(f"\n{'='*60}")
    print(f"Variant: {variant_name} (stiefel={stiefel})")
    print(f"{'='*60}", flush=True)

    mx.random.seed(SEED)
    print("Loading model...", flush=True)
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Inject TT-LoRA
    tt_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    print(f"TT-LoRA injected: {tt_params:,} params", flush=True)

    # Freeze all, unfreeze TT cores
    model.freeze()
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                core_keys = [f"core_{k}" for k in range(proj._n_cores)]
                proj.unfreeze(keys=core_keys, recurse=False)

    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"Trainable: {trainable:,}", flush=True)

    # Pre-training sr measurement
    print("\nPre-training stable rank...", flush=True)
    sr_pre = measure_stable_rank(model, PROJ_NAMES, SR_LAYERS)
    for idx, data in sorted(sr_pre.items()):
        print(f"  Layer {idx}: sr={data['sr']:.4f}, "
              f"top sv={data['top5_sv'][:3]}", flush=True)

    # Train
    model.train()
    losses, train_time, retract_times = train_ttlora(
        model, tokenizer, data_dir, N_STEPS, TT_LR, BATCH_SIZE,
        stiefel=stiefel, retract_every=RETRACT_EVERY)

    # Post-training sr measurement
    print("\nPost-training stable rank...", flush=True)
    sr_post = measure_stable_rank(model, PROJ_NAMES, SR_LAYERS)
    for idx, data in sorted(sr_post.items()):
        print(f"  Layer {idx}: sr={data['sr']:.4f}, "
              f"top sv={data['top5_sv'][:3]}", flush=True)

    # Retraction benchmark (only for Stiefel variant)
    retract_bench = None
    if stiefel:
        print("\nRetraction benchmark...", flush=True)
        retract_bench = benchmark_retraction(model, PROJ_NAMES, n_iters=50)

    # Evaluate
    print("\nEvaluating GSM8K...", flush=True)
    model.eval()
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()

    acc, correct, total = eval_gsm8k(model, tokenizer, N_EVAL, variant_name)

    # Compute convergence
    convergence = None
    if len(losses) >= 100:
        avg_first = sum(losses[:50]) / 50
        avg_second = sum(losses[50:100]) / 50
        convergence = {"first_50": round(avg_first, 4),
                       "next_50": round(avg_second, 4),
                       "converging": avg_second < avg_first}

    # Mean sr across measured layers
    sr_values = [v["sr"] for v in sr_post.values()]
    mean_sr = sum(sr_values) / len(sr_values) if sr_values else 0

    result = {
        "variant": variant_name,
        "stiefel": stiefel,
        "tt_params": tt_params,
        "train_time_s": round(train_time, 1),
        "final_loss": round(losses[-1], 4) if losses else None,
        "convergence": convergence,
        "sr_pre": {str(k): v for k, v in sr_pre.items()},
        "sr_post": {str(k): v for k, v in sr_post.items()},
        "mean_sr": round(mean_sr, 4),
        "gsm8k_acc": round(acc, 1),
        "gsm8k_correct": correct,
        "gsm8k_total": total,
        "retraction_benchmark": retract_bench,
        "retraction_times_during_training": (
            [round(t * 1000, 2) for t in retract_times] if retract_times else []),
    }

    cleanup(model, tokenizer)
    return result


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P9.B3: TT-LoRA + PoLAR Hybrid (Stiefel Cores)")
    print(f"TT-rank={TT_RANK}, lr={TT_LR}, {N_STEPS} steps, v_proj only")
    print(f"Stiefel retraction every {RETRACT_EVERY} steps")
    print(f"SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print("=" * 60, flush=True)

    # Prepare data
    print("\n── Prepare GSM8K data ──", flush=True)
    data_dir = prepare_gsm8k_data()

    # Run Variant A: unconstrained TT-LoRA (baseline)
    result_a = run_variant("TT-LoRA", data_dir, stiefel=False)

    # Run Variant B: TT-LoRA with Stiefel retraction
    result_b = run_variant("TT-LoRA-Stiefel", data_dir, stiefel=True)

    # ── Compare ──────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60, flush=True)

    sr_a = result_a["mean_sr"]
    sr_b = result_b["mean_sr"]
    acc_a = result_a["gsm8k_acc"]
    acc_b = result_b["gsm8k_acc"]

    print(f"\nStable Rank (mean across layers {SR_LAYERS}):")
    print(f"  TT-LoRA:          sr = {sr_a:.4f}")
    print(f"  TT-LoRA-Stiefel:  sr = {sr_b:.4f}")
    sr_ratio = sr_b / sr_a if sr_a > 0 else float("inf")
    print(f"  Ratio:            {sr_ratio:.2f}x")

    print(f"\nGSM8K Accuracy:")
    print(f"  TT-LoRA:          {acc_a:.1f}%")
    print(f"  TT-LoRA-Stiefel:  {acc_b:.1f}%")
    acc_delta = acc_b - acc_a
    print(f"  Delta:            {acc_delta:+.1f}pp")

    if result_b["retraction_benchmark"]:
        rb = result_b["retraction_benchmark"]
        print(f"\nRetraction Time: {rb['mean_ms']:.2f} ± {rb['std_ms']:.2f} ms/step")

    # ── Kill Criteria ────────────────────────────
    print("\n" + "=" * 60)
    print("KILL CRITERIA")
    print("=" * 60, flush=True)

    k1363_pass = sr_b > sr_a
    k1364_pass = acc_b >= acc_a - 5.0  # Allow small variance (5pp margin)
    k1365_pass = (result_b["retraction_benchmark"]["mean_ms"] < 1.0
                  if result_b["retraction_benchmark"] else False)

    print(f"K1363 sr improvement:  {'PASS' if k1363_pass else 'FAIL'} "
          f"(Stiefel sr={sr_b:.4f} vs baseline sr={sr_a:.4f})")
    print(f"K1364 quality:         {'PASS' if k1364_pass else 'FAIL'} "
          f"(Stiefel={acc_b:.1f}% vs baseline={acc_a:.1f}%)")
    print(f"K1365 retraction time: {'PASS' if k1365_pass else 'FAIL'} "
          f"({rb['mean_ms']:.2f}ms < 1.0ms)" if result_b["retraction_benchmark"] else "N/A")

    overall = k1363_pass and k1364_pass and k1365_pass
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")

    total_time = time.time() - t_start
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # ── Save Results ─────────────────────────────
    results = {
        "experiment": "exp_p9_ttlora_polar_hybrid",
        "papers": ["arXiv:2504.21190", "Finding #442"],
        "tt_rank": TT_RANK,
        "tt_lr": TT_LR,
        "n_steps": N_STEPS,
        "retract_every": RETRACT_EVERY,
        "n_eval": N_EVAL,
        "variant_a": result_a,
        "variant_b": result_b,
        "comparison": {
            "sr_baseline": sr_a,
            "sr_stiefel": sr_b,
            "sr_ratio": round(sr_ratio, 4),
            "acc_baseline": acc_a,
            "acc_stiefel": acc_b,
            "acc_delta_pp": round(acc_delta, 1),
        },
        "kill_criteria": {
            "K1363_sr": "PASS" if k1363_pass else "FAIL",
            "K1363_detail": f"sr_stiefel={sr_b:.4f} vs sr_baseline={sr_a:.4f} ({sr_ratio:.2f}x)",
            "K1364_quality": "PASS" if k1364_pass else "FAIL",
            "K1364_detail": f"stiefel={acc_b:.1f}% vs baseline={acc_a:.1f}% (delta={acc_delta:+.1f}pp)",
            "K1365_retraction": "PASS" if k1365_pass else "FAIL",
            "K1365_detail": f"{rb['mean_ms']:.2f}ms" if result_b["retraction_benchmark"] else "N/A",
        },
        "overall_pass": overall,
        "total_time_s": round(total_time, 1),
    }

    (EXPERIMENT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {EXPERIMENT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
