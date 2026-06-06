#!/usr/bin/env python3
"""
BitNet-2B Cosine Convergence Trajectory Experiment

Tests whether |cos| between LoRA adapters inflates toward 0.05+ as training
progresses to full convergence, or whether it plateaus below 0.01 (supporting
the structural orthogonality claim for ternary bases).

Prior findings:
  - |cos|=0.001 at 400 steps (bitnet_2b_real_composition, single seed)
  - Macro Qwen FP16 converged adapters: cos=0.142 (142x higher)
  - Ternary adapters compose 4.4% better than FP16 (micro, 3 seeds)

Kill criteria:
  K1: mean |cos| at convergence (loss plateau) exceeds 0.05
  K2: cos trajectory shows monotonic increase with no plateau

Design:
  - 5 domains: medical, code, math, legal, creative
  - Train each to 2000 steps (well past expected convergence ~400-800 steps)
  - Checkpoint adapter weights every 100 steps
  - Compute all 10 pairwise |cos| at each checkpoint
  - Compute composition PPL at steps {200, 400, 800, 1200, 1600, 2000}
  - Detect convergence: sliding window of 200 steps, <1% improvement = plateau

Platform: Apple Silicon MLX, $0 compute.
Expected runtime: ~2-3 hours.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128  # match prior ternary convergence experiment
LEARNING_RATE = 1e-4
VAL_BATCHES = 25

TOTAL_STEPS = 2000
COSINE_CHECKPOINT_INTERVAL = 100  # compute |cos| every 100 steps
PPL_CHECKPOINT_STEPS = {200, 400, 800, 1200, 1600, 2000}  # composition PPL

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from prior experiments where available
PRIOR_DATA_DIRS = [
    Path(__file__).parent.parent / "bitnet_ternary_convergence" / "data",
    Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data",
]

# 5 domains
DOMAINS = {
    "medical": {
        "hf_dataset": "medalpaca/medical_meadow_medical_flashcards",
        "text_key": "output",
        "max_samples_train": 800,
        "max_samples_val": 100,
        "prior_name": "medical",
    },
    "code": {
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "max_samples_train": 800,
        "max_samples_val": 100,
        "prior_name": ["code", "python"],  # different names in prior experiments
    },
    "math": {
        "hf_dataset": "gsm8k",
        "hf_subset": "main",
        "text_key": "answer",
        "max_samples_train": 800,
        "max_samples_val": 100,
        "prior_name": "math",
    },
    "legal": {
        "hf_dataset": "jonathanli/law-stack-exchange",
        "text_key": "body",
        "max_samples_train": 500,
        "max_samples_val": 80,
        "prior_name": "legal",
    },
    "creative": {
        "hf_dataset": "roneneldan/TinyStories",
        "text_key": "text",
        "max_samples_train": 800,
        "max_samples_val": 100,
        "prior_name": "creative",
    },
}


# ===========================================================================
# Ternary weight unpacking (from bitnet_2b_real_composition)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
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


# ===========================================================================
# LoRA helpers (from bitnet_2b_real_composition)
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-scale, high=scale, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


# ===========================================================================
# Cosine similarity computation
# ===========================================================================
def compute_pairwise_cosines(adapter_dict):
    """Compute all pairwise |cos| between adapters.

    Args:
        adapter_dict: {domain_name: {param_name: mx.array}}

    Returns:
        list of dicts with pair name and |cos| value, plus mean |cos|
    """
    names = list(adapter_dict.keys())
    # Flatten each adapter to a single vector
    vectors = {}
    for name, params in adapter_dict.items():
        vec = mx.concatenate([v.reshape(-1) for v in params.values()])
        vectors[name] = vec

    cosines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi, vj = vectors[names[i]], vectors[names[j]]
            cos = mx.abs(
                mx.sum(vi * vj) / (mx.sqrt(mx.sum(vi**2)) * mx.sqrt(mx.sum(vj**2)) + 1e-12)
            )
            mx.eval(cos)
            cosines.append({
                "pair": f"{names[i]}-{names[j]}",
                "abs_cos": round(cos.item(), 6),
            })

    mean_cos = sum(c["abs_cos"] for c in cosines) / len(cosines) if cosines else 0.0
    return cosines, mean_cos


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name, domain_config):
    """Download HF dataset or symlink from prior experiment."""
    data_dir = DATA_DIR / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        print(f"  {domain_name}: data exists")
        return data_dir

    # Try to symlink from prior experiments
    prior_names = domain_config.get("prior_name", domain_name)
    if isinstance(prior_names, str):
        prior_names = [prior_names]

    for prior_dir in PRIOR_DATA_DIRS:
        for pname in prior_names:
            src = prior_dir / pname
            if (src / "train.jsonl").exists() and (src / "valid.jsonl").exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(src / "train.jsonl", train_path)
                shutil.copy2(src / "valid.jsonl", valid_path)
                print(f"  {domain_name}: copied from {src}")
                return data_dir

    # Download fresh
    from datasets import load_dataset as hf_load
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {domain_config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]

    ds = hf_load(domain_config["hf_dataset"], **kwargs)
    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    text_key = domain_config["text_key"]
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response", "question"]:
            if alt in split_data.column_names:
                text_key = alt
                break

    max_train = domain_config["max_samples_train"]
    max_val = domain_config["max_samples_val"]
    texts = []
    for row in split_data:
        t = row[text_key]
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
        if len(texts) >= max_train + max_val:
            break

    train_texts = texts[:max_train]
    val_texts = texts[max_train:max_train + max_val]

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")
    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path, max_batches=25):
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0

    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 100))


def compose_adapters(adapter_list, scale_per_adapter=None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


# ===========================================================================
# Convergence detection
# ===========================================================================
def detect_convergence(losses, window=200, threshold=0.01):
    """Detect if loss has plateaued.

    Returns (converged: bool, plateau_step: int or None)
    converged = True if improvement over last `window` steps < threshold.
    """
    if len(losses) < window * 2:
        return False, None

    # Compare average of [t-2w, t-w] vs [t-w, t]
    prev_window = losses[-2 * window:-window]
    curr_window = losses[-window:]
    prev_avg = sum(prev_window) / len(prev_window)
    curr_avg = sum(curr_window) / len(curr_window)

    improvement = (prev_avg - curr_avg) / prev_avg
    if improvement < threshold:
        return True, len(losses) - window
    return False, None


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_experiment_start = time.time()

    results = {
        "experiment": "bitnet_cosine_convergence_trajectory",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "total_steps": TOTAL_STEPS,
        "cosine_interval": COSINE_CHECKPOINT_INTERVAL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("BitNet-2B Cosine Convergence Trajectory")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    # Apply LoRA
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")
    results["trainable_params"] = trainable

    # ------------------------------------------------------------------
    # Phase 1: Prepare data
    # ------------------------------------------------------------------
    print("\n[Phase 1] Preparing data...")
    data_dirs = {}
    for dname, dconfig in DOMAINS.items():
        data_dirs[dname] = prepare_domain_data(dname, dconfig)

    # Tokenize all training data upfront
    domain_tokens = {}
    for dname, data_dir in data_dirs.items():
        texts = []
        with open(data_dir / "train.jsonl") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        tokens = []
        for text in texts:
            toks = tokenizer.encode(text)
            if len(toks) > 2:
                tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))
        domain_tokens[dname] = tokens
        print(f"  {dname}: {len(tokens)} sequences")

    # ------------------------------------------------------------------
    # Phase 2: Base PPL
    # ------------------------------------------------------------------
    print("\n[Phase 2] Base model PPL...")
    zero_lora_params(model)
    base_ppls = {}
    for dname, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[dname] = ppl
        print(f"  {dname}: {ppl:.2f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 3: Train all adapters with checkpointing
    # ------------------------------------------------------------------
    print("\n[Phase 3] Training 5 adapters to 2000 steps with cosine checkpoints...")

    # Storage for all checkpointed adapter params
    # checkpoint_params[step][domain] = {param_name: array}
    checkpoint_params = {}
    cosine_trajectory = {}  # step -> {cosines: [...], mean_cos: float}
    domain_loss_curves = {}
    domain_convergence = {}

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for dname in DOMAINS:
        print(f"\n  --- Training {dname} ({TOTAL_STEPS} steps) ---")
        zero_lora_params(model)

        optimizer = opt.Adam(learning_rate=LEARNING_RATE)
        tokens = domain_tokens[dname]
        losses = []

        t_start = time.time()
        for step in range(1, TOTAL_STEPS + 1):
            idx = (step - 1) % len(tokens)
            tok = tokens[idx]
            x = tok[:-1][None, :]
            y = tok[1:][None, :]

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())

            # Print progress
            if step % 200 == 0 or step == 1:
                avg = sum(losses[-50:]) / min(len(losses), 50)
                elapsed = time.time() - t_start
                print(f"    Step {step}/{TOTAL_STEPS}: loss={loss.item():.4f} "
                      f"avg50={avg:.4f} ({elapsed:.0f}s)")

            # Checkpoint adapter params at every COSINE_CHECKPOINT_INTERVAL
            if step % COSINE_CHECKPOINT_INTERVAL == 0:
                if step not in checkpoint_params:
                    checkpoint_params[step] = {}
                checkpoint_params[step][dname] = get_lora_params(model)

        train_time = time.time() - t_start
        domain_loss_curves[dname] = {
            "first_50": round(sum(losses[:50]) / 50, 4),
            "last_50": round(sum(losses[-50:]) / 50, 4),
            "losses_every_100": [round(sum(losses[max(0, i-50):i+50]) / min(100, i+50), 4)
                                  for i in range(49, len(losses), 100)],
            "train_time_s": round(train_time, 1),
        }

        # Detect convergence
        converged, plateau_step = detect_convergence(losses)
        domain_convergence[dname] = {
            "converged": converged,
            "plateau_step": plateau_step,
            "final_loss": round(losses[-1], 4),
        }
        print(f"  {dname}: loss {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"({'CONVERGED at ~' + str(plateau_step) if converged else 'still improving'}) "
              f"in {train_time:.0f}s")

    results["domain_loss_curves"] = domain_loss_curves
    results["domain_convergence"] = domain_convergence

    # ------------------------------------------------------------------
    # Phase 4: Compute cosine trajectory
    # ------------------------------------------------------------------
    print("\n[Phase 4] Computing cosine similarity trajectory...")

    steps_with_all_domains = sorted([
        s for s in checkpoint_params
        if len(checkpoint_params[s]) == len(DOMAINS)
    ])

    print(f"  Steps with all 5 domain checkpoints: {steps_with_all_domains}")

    for step in steps_with_all_domains:
        cosines, mean_cos = compute_pairwise_cosines(checkpoint_params[step])
        cosine_trajectory[step] = {
            "mean_abs_cos": round(mean_cos, 6),
            "max_abs_cos": round(max(c["abs_cos"] for c in cosines), 6),
            "min_abs_cos": round(min(c["abs_cos"] for c in cosines), 6),
            "cosines": cosines,
        }
        print(f"  Step {step}: mean |cos|={mean_cos:.6f}, "
              f"max={max(c['abs_cos'] for c in cosines):.6f}")

    results["cosine_trajectory"] = {
        str(k): v for k, v in cosine_trajectory.items()
    }

    # ------------------------------------------------------------------
    # Phase 5: Composition PPL at checkpoints
    # ------------------------------------------------------------------
    print("\n[Phase 5] Composition PPL at checkpoints...")

    composition_trajectory = {}
    for step in sorted(PPL_CHECKPOINT_STEPS):
        if step not in checkpoint_params or len(checkpoint_params[step]) < len(DOMAINS):
            print(f"  Step {step}: skipped (not all domains checkpointed)")
            continue

        adapter_list = list(checkpoint_params[step].values())
        merged = compose_adapters(adapter_list)

        zero_lora_params(model)
        apply_adapter_weights(model, merged)
        mx.eval(model.parameters())

        ppls = {}
        for dname, data_dir in data_dirs.items():
            ppl = compute_ppl(model, tokenizer, data_dir)
            ppls[dname] = round(ppl, 2)

        avg_composed = sum(ppls.values()) / len(ppls)
        avg_base = sum(base_ppls.values()) / len(base_ppls)
        ratio = avg_composed / avg_base

        composition_trajectory[step] = {
            "ppls": ppls,
            "avg_composed": round(avg_composed, 2),
            "composition_ratio_vs_base": round(ratio, 4),
        }
        print(f"  Step {step}: avg composed PPL={avg_composed:.2f} "
              f"(ratio vs base: {ratio:.4f})")

    results["composition_trajectory"] = {
        str(k): v for k, v in composition_trajectory.items()
    }

    # ------------------------------------------------------------------
    # Phase 6: Kill criteria assessment
    # ------------------------------------------------------------------
    print("\n[Phase 6] Kill criteria assessment...")

    # K1: mean |cos| at convergence exceeds 0.05
    final_step = steps_with_all_domains[-1] if steps_with_all_domains else None
    if final_step:
        final_mean_cos = cosine_trajectory[final_step]["mean_abs_cos"]
        k1_pass = final_mean_cos < 0.05
        results["k1_final_mean_cos"] = final_mean_cos
        results["k1_pass"] = k1_pass
        print(f"  K1: mean |cos| at step {final_step} = {final_mean_cos:.6f} "
              f"(threshold 0.05) -> {'PASS' if k1_pass else 'KILL'}")
    else:
        results["k1_pass"] = None
        print("  K1: no valid checkpoints")

    # K2: monotonic increase with no plateau
    # Check if trajectory shows a plateau (flattening) vs monotonic increase
    if len(steps_with_all_domains) >= 4:
        cos_values = [cosine_trajectory[s]["mean_abs_cos"] for s in steps_with_all_domains]

        # Check monotonicity: count how many consecutive increases
        increases = sum(1 for i in range(1, len(cos_values)) if cos_values[i] > cos_values[i-1])
        total_transitions = len(cos_values) - 1
        monotonic_fraction = increases / total_transitions

        # Check plateau: is the second half stable (CV < 30%)?
        half = len(cos_values) // 2
        second_half = cos_values[half:]
        if len(second_half) > 1:
            sh_mean = sum(second_half) / len(second_half)
            sh_std = (sum((x - sh_mean) ** 2 for x in second_half) / len(second_half)) ** 0.5
            sh_cv = sh_std / sh_mean if sh_mean > 1e-8 else 0
        else:
            sh_cv = 0

        # K2 KILL: monotonic increase (>80%) AND no plateau (CV > 30% = still changing)
        k2_kill = monotonic_fraction > 0.8 and sh_cv > 0.3
        results["k2_monotonic_fraction"] = round(monotonic_fraction, 3)
        results["k2_second_half_cv"] = round(sh_cv, 3)
        results["k2_pass"] = not k2_kill
        print(f"  K2: monotonic fraction={monotonic_fraction:.3f} "
              f"(>0.8 = monotonic), second-half CV={sh_cv:.3f} "
              f"(>0.3 = no plateau) -> {'KILL' if k2_kill else 'PASS'}")
    else:
        results["k2_pass"] = None
        print("  K2: insufficient checkpoints")

    # ------------------------------------------------------------------
    # Phase 7: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COSINE CONVERGENCE TRAJECTORY SUMMARY")
    print("=" * 70)

    print("\n  Cosine trajectory:")
    for step in steps_with_all_domains:
        ct = cosine_trajectory[step]
        print(f"    Step {step:5d}: mean |cos|={ct['mean_abs_cos']:.6f}, "
              f"max={ct['max_abs_cos']:.6f}, min={ct['min_abs_cos']:.6f}")

    print("\n  Composition PPL trajectory:")
    for step in sorted(composition_trajectory.keys()):
        ct = composition_trajectory[step]
        print(f"    Step {step:5d}: avg PPL={ct['avg_composed']:.2f} "
              f"(ratio vs base: {ct['composition_ratio_vs_base']:.4f})")

    print("\n  Convergence status:")
    for dname, conv in domain_convergence.items():
        print(f"    {dname}: {'CONVERGED' if conv['converged'] else 'not converged'} "
              f"(final loss={conv['final_loss']:.4f})")

    # Overall verdict
    k1 = results.get("k1_pass")
    k2 = results.get("k2_pass")
    if k1 is False or k2 is False:
        verdict = "KILLED"
    elif k1 is True and k2 is True:
        verdict = "SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    results["verdict"] = verdict
    total_time = time.time() - t_experiment_start
    results["total_time_minutes"] = round(total_time / 60, 1)

    print(f"\n  K1 (|cos| < 0.05 at convergence): {k1}")
    print(f"  K2 (not monotonically increasing): {k2}")
    print(f"  VERDICT: {verdict}")
    print(f"  Total time: {total_time / 60:.1f} minutes")

    # Free checkpoint memory
    del checkpoint_params

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
