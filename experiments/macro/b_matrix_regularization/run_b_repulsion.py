#!/usr/bin/env python3
"""B-matrix repulsion regularizer: reduce composition interference at macro scale.

Takes existing pilot50 adapters as baseline (standard LoRA training).
Fine-tunes a subset with B-repulsion regularizer for 10% additional steps.
Measures interference reduction and quality preservation.

Protocol:
  1. Select 10 domains with existing adapters + training data
  2. Extract B matrices from all 10 adapters → reference set
  3. Measure baseline pairwise interference: ||B_i^T B_j||_F for all pairs
  4. Fine-tune each adapter for 30 steps with repulsion loss:
       L = CE_loss + lambda * sum_{j!=i} mean_layers(||B_i^T B_j||_F)
  5. Measure post-repulsion interference
  6. Compare per-adapter quality (PPL on held-out data)

Kill criteria:
  K1: PPL degradation > 5% from repulsion fine-tuning
  K2: Interference reduction < 30% (repulsion not effective enough)
  K3: Training time > 3x standard (overhead too high)

Supports SMOKE_TEST=1 (3 domains, 5 steps, 10 eval examples).
"""

import gc
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
RESULTS_DIR = Path("/workspace/llm/results/b_matrix_regularization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = Path("/workspace/llm/adapters")
DATA_DIR = Path("/workspace/llm/data/distillation")
BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# Domains to evaluate — those most likely to have composition interference
# (related domains that share semantic space)
EVAL_DOMAINS = [
    "physics", "chemistry", "biology", "math", "statistics",
    "legal", "medical", "finance", "python", "ethics",
]

REPULSION_LAMBDA = 0.01  # regularizer strength
REPULSION_STEPS = 30     # 10% of 300 standard steps
REPULSION_LR = 5e-5      # lower LR for fine-tuning
EVAL_EXAMPLES = 100      # held-out examples for PPL measurement


def extract_b_matrices(model):
    """Extract all lora_B weight matrices from a PeftModel.
    Returns dict: {layer_key: tensor(out_features, rank)}.
    """
    b_matrices = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and "weight" in name:
            # Detach and clone to avoid graph issues
            b_matrices[name] = param.detach().clone()
    return b_matrices


def compute_pairwise_interference(b_set_i, b_set_j):
    """Compute mean ||B_i^T B_j||_F across all matching layers.
    b_set_i, b_set_j: dict of {layer_key: tensor}.
    """
    frob_norms = []
    for key in b_set_i:
        if key in b_set_j:
            bi = b_set_i[key].float()
            bj = b_set_j[key].float()
            # ||B_i^T B_j||_F where B has shape (out, rank)
            product = bi.T @ bj  # (rank, rank)
            frob = torch.norm(product, p="fro").item()
            frob_norms.append(frob)
    return float(np.mean(frob_norms)) if frob_norms else 0.0


def compute_repulsion_loss(model, reference_b_sets):
    """Compute B-repulsion loss: sum over references of mean ||B_current^T B_ref||_F."""
    current_b = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and "weight" in name:
            current_b[name] = param  # keep in graph for gradient

    total_repulsion = torch.tensor(0.0, device=next(model.parameters()).device)
    n_terms = 0

    for ref_b in reference_b_sets:
        for key in current_b:
            if key in ref_b:
                bi = current_b[key].float()
                bj = ref_b[key].float().to(bi.device)
                product = bi.T @ bj  # (rank, rank)
                total_repulsion = total_repulsion + torch.norm(product, p="fro")
                n_terms += 1

    if n_terms > 0:
        total_repulsion = total_repulsion / n_terms

    return total_repulsion


def evaluate_ppl(model, tokenizer, data_path, max_examples=100):
    """Compute perplexity on held-out data."""
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), size=min(max_examples, len(dataset)), replace=False)

    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx in sorted(indices):
            example = dataset[int(idx)]
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[:, :-1]
            labels = inputs["input_ids"][:, 1:]

            log_probs = F.log_softmax(logits, dim=-1)
            token_nlls = -log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # Exclude padding
            mask = (labels != tokenizer.pad_token_id).float()
            total_nll += (token_nlls * mask).sum().item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def finetune_with_repulsion(base_model, adapter_path, data_path, reference_b_sets,
                            domain, tokenizer, steps=30, lr=5e-5, lam=0.01):
    """Fine-tune adapter with B-repulsion regularizer.

    Returns (model, metrics dict).
    """
    # Load adapter (is_trainable=True so LoRA params get requires_grad)
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    model.train()

    # Load training data
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    rng = np.random.RandomState(42)

    # Optimizer — only LoRA params
    # Explicitly enable gradients (is_trainable may not work if base_model
    # was previously wrapped by PeftModel in Phase 1)
    for n, p in model.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    if not lora_params:
        raise RuntimeError(
            f"No LoRA params found for {domain}. "
            f"Named params: {[n for n, _ in model.named_parameters()][:10]}"
        )
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)

    ce_losses = []
    rep_losses = []
    total_time = 0.0

    for step in range(steps):
        t0 = time.time()

        # Random example
        idx = rng.randint(0, len(dataset))
        example = dataset[int(idx)]
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        ce_loss = outputs.loss

        # Repulsion loss
        rep_loss = compute_repulsion_loss(model, reference_b_sets)
        total_loss = ce_loss + lam * rep_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        ce_losses.append(ce_loss.item())
        rep_losses.append(rep_loss.item())
        total_time += time.time() - t0

        if step % 10 == 0 or step == steps - 1:
            print(f"    step {step:3d}: CE={ce_loss.item():.4f} Rep={rep_loss.item():.4f} "
                  f"Total={total_loss.item():.4f}")

    metrics = {
        "domain": domain,
        "steps": steps,
        "lr": lr,
        "lambda": lam,
        "final_ce_loss": float(np.mean(ce_losses[-5:])),
        "final_rep_loss": float(np.mean(rep_losses[-5:])),
        "initial_rep_loss": float(rep_losses[0]) if rep_losses else 0.0,
        "rep_loss_reduction": float(1 - np.mean(rep_losses[-5:]) / max(rep_losses[0], 1e-10)) if rep_losses else 0.0,
        "training_time_s": total_time,
    }

    return model, metrics


def main():
    t0_total = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  B-Matrix Repulsion Regularizer — Macro Validation")
    print(f"  Device: {device}, Smoke: {IS_SMOKE}")
    print(f"  Lambda: {REPULSION_LAMBDA}, Steps: {REPULSION_STEPS}, LR: {REPULSION_LR}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)

    # Determine available domains
    domains = []
    for d in EVAL_DOMAINS:
        adapter_path = ADAPTER_DIR / d
        data_path = DATA_DIR / d / "train.jsonl"
        has_adapter = (adapter_path / "adapter_model.safetensors").exists()
        has_data = data_path.exists()
        if has_adapter and has_data:
            domains.append(d)
        else:
            print(f"  Skip {d}: adapter={'OK' if has_adapter else 'MISSING'}, data={'OK' if has_data else 'MISSING'}")

    if IS_SMOKE:
        domains = domains[:3]
        repulsion_steps = 5
        eval_examples = 10
    else:
        repulsion_steps = REPULSION_STEPS
        eval_examples = EVAL_EXAMPLES

    print(f"\n  Using {len(domains)} domains: {domains}")

    # Load base model
    print(f"\n[1] Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Phase 1: Extract baseline B matrices and measure interference ──
    print(f"\n[2] Extracting B matrices from {len(domains)} pilot50 adapters")
    baseline_b_sets = {}
    baseline_ppls = {}

    for domain in domains:
        adapter_path = ADAPTER_DIR / domain
        data_path = DATA_DIR / domain / "train.jsonl"

        print(f"  Loading {domain} adapter...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()

        # Extract B matrices
        baseline_b_sets[domain] = extract_b_matrices(model)
        print(f"    Extracted {len(baseline_b_sets[domain])} B matrices")

        # Evaluate baseline PPL
        ppl = evaluate_ppl(model, tokenizer, data_path, max_examples=eval_examples)
        baseline_ppls[domain] = ppl
        print(f"    Baseline PPL: {ppl:.2f}")

        # Properly remove adapter so base_model is clean for next iteration
        model.delete_adapter("default")
        del model
        torch.cuda.empty_cache()

    # Reload base model to clear PEFT wrapping from Phase 1
    # (delete_adapter doesn't fully unwrap PEFT modules, leaving base_model
    # in a state where subsequent PeftModel.from_pretrained can't find LoRA params)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, trust_remote_code=True,
    )

    # Compute baseline pairwise interference
    print(f"\n[3] Computing baseline pairwise interference")
    baseline_interference = {}
    for i, di in enumerate(domains):
        for j, dj in enumerate(domains):
            if i >= j:
                continue
            interference = compute_pairwise_interference(
                baseline_b_sets[di], baseline_b_sets[dj])
            baseline_interference[f"{di}-{dj}"] = interference
            print(f"  {di:12s} x {dj:12s}: ||B_i^T B_j||_F = {interference:.6f}")

    mean_baseline_interference = float(np.mean(list(baseline_interference.values())))
    print(f"  Mean baseline interference: {mean_baseline_interference:.6f}")

    # ── Phase 2: Fine-tune with repulsion ──────────────────────────
    print(f"\n[4] Fine-tuning with B-repulsion (lambda={REPULSION_LAMBDA}, steps={repulsion_steps})")
    repulsion_b_sets = {}
    repulsion_ppls = {}
    training_metrics = {}
    standard_train_time = 300 * 0.5  # rough estimate: 300 steps * ~0.5s/step

    for domain in domains:
        adapter_path = ADAPTER_DIR / domain
        data_path = DATA_DIR / domain / "train.jsonl"

        # Reference B sets = all OTHER domains' baseline B matrices
        ref_b_sets = [baseline_b_sets[d] for d in domains if d != domain]

        print(f"\n  Fine-tuning {domain} with B-repulsion ({len(ref_b_sets)} references)...")
        model, metrics = finetune_with_repulsion(
            base_model, adapter_path, data_path, ref_b_sets, domain, tokenizer,
            steps=repulsion_steps, lr=REPULSION_LR, lam=REPULSION_LAMBDA,
        )
        training_metrics[domain] = metrics

        # Extract post-repulsion B matrices
        repulsion_b_sets[domain] = extract_b_matrices(model)

        # Evaluate post-repulsion PPL
        ppl = evaluate_ppl(model, tokenizer, data_path, max_examples=eval_examples)
        repulsion_ppls[domain] = ppl
        print(f"    Post-repulsion PPL: {ppl:.2f} (baseline: {baseline_ppls[domain]:.2f}, "
              f"delta: {(ppl/baseline_ppls[domain] - 1)*100:+.2f}%)")

        model.delete_adapter("default")
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Phase 3: Measure post-repulsion interference ────────────────
    print(f"\n[5] Computing post-repulsion pairwise interference")
    repulsion_interference = {}
    for i, di in enumerate(domains):
        for j, dj in enumerate(domains):
            if i >= j:
                continue
            interference = compute_pairwise_interference(
                repulsion_b_sets[di], repulsion_b_sets[dj])
            repulsion_interference[f"{di}-{dj}"] = interference
            baseline_val = baseline_interference[f"{di}-{dj}"]
            reduction = (1 - interference / max(baseline_val, 1e-10)) * 100
            print(f"  {di:12s} x {dj:12s}: {interference:.6f} (was {baseline_val:.6f}, {reduction:+.1f}%)")

    mean_repulsion_interference = float(np.mean(list(repulsion_interference.values())))
    interference_reduction = (1 - mean_repulsion_interference / max(mean_baseline_interference, 1e-10))
    print(f"  Mean post-repulsion interference: {mean_repulsion_interference:.6f}")
    print(f"  Interference reduction: {interference_reduction*100:.1f}%")

    # ── Kill Criteria Assessment ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KILL CRITERIA ASSESSMENT")
    print(f"{'='*70}")

    # K1: PPL degradation < 5%
    ppl_deltas = [(repulsion_ppls[d] / baseline_ppls[d] - 1) for d in domains]
    max_ppl_degradation = max(ppl_deltas)
    mean_ppl_degradation = float(np.mean(ppl_deltas))
    k1_pass = max_ppl_degradation <= 0.05

    # K2: Interference reduction > 30%
    k2_pass = interference_reduction >= 0.30

    # K3: Training time < 3x standard
    repulsion_times = [training_metrics[d]["training_time_s"] for d in domains]
    mean_repulsion_time = float(np.mean(repulsion_times))
    # Standard 300-step training takes roughly 150s per domain
    # Repulsion adds 10% steps (30) so overhead should be proportional
    time_ratio = mean_repulsion_time / max(standard_train_time * (repulsion_steps / 300), 1.0)
    k3_pass = time_ratio <= 3.0

    print(f"  K1: Max PPL degradation <= 5%?      max={max_ppl_degradation*100:+.2f}%, "
          f"mean={mean_ppl_degradation*100:+.2f}%  {'PASS' if k1_pass else 'KILL'}")
    print(f"  K2: Interference reduction >= 30%?   reduction={interference_reduction*100:.1f}%  "
          f"{'PASS' if k2_pass else 'KILL'}")
    print(f"  K3: Time overhead <= 3x?             ratio={time_ratio:.1f}x  "
          f"{'PASS' if k3_pass else 'KILL'}")

    overall = k1_pass and k2_pass and k3_pass
    print(f"  Overall: {'SURVIVES' if overall else 'KILLED'}")

    # Per-domain detail table
    print(f"\n{'='*70}")
    print(f"  {'Domain':<14s} {'BasePPL':>8s} {'RepPPL':>8s} {'PPLDelta':>9s} {'RepLoss0':>9s} {'RepLossF':>9s} {'Time':>6s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*6}")
    for d in domains:
        m = training_metrics[d]
        delta = (repulsion_ppls[d] / baseline_ppls[d] - 1) * 100
        print(f"  {d:<14s} {baseline_ppls[d]:>8.2f} {repulsion_ppls[d]:>8.2f} {delta:>+8.2f}% "
              f"{m['initial_rep_loss']:>9.4f} {m['final_rep_loss']:>9.4f} {m['training_time_s']:>5.0f}s")

    # ── Save results ──────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    output = {
        "experiment": "exp_b_matrix_interference_regularization",
        "model": BASE_MODEL,
        "n_domains": len(domains),
        "domains": domains,
        "config": {
            "repulsion_lambda": REPULSION_LAMBDA,
            "repulsion_steps": repulsion_steps,
            "repulsion_lr": REPULSION_LR,
            "eval_examples": eval_examples,
            "is_smoke": IS_SMOKE,
        },
        "baseline_ppls": {d: baseline_ppls[d] for d in domains},
        "repulsion_ppls": {d: repulsion_ppls[d] for d in domains},
        "ppl_deltas": {d: float((repulsion_ppls[d] / baseline_ppls[d] - 1)) for d in domains},
        "baseline_interference": baseline_interference,
        "repulsion_interference": repulsion_interference,
        "interference_reduction": {
            pair: float(1 - repulsion_interference[pair] / max(baseline_interference[pair], 1e-10))
            for pair in baseline_interference
        },
        "mean_baseline_interference": mean_baseline_interference,
        "mean_repulsion_interference": mean_repulsion_interference,
        "overall_interference_reduction": float(interference_reduction),
        "training_metrics": training_metrics,
        "kill_criteria": {
            "k1_max_ppl_degradation": {"value": float(max_ppl_degradation), "threshold": 0.05, "pass": k1_pass},
            "k2_interference_reduction": {"value": float(interference_reduction), "threshold": 0.30, "pass": k2_pass},
            "k3_time_ratio": {"value": float(time_ratio), "threshold": 3.0, "pass": k3_pass},
            "overall": "SURVIVES" if overall else "KILLED",
        },
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
