#!/usr/bin/env python3
"""Grassmannian AP Init at Macro Scale — test if frozen-A skeleton reduces interference.

Micro proven: Alternating Projection (AP) on Grassmannian Gr(r,d) constructs a skeleton
of frozen A matrices that are 1.3-2.0x more orthogonal than random init. With frozen A,
B-matrix training correlation becomes irrelevant (decorrelation filter at 0.14x baseline).

This experiment validates at production scale (Qwen2.5-7B, d=896, rank=16):
1. Compute AP skeleton for 50 expert slots on Gr(16, 896)
2. Retrain 5 diverse-domain adapters using AP-initialized frozen A
3. Compare pairwise |cos| of resulting adapters vs standard random-init adapters
4. Evaluate composition quality (PPL) of AP-init vs random-init adapters

Kill criteria:
- K1: AP-init adapters show mean |cos| >0.7x random-init (AP provides <30% reduction)
- K2: AP-init individual adapter quality (PPL) >5% worse than random-init
- K3: AP composition at N=5 shows PPL >3% worse than random-init composition

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "grassmannian_init_macro"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

RANK = 16
N_SKELETON_SLOTS = 50
TRAIN_DOMAINS = ["python", "math", "medical", "legal", "creative-fiction"] if not IS_SMOKE else ["python", "math"]
TRAIN_STEPS = 200 if not IS_SMOKE else 20
LR = 1e-4
MAX_SEQ_LEN = 512 if not IS_SMOKE else 128
EVAL_SAMPLES = 30 if not IS_SMOKE else 3
AP_ITERATIONS = 100 if not IS_SMOKE else 10
SEED = 42


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def alternating_projection_grassmannian(d, r, N, n_iters=100):
    """Construct N orthogonal subspaces on Gr(r, d) via Alternating Projection.

    Returns list of N matrices of shape (r, d) — the frozen A matrices.
    """
    log(f"AP skeleton: d={d}, r={r}, N={N}, iters={n_iters}")
    np.random.seed(SEED)

    # Initialize N random subspaces
    bases = []
    for _ in range(N):
        M = np.random.randn(r, d)
        Q, _ = np.linalg.qr(M.T)
        bases.append(Q[:, :r].T)  # (r, d)

    # Alternating projection to minimize pairwise coherence
    for iteration in range(n_iters):
        max_cos = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Compute principal angle cosines
                S = bases[i] @ bases[j].T  # (r, r)
                u, s, vt = np.linalg.svd(S, full_matrices=False)
                cos_val = s[0]  # largest principal angle cosine
                max_cos = max(max_cos, cos_val)

                # Project away overlap
                if cos_val > 1e-6:
                    # Reduce coherence by adjusting both subspaces
                    overlap = u[:, 0:1] @ vt[0:1, :]  # rank-1 overlap
                    perturbation = 0.1 * cos_val
                    Ai = bases[i] - perturbation * (overlap @ bases[j])
                    Aj = bases[j] - perturbation * (overlap.T @ bases[i])

                    # Re-orthogonalize
                    Qi, _ = np.linalg.qr(Ai.T)
                    bases[i] = Qi[:, :r].T
                    Qj, _ = np.linalg.qr(Aj.T)
                    bases[j] = Qj[:, :r].T

        if iteration % 20 == 0:
            log(f"  AP iter {iteration}: max coherence = {max_cos:.6f}")

    # Measure final coherence
    cos_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            S = bases[i] @ bases[j].T
            _, s, _ = np.linalg.svd(S, full_matrices=False)
            cos_matrix[i, j] = cos_matrix[j, i] = s[0]

    mean_cos = cos_matrix[np.triu_indices(N, k=1)].mean()
    max_cos = cos_matrix[np.triu_indices(N, k=1)].max()
    log(f"  AP done: mean coherence={mean_cos:.6f}, max={max_cos:.6f}")

    return bases, {"mean_coherence": float(mean_cos), "max_coherence": float(max_cos)}


def get_hidden_dim():
    """Get hidden dimension from model config without loading full model."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    return config.hidden_size


def load_training_data(domain, max_examples=500):
    """Load training data for a domain."""
    domain_dir = DATA_DIR / domain
    if not domain_dir.exists():
        log(f"WARN: no data for {domain}")
        return []
    texts = []
    for f in sorted(domain_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", obj.get("output", "")))
                    if text and len(text) > 50:
                        texts.append(text)
                except Exception:
                    continue
        if len(texts) >= max_examples:
            break
    return texts[:max_examples]


def train_adapter_with_init(model_name, domain, train_texts, init_A=None,
                            freeze_A=False, output_dir=None, steps=200, lr=1e-4):
    """Train a LoRA adapter, optionally with custom A-matrix initialization."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    log(f"  Training {domain} adapter (freeze_A={freeze_A}, steps={steps})...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Custom A-matrix initialization
    if init_A is not None:
        log(f"  Injecting AP skeleton A-matrices...")
        a_idx = 0
        for name, param in model.named_parameters():
            if "lora_A" in name and "weight" in name:
                if a_idx < len(init_A):
                    A_np = init_A[a_idx]  # (r, d)
                    # Adjust if dimensions don't match
                    if param.shape == (RANK, A_np.shape[1]):
                        param.data = torch.from_numpy(A_np).to(param.dtype).to(param.device)
                        if freeze_A:
                            param.requires_grad = False
                    a_idx += 1

    model.train()
    model.print_trainable_parameters()

    # Prepare training data
    encodings = tokenizer(train_texts, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN, padding=True)
    dataset_size = encodings["input_ids"].shape[0]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    losses = []
    for step in range(steps):
        idx = step % dataset_size
        input_ids = encodings["input_ids"][idx:idx+1].to(model.device)
        attn_mask = encodings["attention_mask"][idx:idx+1].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if step % 50 == 0:
            log(f"    step {step}: loss={loss.item():.4f}")

    # Save adapter
    if output_dir:
        model.save_pretrained(output_dir)
        log(f"  Saved to {output_dir}")

    final_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return float(final_loss)


def measure_pairwise_cosines(adapter_dirs):
    """Measure pairwise |cos| between adapters using flattened delta vectors."""
    from safetensors.torch import load_file

    vectors = {}
    for name, path in adapter_dirs.items():
        sf_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(sf_path):
            log(f"WARN: no safetensors at {sf_path}")
            continue
        tensors = load_file(sf_path, device="cpu")
        flat = np.concatenate([t.float().numpy().flatten() for t in tensors.values()])
        vectors[name] = flat
        del tensors

    pairs = []
    names = list(vectors.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = vectors[names[i]], vectors[names[j]]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = abs(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
            pairs.append({
                "adapter_a": names[i],
                "adapter_b": names[j],
                "abs_cosine": float(cos),
            })

    return pairs


def evaluate_ppl(adapter_dir, eval_texts, base_model_name=BASE_MODEL):
    """Evaluate PPL of an adapter on eval texts."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
        del inputs, outputs
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def main():
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log("=" * 60)
    log("GRASSMANNIAN AP INIT — MACRO VALIDATION")
    log("=" * 60)

    # Get hidden dimension
    d = get_hidden_dim()
    log(f"Hidden dim: d={d}, rank={RANK}")

    # Step 1: Compute AP skeleton
    log("\nPhase 1: Computing AP skeleton...")
    # We need skeletons per target module. For simplicity, compute one
    # skeleton and replicate (AP per-layer would be better but expensive)
    skeletons, ap_stats = alternating_projection_grassmannian(d, RANK, N_SKELETON_SLOTS, AP_ITERATIONS)
    log(f"AP skeleton: {len(skeletons)} slots, mean coherence={ap_stats['mean_coherence']:.6f}")

    # Also compute random baseline coherence
    random_bases = [np.random.randn(RANK, d) for _ in range(N_SKELETON_SLOTS)]
    for i in range(len(random_bases)):
        Q, _ = np.linalg.qr(random_bases[i].T)
        random_bases[i] = Q[:, :RANK].T
    random_cos = []
    for i in range(min(10, len(random_bases))):
        for j in range(i + 1, min(10, len(random_bases))):
            S = random_bases[i] @ random_bases[j].T
            _, s, _ = np.linalg.svd(S, full_matrices=False)
            random_cos.append(s[0])
    random_mean_cos = np.mean(random_cos) if random_cos else 0.0
    log(f"Random baseline mean coherence: {random_mean_cos:.6f}")

    results = {
        "config": {
            "hidden_dim": d,
            "rank": RANK,
            "n_skeleton_slots": N_SKELETON_SLOTS,
            "train_domains": TRAIN_DOMAINS,
            "train_steps": TRAIN_STEPS,
            "ap_iterations": AP_ITERATIONS,
        },
        "ap_skeleton_stats": ap_stats,
        "random_baseline_coherence": float(random_mean_cos),
    }

    # Step 2: Train adapters with AP init (frozen A) and random init
    log("\nPhase 2: Training adapters...")
    ap_adapter_dirs = {}
    random_adapter_dirs = {}
    ap_losses = {}
    random_losses = {}

    for i, domain in enumerate(TRAIN_DOMAINS):
        train_texts = load_training_data(domain, max_examples=200 if not IS_SMOKE else 20)
        if not train_texts:
            log(f"SKIP {domain}: no training data")
            continue
        eval_texts = load_training_data(domain, max_examples=EVAL_SAMPLES)

        # AP-initialized (frozen A)
        ap_dir = tempfile.mkdtemp(prefix=f"ap_{domain}_")
        ap_loss = train_adapter_with_init(
            BASE_MODEL, domain, train_texts,
            init_A=[skeletons[i]] * 100,  # replicate for all layers
            freeze_A=True, output_dir=ap_dir,
            steps=TRAIN_STEPS, lr=LR
        )
        ap_adapter_dirs[f"ap_{domain}"] = ap_dir
        ap_losses[domain] = ap_loss

        # Random-initialized (standard LoRA)
        rand_dir = tempfile.mkdtemp(prefix=f"rand_{domain}_")
        rand_loss = train_adapter_with_init(
            BASE_MODEL, domain, train_texts,
            init_A=None, freeze_A=False, output_dir=rand_dir,
            steps=TRAIN_STEPS, lr=LR
        )
        random_adapter_dirs[f"rand_{domain}"] = rand_dir
        random_losses[domain] = rand_loss

        log(f"  {domain}: AP loss={ap_loss:.4f}, random loss={rand_loss:.4f}")

    # Step 3: Measure pairwise cosines
    log("\nPhase 3: Measuring pairwise cosines...")
    ap_cosines = measure_pairwise_cosines(ap_adapter_dirs)
    random_cosines = measure_pairwise_cosines(random_adapter_dirs)

    ap_mean_cos = np.mean([p["abs_cosine"] for p in ap_cosines]) if ap_cosines else 0.0
    rand_mean_cos = np.mean([p["abs_cosine"] for p in random_cosines]) if random_cosines else 0.0
    cos_ratio = ap_mean_cos / rand_mean_cos if rand_mean_cos > 0 else float("inf")

    log(f"AP mean |cos|: {ap_mean_cos:.6f}")
    log(f"Random mean |cos|: {rand_mean_cos:.6f}")
    log(f"Ratio (AP/random): {cos_ratio:.4f}")

    results["cosine_comparison"] = {
        "ap_mean_cos": float(ap_mean_cos),
        "random_mean_cos": float(rand_mean_cos),
        "ratio_ap_over_random": float(cos_ratio),
        "ap_pairs": ap_cosines,
        "random_pairs": random_cosines,
    }

    # Step 4: Evaluate individual adapter PPL
    log("\nPhase 4: Evaluating individual PPL...")
    ap_ppls = {}
    random_ppls = {}
    for domain in TRAIN_DOMAINS:
        eval_texts = load_training_data(domain, max_examples=EVAL_SAMPLES)
        if not eval_texts:
            continue

        ap_key = f"ap_{domain}"
        rand_key = f"rand_{domain}"
        if ap_key in ap_adapter_dirs:
            ap_ppls[domain] = evaluate_ppl(ap_adapter_dirs[ap_key], eval_texts)
        if rand_key in random_adapter_dirs:
            random_ppls[domain] = evaluate_ppl(random_adapter_dirs[rand_key], eval_texts)

        log(f"  {domain}: AP PPL={ap_ppls.get(domain, 'N/A'):.2f}, "
            f"Random PPL={random_ppls.get(domain, 'N/A'):.2f}")

    # PPL quality comparison
    ppl_ratios = []
    for domain in TRAIN_DOMAINS:
        if domain in ap_ppls and domain in random_ppls and random_ppls[domain] > 0:
            ratio = (ap_ppls[domain] - random_ppls[domain]) / random_ppls[domain] * 100
            ppl_ratios.append(ratio)

    mean_ppl_change = np.mean(ppl_ratios) if ppl_ratios else 0.0

    results["individual_ppl"] = {
        "ap_ppls": {k: float(v) for k, v in ap_ppls.items()},
        "random_ppls": {k: float(v) for k, v in random_ppls.items()},
        "mean_ppl_change_pct": float(mean_ppl_change),
    }

    # Step 5: Evaluate composition PPL
    log("\nPhase 5: Evaluating composition PPL...")
    # Compose AP adapters
    ap_adapter_names_for_compose = list(ap_adapter_dirs.keys())
    random_adapter_names_for_compose = list(random_adapter_dirs.keys())

    # Simple CPU composition
    from safetensors.torch import load_file, save_file

    def compose_dirs(adapter_dirs_dict):
        composed = {}
        for name, path in adapter_dirs_dict.items():
            sf_path = os.path.join(path, "adapter_model.safetensors")
            if not os.path.exists(sf_path):
                continue
            tensors = load_file(sf_path, device="cpu")
            for key, val in tensors.items():
                if key in composed:
                    composed[key] = composed[key] + val.float()
                else:
                    composed[key] = val.float()
            del tensors
        tmpdir = tempfile.mkdtemp(prefix="composed_")
        save_file({k: v.to(torch.bfloat16) for k, v in composed.items()},
                  os.path.join(tmpdir, "adapter_model.safetensors"))
        # Copy config from first adapter
        first_dir = list(adapter_dirs_dict.values())[0]
        cfg = os.path.join(first_dir, "adapter_config.json")
        if os.path.exists(cfg):
            shutil.copy(cfg, os.path.join(tmpdir, "adapter_config.json"))
        return tmpdir

    ap_composed_dir = compose_dirs(ap_adapter_dirs)
    rand_composed_dir = compose_dirs(random_adapter_dirs)

    # Eval composition on all domains
    comp_ap_ppls = {}
    comp_rand_ppls = {}
    for domain in TRAIN_DOMAINS:
        eval_texts = load_training_data(domain, max_examples=EVAL_SAMPLES)
        if not eval_texts:
            continue
        comp_ap_ppls[domain] = evaluate_ppl(ap_composed_dir, eval_texts)
        comp_rand_ppls[domain] = evaluate_ppl(rand_composed_dir, eval_texts)
        log(f"  Composed {domain}: AP={comp_ap_ppls[domain]:.2f}, Random={comp_rand_ppls[domain]:.2f}")

    comp_ppl_ratios = []
    for domain in TRAIN_DOMAINS:
        if domain in comp_ap_ppls and domain in comp_rand_ppls and comp_rand_ppls[domain] > 0:
            ratio = (comp_ap_ppls[domain] - comp_rand_ppls[domain]) / comp_rand_ppls[domain] * 100
            comp_ppl_ratios.append(ratio)

    mean_comp_change = np.mean(comp_ppl_ratios) if comp_ppl_ratios else 0.0

    results["composition_ppl"] = {
        "ap_composed_ppls": {k: float(v) for k, v in comp_ap_ppls.items()},
        "random_composed_ppls": {k: float(v) for k, v in comp_rand_ppls.items()},
        "mean_composition_ppl_change_pct": float(mean_comp_change),
    }

    # Cleanup
    for d_path in list(ap_adapter_dirs.values()) + list(random_adapter_dirs.values()):
        shutil.rmtree(d_path, ignore_errors=True)
    shutil.rmtree(ap_composed_dir, ignore_errors=True)
    shutil.rmtree(rand_composed_dir, ignore_errors=True)

    # Kill criteria assessment
    k1_pass = cos_ratio <= 0.7  # AP reduces cosine by >30%
    k2_pass = abs(mean_ppl_change) <= 5.0  # AP quality within 5%
    k3_pass = abs(mean_comp_change) <= 3.0  # composition quality within 3%

    results["kill_criteria"] = {
        "K1_cos_ratio_ap_over_random": float(cos_ratio),
        "K1_threshold": 0.7,
        "K1_pass": k1_pass,
        "K1_interpretation": "AP reduces interference by >30%" if k1_pass else "AP reduction <30%",
        "K2_individual_ppl_change_pct": float(mean_ppl_change),
        "K2_threshold_pct": 5.0,
        "K2_pass": k2_pass,
        "K3_composition_ppl_change_pct": float(mean_comp_change),
        "K3_threshold_pct": 3.0,
        "K3_pass": k3_pass,
        "overall_pass": k1_pass and k2_pass and k3_pass,
    }

    log(f"\n{'='*60}")
    log("KILL CRITERIA ASSESSMENT")
    log(f"{'='*60}")
    log(f"K1: cos ratio AP/random = {cos_ratio:.4f} (threshold <0.7): {'PASS' if k1_pass else 'FAIL'}")
    log(f"K2: individual PPL change = {mean_ppl_change:+.2f}% (threshold <5%): {'PASS' if k2_pass else 'FAIL'}")
    log(f"K3: composition PPL change = {mean_comp_change:+.2f}% (threshold <3%): {'PASS' if k3_pass else 'FAIL'}")
    log(f"Overall: {'PASS' if k1_pass and k2_pass and k3_pass else 'FAIL'}")

    # Save
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
