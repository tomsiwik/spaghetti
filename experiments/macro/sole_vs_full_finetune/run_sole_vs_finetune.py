#!/usr/bin/env python3
"""SOLE (50 composed experts) vs single LoRA trained on union of all 50 datasets.

THE fundamental value proposition test. If SOLE wins, modular composition has
genuine quality advantages over monolithic training. If it loses, the value
is purely in updatability/cost, not quality.

Approach:
1. Load base Qwen2.5-7B + all 50 pilot adapters → pre-merge into composed model
2. Concatenate all 50 training datasets → train a single union-LoRA (same rank, same total steps)
3. Evaluate both on per-domain PPL and aggregate metrics
4. Compare: does SOLE (sum of specialists) beat monolithic (one generalist)?

Kill criteria:
- K1: Union LoRA beats SOLE on >70% of domains (modularity hurts quality)
- K2: Union LoRA aggregate PPL >10% better than SOLE (specialization is wasteful)
- K3: Union LoRA training cost >5x SOLE training cost (monolithic is impractical)

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

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "sole_vs_full_finetune"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# Training params — match pilot50 settings
LORA_RANK = 16
LORA_ALPHA = 16
MAX_SEQ_LEN = 512 if not IS_SMOKE else 256
UNION_TRAIN_STEPS = 500 if not IS_SMOKE else 10  # 50 experts * ~10 steps each = comparable total compute
UNION_LR = 1e-4
UNION_BATCH_SIZE = 4
EVAL_SAMPLES = 50 if not IS_SMOKE else 5

# Sample of domains for evaluation (representative cross-section)
EVAL_DOMAINS = [
    "python", "math", "medical", "bash", "physics",
    "creative-fiction", "legal", "rust", "statistics", "journalism",
] if not IS_SMOKE else ["python", "math"]


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    """Find all valid adapters in the adapter directory."""
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def load_eval_texts(domain, tokenizer, n=50):
    """Load evaluation texts for a domain (last portion to avoid train contamination)."""
    texts = []
    for fname in ["eval.jsonl", "train.jsonl"]:
        f = DATA_DIR / domain / fname
        if not f.exists():
            continue
        with open(f) as fh:
            lines = fh.readlines()
        if fname == "train.jsonl":
            lines = lines[-min(200, len(lines)):]
        for line in lines:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
            elif "text" in record:
                text = record["text"]
            else:
                continue
            texts.append(text)
    random.shuffle(texts)
    return texts[:n]


def load_union_training_data(tokenizer, domains, max_per_domain=100):
    """Load and concatenate training data from all domains."""
    all_texts = []
    for domain in domains:
        texts = []
        f = DATA_DIR / domain / "train.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            lines = fh.readlines()
        # Use first portion for training
        lines = lines[:min(max_per_domain, len(lines))]
        for line in lines:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
            elif "text" in record:
                text = record["text"]
            else:
                continue
            texts.append(text)
        all_texts.extend(texts)
        log(f"  Loaded {len(texts)} examples from {domain}")
    random.shuffle(all_texts)
    return all_texts


def compose_adapters_on_cpu(adapter_names, adapter_dir, weights=None):
    """Compose multiple LoRA adapters on CPU by averaging safetensors weights.

    Returns path to a temporary directory containing the composed adapter.
    Caller is responsible for cleanup (shutil.rmtree).
    """
    from safetensors.torch import load_file, save_file
    import torch as _torch

    if weights is None:
        weights = [1.0 / len(adapter_names)] * len(adapter_names)

    composed = {}
    adapter_config = None

    for i, name in enumerate(adapter_names):
        adapter_path = adapter_dir / name
        st_path = adapter_path / "adapter_model.safetensors"
        if not st_path.exists():
            continue
        tensors = load_file(str(st_path), device="cpu")
        w = weights[i]
        for key, val in tensors.items():
            if key in composed:
                composed[key] = composed[key] + val.float() * w
            else:
                composed[key] = val.float() * w

        # Grab adapter_config from first adapter
        if adapter_config is None:
            cfg_path = adapter_path / "adapter_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    adapter_config = json.load(f)

    # Cast back to bf16
    for key in composed:
        composed[key] = composed[key].to(_torch.bfloat16)

    # Save to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="composed_adapter_")
    save_file(composed, os.path.join(tmp_dir, "adapter_model.safetensors"))
    if adapter_config:
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

    return tmp_dir


def compute_ppl(model, tokenizer, texts, max_len=512):
    """Compute perplexity on a list of texts."""
    import torch
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_len).to(model.device)
            if inputs.input_ids.shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs.input_ids)
            n_tokens = inputs.input_ids.shape[1] - 1
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def run_experiment():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model

    log("=" * 70)
    log("SOLE vs Full Fine-tune Union Experiment")
    log("=" * 70)

    # Discover adapters
    adapters = discover_adapters()
    log(f"Found {len(adapters)} adapters")
    if IS_SMOKE:
        adapters = adapters[:5]

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config (fits 7B model in ~4GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    # ============================================================
    # PHASE 1: Evaluate BASE model (no adapters) — do this first
    # since we reuse the base model for SOLE composition
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 1: Base model evaluation")
    log("=" * 70)

    log("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, trust_remote_code=True,
    )

    base_results = {}
    for domain in EVAL_DOMAINS:
        texts = load_eval_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            continue
        ppl = compute_ppl(model, tokenizer, texts, MAX_SEQ_LEN)
        base_results[domain] = ppl
        log(f"  Base PPL on {domain}: {ppl:.2f}")

    # ============================================================
    # PHASE 2: Evaluate SOLE (composed 50-expert model)
    # Uses add_weighted_adapter to compose without dequantizing base
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 2: SOLE composed model evaluation")
    log("=" * 70)

    log(f"Composing {len(adapters)} adapters on CPU (avoids GPU OOM)...")
    merge_start = time.time()

    composed_dir = compose_adapters_on_cpu(adapters, ADAPTER_DIR)
    n = len(adapters)
    peft_model = PeftModel.from_pretrained(model, composed_dir, adapter_name="composed")
    shutil.rmtree(composed_dir)

    merge_time = time.time() - merge_start
    log(f"Composed {n} adapters in {merge_time:.1f}s")

    # Evaluate SOLE on each domain
    sole_results = {}
    peft_model.eval()
    for domain in EVAL_DOMAINS:
        texts = load_eval_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            log(f"  SKIP {domain}: not enough eval data ({len(texts)} texts)")
            continue
        ppl = compute_ppl(peft_model, tokenizer, texts, MAX_SEQ_LEN)
        sole_results[domain] = ppl
        log(f"  SOLE PPL on {domain}: {ppl:.2f}")

    n_merged = n  # for results output

    # Free SOLE model
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE 3: Train union LoRA on concatenated data (QLoRA)
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 3: Train union LoRA on all 50 domains")
    log("=" * 70)

    # Get all domains from adapters
    all_domains = [a for a in adapters if (DATA_DIR / a).exists()]
    max_per_domain = 20 if IS_SMOKE else 100  # ~5000 total examples
    union_data = load_union_training_data(tokenizer, all_domains, max_per_domain)
    log(f"Union dataset: {len(union_data)} examples from {len(all_domains)} domains")

    # Reload base model for training (need fresh model without composed adapters)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, trust_remote_code=True,
    )

    # Apply LoRA config (same as pilot50)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable params: {trainable_params:,}")

    # Simple training loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=UNION_LR)
    model.train()

    train_start = time.time()
    step = 0
    losses = []
    while step < UNION_TRAIN_STEPS:
        random.shuffle(union_data)
        for text in union_data:
            if step >= UNION_TRAIN_STEPS:
                break
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN, padding=False).to(model.device)
            if inputs.input_ids.shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                log(f"  Step {step}/{UNION_TRAIN_STEPS}, loss: {avg_loss:.4f}")
    train_time = time.time() - train_start
    log(f"Union training: {step} steps in {train_time:.1f}s")

    # Evaluate union model on each domain (keep as PeftModel — can't merge into 4-bit base)
    union_results = {}
    model.eval()
    for domain in EVAL_DOMAINS:
        texts = load_eval_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            continue
        ppl = compute_ppl(model, tokenizer, texts, MAX_SEQ_LEN)
        union_results[domain] = ppl
        log(f"  Union PPL on {domain}: {ppl:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # Compare SOLE vs Union per domain
    common_domains = sorted(set(sole_results) & set(union_results) & set(base_results))
    sole_wins = 0
    union_wins = 0
    domain_comparisons = []

    for domain in common_domains:
        sole_ppl = sole_results[domain]
        union_ppl = union_results[domain]
        base_ppl = base_results[domain]
        sole_improvement = (base_ppl - sole_ppl) / base_ppl * 100
        union_improvement = (base_ppl - union_ppl) / base_ppl * 100
        winner = "SOLE" if sole_ppl < union_ppl else "Union"
        if sole_ppl < union_ppl:
            sole_wins += 1
        else:
            union_wins += 1
        comparison = {
            "domain": domain,
            "base_ppl": base_ppl,
            "sole_ppl": sole_ppl,
            "union_ppl": union_ppl,
            "sole_improvement_pct": sole_improvement,
            "union_improvement_pct": union_improvement,
            "winner": winner,
        }
        domain_comparisons.append(comparison)
        log(f"  {domain}: SOLE={sole_ppl:.2f} Union={union_ppl:.2f} Base={base_ppl:.2f} Winner={winner}")

    # Aggregate metrics
    avg_sole_ppl = sum(sole_results[d] for d in common_domains) / len(common_domains)
    avg_union_ppl = sum(union_results[d] for d in common_domains) / len(common_domains)
    avg_base_ppl = sum(base_results[d] for d in common_domains) / len(common_domains)
    union_vs_sole_pct = (avg_sole_ppl - avg_union_ppl) / avg_sole_ppl * 100

    log(f"\nAggregate PPL: SOLE={avg_sole_ppl:.2f}, Union={avg_union_ppl:.2f}, Base={avg_base_ppl:.2f}")
    log(f"Union vs SOLE: {union_vs_sole_pct:+.1f}% ({'Union better' if union_vs_sole_pct > 0 else 'SOLE better'})")
    log(f"Domain wins: SOLE={sole_wins}, Union={union_wins} / {len(common_domains)}")

    # Kill criteria
    union_win_rate = union_wins / len(common_domains) if common_domains else 0
    k1_result = "KILLED" if union_win_rate > 0.70 else "SURVIVES"
    k2_result = "KILLED" if union_vs_sole_pct > 10 else "SURVIVES"
    # K3: training cost comparison
    sole_train_cost = len(adapters) * 15 / 60  # ~15 min per adapter (from pilot50 notes)
    union_train_cost = train_time / 60
    cost_ratio = sole_train_cost / union_train_cost if union_train_cost > 0 else float("inf")
    k3_result = "KILLED" if cost_ratio > 5 else "SURVIVES"

    log(f"\nKILL CRITERIA:")
    log(f"  K1 (Union beats SOLE on >70% of domains): {k1_result} ({union_win_rate:.0%})")
    log(f"  K2 (Union aggregate PPL >10% better): {k2_result} ({union_vs_sole_pct:+.1f}%)")
    log(f"  K3 (Union cost >5x SOLE): {k3_result} (ratio={cost_ratio:.1f}x)")

    overall = "KILLED" if "KILLED" in [k1_result, k2_result] else "SURVIVES"
    log(f"\n  OVERALL: {overall}")

    # Save results
    results = {
        "experiment": "sole_vs_full_finetune",
        "n_adapters": len(adapters),
        "n_merged": n_merged,
        "eval_domains": common_domains,
        "domain_comparisons": domain_comparisons,
        "aggregate": {
            "sole_avg_ppl": avg_sole_ppl,
            "union_avg_ppl": avg_union_ppl,
            "base_avg_ppl": avg_base_ppl,
            "union_vs_sole_pct": union_vs_sole_pct,
            "sole_wins": sole_wins,
            "union_wins": union_wins,
        },
        "kill_criteria": {
            "K1_union_win_rate": union_win_rate,
            "K1_threshold": 0.70,
            "K1_result": k1_result,
            "K2_union_vs_sole_pct": union_vs_sole_pct,
            "K2_threshold": 10.0,
            "K2_result": k2_result,
            "K3_cost_ratio": cost_ratio,
            "K3_threshold": 5.0,
            "K3_result": k3_result,
        },
        "training": {
            "union_steps": step,
            "union_train_time_s": train_time,
            "union_final_loss": losses[-1] if losses else None,
            "compose_time_s": merge_time,
            "sole_estimated_train_time_min": sole_train_cost,
            "union_train_time_min": union_train_cost,
        },
        "overall": overall,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_experiment()
