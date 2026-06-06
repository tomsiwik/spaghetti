#!/usr/bin/env python3
"""Test continual addition of new experts to existing composed model.

THE modularity test. SOLE promises "add without retraining." This experiment:
1. Compose the existing 50 pilot experts (baseline)
2. Train 10 NEW experts on domains NOT in pilot50
3. Add new experts one-by-one to composed model
4. Measure: existing expert quality preservation + new expert quality

Kill criteria:
- K1: Adding 10 new experts degrades existing domain PPL by >3% (interference)
- K2: New experts achieve <50% of the PPL improvement that pilot experts achieve (poor composition)
- K3: Training 10 new experts costs >$5 total (economics don't scale)

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
RESULTS_DIR = REPO_ROOT / "results" / "expert_continual_addition"
NEW_ADAPTER_DIR = RESULTS_DIR / "new_adapters"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NEW_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

LORA_RANK = 16
LORA_ALPHA = 16
MAX_SEQ_LEN = 512 if not IS_SMOKE else 256
TRAIN_STEPS_PER_EXPERT = 200 if not IS_SMOKE else 5
TRAIN_LR = 1e-4
EVAL_SAMPLES = 50 if not IS_SMOKE else 5

# Existing pilot domains to monitor for regression
MONITOR_DOMAINS = [
    "python", "math", "medical", "bash", "physics",
] if not IS_SMOKE else ["python", "math"]

# New domains to train (NOT in pilot50)
# These are domains present in data/distillation but checking which have enough data
NEW_DOMAINS_CANDIDATES = [
    "geology", "neuroscience", "ecology", "game-theory", "ethics",
    "grant-writing", "screenplay", "astronomy", "genetics", "debate",
]
N_NEW = 10 if not IS_SMOKE else 2


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    """Find all valid pilot adapters."""
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def load_texts(domain, tokenizer, n=50, split="eval"):
    """Load texts for a domain."""
    texts = []
    fnames = ["eval.jsonl", "train.jsonl"] if split == "eval" else ["train.jsonl"]
    for fname in fnames:
        f = DATA_DIR / domain / fname
        if not f.exists():
            continue
        with open(f) as fh:
            lines = fh.readlines()
        if split == "eval" and fname == "train.jsonl":
            lines = lines[-min(200, len(lines)):]
        elif split == "train":
            lines = lines[:min(n, len(lines))]
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
        if isinstance(name, Path):
            adapter_path = name
        else:
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

        if adapter_config is None:
            cfg_path = adapter_path / "adapter_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    adapter_config = json.load(f)

    for key in composed:
        composed[key] = composed[key].to(_torch.bfloat16)

    tmp_dir = tempfile.mkdtemp(prefix="composed_adapter_")
    save_file(composed, os.path.join(tmp_dir, "adapter_model.safetensors"))
    if adapter_config:
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

    return tmp_dir


def compute_ppl(model, tokenizer, texts, max_len=512):
    """Compute perplexity."""
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


def train_expert(model, tokenizer, domain, steps, lr):
    """Train a single LoRA expert on domain data."""
    import torch
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    texts = load_texts(domain, tokenizer, n=500, split="train")
    if len(texts) < 10:
        log(f"  WARNING: only {len(texts)} training texts for {domain}")
        return None

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
    peft_model.train()
    step = 0
    losses = []
    while step < steps:
        random.shuffle(texts)
        for text in texts:
            if step >= steps:
                break
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN).to(peft_model.device)
            if inputs.input_ids.shape[1] < 2:
                continue
            outputs = peft_model(**inputs, labels=inputs.input_ids)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(outputs.loss.item())
            step += 1

    # Save adapter
    save_path = NEW_ADAPTER_DIR / domain
    peft_model.save_pretrained(str(save_path))
    final_loss = losses[-1] if losses else None
    log(f"  Trained {domain}: {step} steps, final_loss={final_loss:.4f}")

    # Clean up (don't merge_and_unload — incompatible with 4-bit quantization)
    del peft_model
    return None, final_loss


def run_experiment():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    log("=" * 70)
    log("Expert Continual Addition Experiment")
    log("=" * 70)

    adapters = discover_adapters()
    log(f"Found {len(adapters)} pilot adapters")
    if IS_SMOKE:
        adapters = adapters[:5]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select new domains (not in pilot)
    pilot_set = set(adapters)
    new_domains = [d for d in NEW_DOMAINS_CANDIDATES
                   if d not in pilot_set and (DATA_DIR / d / "train.jsonl").exists()][:N_NEW]
    log(f"New domains to train: {new_domains}")

    # 4-bit quantization config (fits 7B model in ~4GB VRAM, leaves room for adapters)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    # ============================================================
    # PHASE 1: Evaluate base model
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 1: Base model PPL")
    log("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    base_ppl = {}
    all_eval_domains = MONITOR_DOMAINS + new_domains
    for domain in all_eval_domains:
        texts = load_texts(domain, tokenizer, EVAL_SAMPLES, split="eval")
        if len(texts) < 3:
            continue
        ppl = compute_ppl(model, tokenizer, texts, MAX_SEQ_LEN)
        base_ppl[domain] = ppl
        log(f"  Base PPL {domain}: {ppl:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE 2: Pre-merge pilot50 and evaluate
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 2: SOLE-50 composed model (before new experts)")
    log("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    # Compose adapters on CPU to avoid GPU OOM with 50 simultaneous adapters
    log(f"Composing {len(adapters)} pilot adapters on CPU...")
    composed_dir = compose_adapters_on_cpu(adapters, ADAPTER_DIR)
    n_merged = len(adapters)
    peft_model = PeftModel.from_pretrained(model, composed_dir, adapter_name="sole50")
    shutil.rmtree(composed_dir)
    log(f"Composed {n_merged} pilot adapters")

    sole50_ppl = {}
    peft_model.eval()
    for domain in all_eval_domains:
        texts = load_texts(domain, tokenizer, EVAL_SAMPLES, split="eval")
        if len(texts) < 3:
            continue
        ppl = compute_ppl(peft_model, tokenizer, texts, MAX_SEQ_LEN)
        sole50_ppl[domain] = ppl
        log(f"  SOLE-50 PPL {domain}: {ppl:.2f}")

    # ============================================================
    # PHASE 3: Train and add new experts one by one
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 3: Train and add 10 new experts")
    log("=" * 70)

    # We need fresh base for training each new expert
    del peft_model, model
    gc.collect()
    torch.cuda.empty_cache()

    new_expert_losses = {}
    train_times = {}
    for domain in new_domains:
        log(f"\nTraining new expert: {domain}")
        # Load fresh base for each expert (clean slate, 4-bit to fit in 24GB GPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, cache_dir=HF_CACHE, quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        t0 = time.time()
        result = train_expert(base_model, tokenizer, domain,
                              TRAIN_STEPS_PER_EXPERT, TRAIN_LR)
        train_times[domain] = time.time() - t0
        if result is not None:
            _, final_loss = result
            new_expert_losses[domain] = final_loss
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    log(f"\nTrained {len(new_expert_losses)} new experts")

    # ============================================================
    # PHASE 4: Compose SOLE-50 + new experts, evaluate
    # ============================================================
    log("\n" + "=" * 70)
    log("PHASE 4: SOLE-60 composed model (50 pilot + 10 new)")
    log("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    # Compose all pilot + new adapters on CPU to avoid GPU OOM
    # Collect all adapter paths (pilot from ADAPTER_DIR, new from NEW_ADAPTER_DIR)
    all_adapter_paths = []
    for name in adapters:
        p = ADAPTER_DIR / name
        if (p / "adapter_model.safetensors").exists():
            all_adapter_paths.append(p)

    n_new_merged = 0
    for domain in new_domains:
        p = NEW_ADAPTER_DIR / domain
        if (p / "adapter_model.safetensors").exists():
            all_adapter_paths.append(p)
            n_new_merged += 1

    n_total = len(all_adapter_paths)
    log(f"Composing {n_total} adapters on CPU ({n_new_merged} new + {n_merged} pilot)...")
    composed_dir = compose_adapters_on_cpu(all_adapter_paths, Path("."), weights=[1.0 / n_total] * n_total)
    peft_model = PeftModel.from_pretrained(model, composed_dir, adapter_name="sole60")
    shutil.rmtree(composed_dir)
    log(f"Composed {n_new_merged} new + {n_merged} pilot adapters = {n_total} total")

    sole60_ppl = {}
    peft_model.eval()
    for domain in all_eval_domains:
        texts = load_texts(domain, tokenizer, EVAL_SAMPLES, split="eval")
        if len(texts) < 3:
            continue
        ppl = compute_ppl(peft_model, tokenizer, texts, MAX_SEQ_LEN)
        sole60_ppl[domain] = ppl
        log(f"  SOLE-60 PPL {domain}: {ppl:.2f}")

    del peft_model, model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS
    # ============================================================
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # K1: Existing domain regression
    regressions = []
    for domain in MONITOR_DOMAINS:
        if domain in sole50_ppl and domain in sole60_ppl:
            ppl_before = sole50_ppl[domain]
            ppl_after = sole60_ppl[domain]
            regression_pct = (ppl_after - ppl_before) / ppl_before * 100
            regressions.append({"domain": domain, "before": ppl_before,
                                "after": ppl_after, "regression_pct": regression_pct})
            log(f"  Existing {domain}: {ppl_before:.2f} -> {ppl_after:.2f} ({regression_pct:+.1f}%)")

    max_regression = max(r["regression_pct"] for r in regressions) if regressions else 0
    avg_regression = sum(r["regression_pct"] for r in regressions) / len(regressions) if regressions else 0

    # K2: New expert quality
    new_improvements = []
    for domain in new_domains:
        if domain in base_ppl and domain in sole60_ppl:
            base = base_ppl[domain]
            sole60 = sole60_ppl[domain]
            improvement = (base - sole60) / base * 100
            new_improvements.append({"domain": domain, "base_ppl": base,
                                     "sole60_ppl": sole60, "improvement_pct": improvement})
            log(f"  New {domain}: base={base:.2f} sole60={sole60:.2f} ({improvement:+.1f}%)")

    # Compare to pilot average improvement
    pilot_improvements = []
    for domain in MONITOR_DOMAINS:
        if domain in base_ppl and domain in sole50_ppl:
            improvement = (base_ppl[domain] - sole50_ppl[domain]) / base_ppl[domain] * 100
            pilot_improvements.append(improvement)

    avg_pilot_improvement = sum(pilot_improvements) / len(pilot_improvements) if pilot_improvements else 1
    avg_new_improvement = sum(n["improvement_pct"] for n in new_improvements) / len(new_improvements) if new_improvements else 0
    quality_ratio = avg_new_improvement / avg_pilot_improvement if avg_pilot_improvement != 0 else 0

    # K3: Training cost (GPU time)
    total_train_time = sum(train_times.values())
    # A5000 ~$0.40/hr on RunPod
    total_cost = (total_train_time / 3600) * 0.40

    k1_result = "KILLED" if avg_regression > 3 else "SURVIVES"
    k2_result = "KILLED" if quality_ratio < 0.50 else "SURVIVES"
    k3_result = "KILLED" if total_cost > 5 else "SURVIVES"

    log(f"\nKILL CRITERIA:")
    log(f"  K1 (existing domain regression >3%): {k1_result} (avg={avg_regression:+.1f}%, max={max_regression:+.1f}%)")
    log(f"  K2 (new expert quality <50% of pilot): {k2_result} (ratio={quality_ratio:.2f})")
    log(f"  K3 (training cost >$5): {k3_result} (${total_cost:.2f})")

    overall = "KILLED" if "KILLED" in [k1_result, k2_result] else "SURVIVES"
    log(f"\n  OVERALL: {overall}")

    results = {
        "experiment": "expert_continual_addition",
        "n_pilot_adapters": n_merged,
        "n_new_experts": n_new_merged,
        "new_domains": new_domains,
        "monitor_domains": MONITOR_DOMAINS,
        "regressions": regressions,
        "new_improvements": new_improvements,
        "kill_criteria": {
            "K1_avg_regression_pct": avg_regression,
            "K1_max_regression_pct": max_regression,
            "K1_threshold": 3.0,
            "K1_result": k1_result,
            "K2_quality_ratio": quality_ratio,
            "K2_threshold": 0.50,
            "K2_result": k2_result,
            "K3_total_cost": total_cost,
            "K3_threshold": 5.0,
            "K3_result": k3_result,
        },
        "training": {
            "per_expert_times": train_times,
            "total_train_time_s": total_train_time,
            "estimated_cost": total_cost,
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
