#!/usr/bin/env python3
"""FFN-only vs all-modules LoRA composition at macro scale (Qwen2.5-7B).

Tests whether FFN-only LoRA experts (gate_proj, up_proj, down_proj only)
produce better orthogonality and comparable quality vs all-modules experts
(q/k/v/o/gate/up/down) from the pilot50 run.

Experiment:
  1. Train 10 FFN-only experts on same domains as matched pilot50 experts
  2. Compare individual quality (per-domain PPL)
  3. Compare pairwise orthogonality (mean|cos|)
  4. Compare composition quality at N=5,10

Kill criteria:
  K1: FFN-only per-domain PPL >5% worse than all-modules
  K2: FFN-only pairwise cos NOT lower than all-modules
  K3: FFN-only N=10 composition quality >3% worse than all-modules

Usage (on RunPod):
    cd /workspace/llm
    python macro/ffn_only_macro_composition/run_ffn_only_macro.py
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # /workspace/llm
DATA_DIR = REPO_ROOT / "data" / "distillation"
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
RESULTS_DIR = REPO_ROOT / "results" / "ffn_only_macro"
FFN_ADAPTER_DIR = RESULTS_DIR / "adapters"
PILOT50_ADAPTER_DIR = REPO_ROOT / "data" / "adapters"

# 10 domains spanning clusters: 3 code, 3 science, 2 professional, 2 writing
DOMAINS = [
    "python", "rust", "javascript",       # code cluster
    "physics", "biology", "chemistry",     # science cluster
    "legal", "finance",                    # professional cluster
    "creative-fiction", "technical-writing" # writing cluster
]

# Training config (matches pilot50 exactly, except target_modules)
RANK = 16
ALPHA = 16
STEPS = 300
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LENGTH = 1024
SEED = 42

# Module configs
FFN_ONLY_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]

# Eval config
EVAL_SAMPLES = 200  # per-domain eval samples
COMPOSITION_NS = [5, 10]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Phase 1: Train FFN-only experts ───────────────────────────────────

def train_ffn_only_expert(domain: str):
    """Train a single FFN-only QLoRA expert."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    adapter_out = FFN_ADAPTER_DIR / domain
    adapter_out.mkdir(parents=True, exist_ok=True)

    if (adapter_out / "adapter_config.json").exists():
        log(f"  {domain}: FFN-only adapter exists, skipping")
        return 0.0

    data_path = DATA_DIR / domain / "train.jsonl"
    if not data_path.exists():
        log(f"  {domain}: no training data at {data_path}, skipping")
        return -1.0

    log(f"Training FFN-only expert: {domain}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=FFN_ONLY_MODULES,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def format_messages(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_messages)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(adapter_out / "checkpoints"),
            max_steps=STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=min(10, STEPS // 10),
            logging_steps=50,
            save_steps=STEPS,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=SEED,
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=True,
            report_to="none",
        ),
    )

    train_result = trainer.train()
    train_loss = train_result.training_loss

    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    ckpt_dir = adapter_out / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    elapsed = time.time() - start
    log(f"  {domain}: done in {elapsed:.0f}s, loss={train_loss:.4f}")

    meta = {
        "domain": domain,
        "type": "ffn_only",
        "target_modules": FFN_ONLY_MODULES,
        "rank": RANK, "steps": STEPS, "lr": LR,
        "train_loss": float(train_loss),
        "train_time_s": float(elapsed),
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(adapter_out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    del trainer, model, tokenizer, dataset
    gc.collect()
    import torch as _torch
    _torch.cuda.empty_cache()
    _torch.cuda.reset_peak_memory_stats()

    return elapsed


def phase1_train():
    """Train all 10 FFN-only experts."""
    log("=" * 72)
    log("PHASE 1: Train FFN-only experts")
    log(f"  Domains: {DOMAINS}")
    log(f"  Modules: {FFN_ONLY_MODULES}")
    log("=" * 72)

    results = {}
    for domain in DOMAINS:
        elapsed = train_ffn_only_expert(domain)
        results[domain] = elapsed

    log(f"Phase 1 complete. Trained {sum(1 for v in results.values() if v > 0)} experts.")
    return results


# ── Phase 2: Compare quality (per-domain PPL) ────────────────────────

def eval_ppl(model, tokenizer, texts, max_length=MAX_SEQ_LENGTH):
    """Compute mean perplexity over a list of texts."""
    import torch

    total_loss = 0.0
    total_tokens = 0
    model.eval()

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float("inf")


def load_eval_data(domain, n_samples=EVAL_SAMPLES):
    """Load eval data for a domain (last N samples from train.jsonl as held-out)."""
    data_path = DATA_DIR / domain / "train.jsonl"
    with open(data_path) as f:
        lines = f.readlines()

    # Use last n_samples as eval (not seen during training with packing)
    eval_lines = lines[-n_samples:] if len(lines) > n_samples else lines
    texts = []
    for line in eval_lines:
        example = json.loads(line)
        # Just use the raw text for PPL eval
        if "messages" in example:
            text = " ".join(m.get("content", "") for m in example["messages"])
        elif "text" in example:
            text = example["text"]
        else:
            continue
        texts.append(text)
    return texts


def phase2_quality():
    """Compare per-domain PPL: FFN-only vs all-modules vs base."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    log("=" * 72)
    log("PHASE 2: Per-domain PPL comparison")
    log("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    results = {}

    for domain in DOMAINS:
        log(f"  Evaluating: {domain}")
        eval_texts = load_eval_data(domain)
        if not eval_texts:
            log(f"    No eval data, skipping")
            continue

        domain_results = {}

        # Base model PPL
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE, trust_remote_code=True)
        domain_results["base_ppl"] = eval_ppl(base_model, tokenizer, eval_texts)
        log(f"    Base PPL: {domain_results['base_ppl']:.4f}")

        # All-modules (pilot50) PPL
        pilot_adapter = PILOT50_ADAPTER_DIR / domain
        if (pilot_adapter / "adapter_config.json").exists():
            pilot_model = PeftModel.from_pretrained(base_model, str(pilot_adapter))
            domain_results["all_modules_ppl"] = eval_ppl(pilot_model, tokenizer, eval_texts)
            log(f"    All-modules PPL: {domain_results['all_modules_ppl']:.4f}")
            del pilot_model

        # FFN-only PPL
        ffn_adapter = FFN_ADAPTER_DIR / domain
        if (ffn_adapter / "adapter_config.json").exists():
            ffn_model = PeftModel.from_pretrained(base_model, str(ffn_adapter))
            domain_results["ffn_only_ppl"] = eval_ppl(ffn_model, tokenizer, eval_texts)
            log(f"    FFN-only PPL: {domain_results['ffn_only_ppl']:.4f}")
            del ffn_model

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        # Compute ratios
        if "all_modules_ppl" in domain_results and "ffn_only_ppl" in domain_results:
            ratio = domain_results["ffn_only_ppl"] / domain_results["all_modules_ppl"]
            domain_results["ffn_vs_all_ratio"] = ratio
            log(f"    FFN/All ratio: {ratio:.4f} (>1.05 = K1 fail)")

        results[domain] = domain_results

    return results


# ── Phase 3: Compare orthogonality ────────────────────────────────────

def extract_lora_delta(adapter_path):
    """Extract flattened LoRA delta vector (B @ A for all modules)."""
    import torch
    from safetensors.torch import load_file

    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.exists():
        return None

    state_dict = load_file(str(weights_path))

    # Group A and B matrices by module
    deltas = []
    modules = set()
    for key in state_dict:
        parts = key.rsplit(".", 2)
        if len(parts) >= 2:
            module_name = key.rsplit(".lora_", 1)[0] if ".lora_" in key else parts[0]
            modules.add(module_name)

    for module in sorted(modules):
        a_key = f"{module}.lora_A.weight"
        b_key = f"{module}.lora_B.weight"
        if a_key in state_dict and b_key in state_dict:
            A = state_dict[a_key].float()
            B = state_dict[b_key].float()
            delta = (B @ A).flatten()
            deltas.append(delta)

    if not deltas:
        return None
    return torch.cat(deltas).numpy()


def phase3_orthogonality():
    """Compare pairwise cosine similarity: FFN-only vs all-modules."""
    log("=" * 72)
    log("PHASE 3: Pairwise orthogonality comparison")
    log("=" * 72)

    # Extract FFN-only deltas
    ffn_deltas = {}
    for domain in DOMAINS:
        delta = extract_lora_delta(FFN_ADAPTER_DIR / domain)
        if delta is not None:
            ffn_deltas[domain] = delta

    # Extract all-modules deltas
    all_deltas = {}
    for domain in DOMAINS:
        delta = extract_lora_delta(PILOT50_ADAPTER_DIR / domain)
        if delta is not None:
            all_deltas[domain] = delta

    # Compute pairwise cosines for FFN-only
    ffn_cosines = []
    ffn_domains_list = sorted(ffn_deltas.keys())
    for i in range(len(ffn_domains_list)):
        for j in range(i + 1, len(ffn_domains_list)):
            d1, d2 = ffn_domains_list[i], ffn_domains_list[j]
            v1, v2 = ffn_deltas[d1], ffn_deltas[d2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            ffn_cosines.append({"pair": f"{d1}-{d2}", "cos": float(abs(cos))})

    # For all-modules, we can only compare cosines among the FFN components
    # since FFN-only doesn't have attention components.
    # But we compare absolute cosines of the full delta vectors.
    all_cosines = []
    all_domains_list = sorted(d for d in all_deltas.keys() if d in ffn_deltas)
    for i in range(len(all_domains_list)):
        for j in range(i + 1, len(all_domains_list)):
            d1, d2 = all_domains_list[i], all_domains_list[j]
            v1, v2 = all_deltas[d1], all_deltas[d2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            all_cosines.append({"pair": f"{d1}-{d2}", "cos": float(abs(cos))})

    ffn_mean_cos = np.mean([c["cos"] for c in ffn_cosines]) if ffn_cosines else float("nan")
    all_mean_cos = np.mean([c["cos"] for c in all_cosines]) if all_cosines else float("nan")

    log(f"  FFN-only mean|cos|: {ffn_mean_cos:.6f}")
    log(f"  All-modules mean|cos|: {all_mean_cos:.6f}")
    log(f"  Ratio (FFN/All): {ffn_mean_cos / all_mean_cos:.4f} (<1.0 = FFN more orthogonal)")

    return {
        "ffn_mean_cos": float(ffn_mean_cos),
        "all_mean_cos": float(all_mean_cos),
        "ratio": float(ffn_mean_cos / all_mean_cos) if all_mean_cos > 0 else float("nan"),
        "ffn_pairs": ffn_cosines,
        "all_pairs": all_cosines,
    }


# ── Phase 4: Compare composition quality ──────────────────────────────

def phase4_composition():
    """Compare composed model quality at N=5,10."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from safetensors.torch import load_file

    log("=" * 72)
    log("PHASE 4: Composition quality comparison")
    log("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    results = {}

    for N in COMPOSITION_NS:
        domains_subset = DOMAINS[:N]
        log(f"\n  Composition N={N}: {domains_subset}")

        for config_name, adapter_dir, modules in [
            ("ffn_only", FFN_ADAPTER_DIR, FFN_ONLY_MODULES),
            ("all_modules", PILOT50_ADAPTER_DIR, ALL_MODULES),
        ]:
            log(f"    Config: {config_name}")

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=bnb_config,
                device_map="auto", torch_dtype=torch.bfloat16,
                cache_dir=HF_CACHE, trust_remote_code=True)

            # Compute merged delta: sum of all adapter deltas
            merged_delta = {}
            n_loaded = 0
            for domain in domains_subset:
                adapter_path = adapter_dir / domain / "adapter_model.safetensors"
                if not adapter_path.exists():
                    log(f"      {domain}: no adapter, skipping")
                    continue

                state_dict = load_file(str(adapter_path))
                for key, tensor in state_dict.items():
                    if key not in merged_delta:
                        merged_delta[key] = tensor.clone().float()
                    else:
                        merged_delta[key] += tensor.float()
                n_loaded += 1

            if n_loaded == 0:
                log(f"      No adapters loaded, skipping")
                continue

            # Average the merged delta
            for key in merged_delta:
                merged_delta[key] /= n_loaded

            # Apply merged delta to base model
            # Load as single adapter then manually set weights
            first_adapter = None
            for domain in domains_subset:
                p = adapter_dir / domain
                if (p / "adapter_config.json").exists():
                    first_adapter = p
                    break

            if first_adapter is None:
                continue

            merged_model = PeftModel.from_pretrained(model, str(first_adapter))

            # Replace adapter weights with merged average
            for name, param in merged_model.named_parameters():
                # Map parameter name to safetensors key
                for key, val in merged_delta.items():
                    if key.replace("base_model.model.", "") in name or name.endswith(key):
                        param.data.copy_(val.to(param.device, param.dtype))
                        break

            # Evaluate on all domains in the subset
            domain_ppls = {}
            for domain in domains_subset:
                eval_texts = load_eval_data(domain, n_samples=50)
                if eval_texts:
                    ppl = eval_ppl(merged_model, tokenizer, eval_texts)
                    domain_ppls[domain] = ppl
                    log(f"      {domain}: PPL={ppl:.4f}")

            avg_ppl = np.mean(list(domain_ppls.values())) if domain_ppls else float("nan")
            log(f"    {config_name} N={N} avg PPL: {avg_ppl:.4f}")

            results[f"{config_name}_N{N}"] = {
                "avg_ppl": float(avg_ppl),
                "domain_ppls": {k: float(v) for k, v in domain_ppls.items()},
                "n_experts": n_loaded,
            }

            del merged_model, model
            gc.collect()
            torch.cuda.empty_cache()

    # Compute kill criteria
    for N in COMPOSITION_NS:
        ffn_key = f"ffn_only_N{N}"
        all_key = f"all_modules_N{N}"
        if ffn_key in results and all_key in results:
            ratio = results[ffn_key]["avg_ppl"] / results[all_key]["avg_ppl"]
            results[f"composition_ratio_N{N}"] = float(ratio)
            log(f"  N={N} FFN/All composition ratio: {ratio:.4f} (>1.03 = K3 fail)")

    return results


# ── Main ──────────────────────────────────────────────────────────────

def run_experiment():
    """Run full experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("FFN-ONLY vs ALL-MODULES MACRO COMPOSITION EXPERIMENT")
    log(f"  Base model: {BASE_MODEL}")
    log(f"  Domains: {DOMAINS}")
    log(f"  FFN modules: {FFN_ONLY_MODULES}")
    log(f"  All modules: {ALL_MODULES}")
    log("=" * 72)

    t0 = time.time()
    all_results = {}

    # Phase 1: Train FFN-only experts
    train_results = phase1_train()
    all_results["training"] = {k: float(v) for k, v in train_results.items()}

    # Phase 2: Per-domain PPL comparison
    quality_results = phase2_quality()
    all_results["quality"] = quality_results

    # Phase 3: Orthogonality comparison
    ortho_results = phase3_orthogonality()
    all_results["orthogonality"] = ortho_results

    # Phase 4: Composition quality
    comp_results = phase4_composition()
    all_results["composition"] = comp_results

    # ── Kill criteria assessment ──
    log("\n" + "=" * 72)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 72)

    kills = {}

    # K1: Per-domain PPL
    ratios = [v.get("ffn_vs_all_ratio", 1.0) for v in quality_results.values()
              if isinstance(v, dict) and "ffn_vs_all_ratio" in v]
    if ratios:
        max_ratio = max(ratios)
        mean_ratio = np.mean(ratios)
        k1_pass = max_ratio <= 1.05
        kills["K1_per_domain_ppl"] = {
            "pass": bool(k1_pass),
            "mean_ratio": float(mean_ratio),
            "max_ratio": float(max_ratio),
            "threshold": 1.05,
        }
        log(f"  K1 (per-domain PPL): {'PASS' if k1_pass else 'FAIL'} "
            f"(max ratio={max_ratio:.4f}, threshold=1.05)")

    # K2: Orthogonality
    if "ratio" in ortho_results and not np.isnan(ortho_results["ratio"]):
        k2_pass = ortho_results["ratio"] < 1.0
        kills["K2_orthogonality"] = {
            "pass": bool(k2_pass),
            "ffn_cos": ortho_results["ffn_mean_cos"],
            "all_cos": ortho_results["all_mean_cos"],
            "ratio": ortho_results["ratio"],
        }
        log(f"  K2 (orthogonality): {'PASS' if k2_pass else 'FAIL'} "
            f"(ratio={ortho_results['ratio']:.4f}, need <1.0)")

    # K3: Composition quality
    for N in COMPOSITION_NS:
        key = f"composition_ratio_N{N}"
        if key in comp_results:
            ratio = comp_results[key]
            k3_pass = ratio <= 1.03
            kills[f"K3_composition_N{N}"] = {
                "pass": bool(k3_pass),
                "ratio": ratio,
                "threshold": 1.03,
            }
            log(f"  K3 N={N} (composition): {'PASS' if k3_pass else 'FAIL'} "
                f"(ratio={ratio:.4f}, threshold=1.03)")

    all_results["kill_criteria"] = kills
    all_results["elapsed_s"] = time.time() - t0

    # Overall verdict
    all_pass = all(v.get("pass", False) for v in kills.values())
    any_fail = any(not v.get("pass", True) for v in kills.values())
    verdict = "PROVEN" if all_pass and kills else "KILLED" if any_fail else "INCONCLUSIVE"
    all_results["verdict"] = verdict
    log(f"\n  VERDICT: {verdict}")

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment()
