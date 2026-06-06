#!/usr/bin/env python3
"""Adaptive rank macro validation: spectral profiling + retraining experiment.

Phase 1: Measure SVD spectral profiles of all 50 pilot adapters.
  - Compute r_99, r_95, effective rank, SNR for each adapter's weight deltas
  - Apply the r_99/r_95 heuristic to predict optimal rank per domain

Phase 2: Retrain 5 diverse-domain experts at predicted optimal rank.
  - Compare quality (PPL on domain data) against fixed rank-16 baseline
  - Measure if rank reduction saves parameters without quality loss

Kill criteria:
  K1: predicted rank does not correlate with empirical optimal rank (rho < 0.5)
  K2: per-domain rank does not improve mean quality over fixed rank-16 by >3%
  K3: spectral measurement takes >60s per adapter (impractical at N=500)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/adaptive_rank_macro_validation")
SEED = 42


def get_available_adapters():
    """Get list of available adapter names."""
    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def load_adapter_deltas(adapter_name):
    """Load LoRA A and B matrices from an adapter.

    Returns list of (layer_name, A, B) tuples where delta_W = B @ A.
    """
    adapter_path = ADAPTER_DIR / adapter_name

    # Try safetensors first, then pytorch
    st_path = adapter_path / "adapter_model.safetensors"
    pt_path = adapter_path / "adapter_model.bin"

    if st_path.exists():
        state_dict = load_file(str(st_path))
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    # Group A and B matrices by layer
    layers = {}
    for key, tensor in state_dict.items():
        if "lora_A" in key:
            layer_name = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name]["A"] = tensor.float()
        elif "lora_B" in key:
            layer_name = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name]["B"] = tensor.float()

    result = []
    for layer_name in sorted(layers.keys()):
        if "A" in layers[layer_name] and "B" in layers[layer_name]:
            result.append((layer_name, layers[layer_name]["A"], layers[layer_name]["B"]))

    return result


def compute_spectral_profile(A, B):
    """Compute spectral profile metrics for a LoRA delta W = B @ A.

    Returns dict with r_99, r_95, effective_rank, SNR, singular_values.
    """
    # Compute delta_W = B @ A
    delta_W = B @ A  # (d_out, d_in)

    # SVD
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    S = S.numpy()

    if S[0] < 1e-12:
        return {
            "r_99": 0, "r_95": 0, "effective_rank": 0.0,
            "snr": 0.0, "singular_values": S.tolist()[:16],
            "total_energy": 0.0,
        }

    # Energy fractions
    energy = S ** 2
    total_energy = energy.sum()
    cumulative = np.cumsum(energy) / total_energy

    # r_99: smallest r such that cumsum >= 0.99
    r_99 = int(np.searchsorted(cumulative, 0.99)) + 1
    r_95 = int(np.searchsorted(cumulative, 0.95)) + 1

    # Shannon effective rank (exponential of spectral entropy)
    p = energy / total_energy
    p = p[p > 1e-12]  # avoid log(0)
    spectral_entropy = -np.sum(p * np.log(p))
    effective_rank = math.exp(spectral_entropy)

    # SNR estimate: ratio of signal energy to noise floor
    if len(S) > 1:
        noise_floor = np.median(S[len(S)//2:]) if len(S) > 4 else S[-1]
        snr = S[0] / max(noise_floor, 1e-12)
    else:
        snr = float("inf")

    return {
        "r_99": int(r_99),
        "r_95": int(r_95),
        "effective_rank": round(float(effective_rank), 2),
        "snr": round(float(snr), 2),
        "singular_values": S.tolist()[:16],
        "total_energy": round(float(total_energy), 6),
        "r_99_r_95_ratio": round(r_99 / max(r_95, 1), 2),
    }


def predict_rank(profile, current_rank=16):
    """Apply r_99/r_95 heuristic to predict optimal rank.

    Heuristic from exp_adaptive_rank_snr_fallback:
    - Default: use r_99
    - If r_99/r_95 > 2.0 (SNR<=10 regime): fall back to r_95
    - Snap to standard LoRA ranks: 4, 8, 16, 32, 64
    """
    standard_ranks = [4, 8, 16, 32, 64]

    ratio = profile["r_99_r_95_ratio"]
    if ratio > 2.0:
        raw_rank = profile["r_95"]
    else:
        raw_rank = profile["r_99"]

    # Snap to nearest standard rank
    predicted = min(standard_ranks, key=lambda r: abs(r - raw_rank))
    return predicted


def retrain_adapter(base_model_path, data_path, rank, domain_name, steps=200, lr=1e-4):
    """Retrain a LoRA adapter at the specified rank.

    Returns (adapter_path, metrics).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # LoRA config at specified rank
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,  # alpha=rank → scaling=1.0
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.train()

    # Load training data
    if data_path.exists():
        dataset = load_dataset("json", data_files=str(data_path), split="train")
    else:
        print(f"  WARNING: no data at {data_path}, skipping retrain")
        return None, None

    # Training loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    rng = np.random.RandomState(SEED)
    losses = []

    for step in range(steps):
        idx = rng.randint(0, len(dataset))
        example = dataset[idx]

        text = example.get("text", example.get("output", ""))
        if not text:
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"    Step {step+1}/{steps}: loss={avg_loss:.4f}")

    # Save adapter
    save_dir = RESULTS_DIR / f"retrained_{domain_name}_r{rank}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))

    final_loss = np.mean(losses[-20:]) if losses else float("inf")
    metrics = {
        "rank": rank,
        "domain": domain_name,
        "steps": steps,
        "final_loss": round(float(final_loss), 4),
        "save_path": str(save_dir),
    }

    del model, optimizer
    torch.cuda.empty_cache()

    return save_dir, metrics


def evaluate_ppl(model_path, eval_data_path, tokenizer_path, is_adapter=False, base_model=None):
    """Evaluate PPL on eval data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter and base_model is not None:
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    model.eval()

    if eval_data_path.exists():
        dataset = load_dataset("json", data_files=str(eval_data_path), split="train")
    else:
        return float("inf")

    total_nll = 0.0
    total_tokens = 0

    for i, example in enumerate(dataset):
        if i >= 200:  # cap eval
            break
        text = example.get("text", example.get("output", ""))
        if not text:
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        inputs["labels"] = inputs["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            n_tokens = inputs["input_ids"].shape[1]
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-domains", type=int, default=5, help="Number of domains to retrain")
    parser.add_argument("--retrain-steps", type=int, default=200, help="Training steps per retrained adapter")
    parser.add_argument("--skip-retrain", action="store_true", help="Only do spectral profiling (Phase 1)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    available_adapters = get_available_adapters()
    n_adapters = len(available_adapters)
    print(f"Found {n_adapters} adapters")

    all_results = {
        "config": {
            "n_adapters": n_adapters,
            "retrain_domains": args.retrain_domains,
            "retrain_steps": args.retrain_steps,
            "seed": SEED,
        },
        "spectral_profiles": {},
        "rank_predictions": {},
        "retrain_results": {},
    }

    # Phase 1: Spectral profiling
    print("\n=== Phase 1: Spectral profiling ===")
    measurement_times = []

    for adapter_name in available_adapters:
        t0 = time.time()
        try:
            layers = load_adapter_deltas(adapter_name)
        except Exception as e:
            print(f"  {adapter_name}: SKIP ({e})")
            continue

        # Aggregate spectral profile across layers
        layer_profiles = {}
        agg_r99 = []
        agg_r95 = []
        agg_eff_rank = []
        agg_snr = []

        for layer_name, A, B in layers:
            profile = compute_spectral_profile(A, B)
            layer_profiles[layer_name] = profile
            agg_r99.append(profile["r_99"])
            agg_r95.append(profile["r_95"])
            agg_eff_rank.append(profile["effective_rank"])
            agg_snr.append(profile["snr"])

        dt = time.time() - t0
        measurement_times.append(dt)

        summary = {
            "mean_r99": round(np.mean(agg_r99), 1),
            "mean_r95": round(np.mean(agg_r95), 1),
            "mean_effective_rank": round(np.mean(agg_eff_rank), 2),
            "mean_snr": round(np.mean(agg_snr), 2),
            "median_r99": int(np.median(agg_r99)),
            "median_r95": int(np.median(agg_r95)),
            "r99_range": [int(min(agg_r99)), int(max(agg_r99))],
            "r95_range": [int(min(agg_r95)), int(max(agg_r95))],
            "n_layers": len(layers),
            "measurement_time_s": round(dt, 2),
        }

        # Predict optimal rank
        agg_profile = {
            "r_99": int(np.median(agg_r99)),
            "r_95": int(np.median(agg_r95)),
            "r_99_r_95_ratio": round(np.median(agg_r99) / max(np.median(agg_r95), 1), 2),
        }
        predicted_rank = predict_rank(agg_profile)

        all_results["spectral_profiles"][adapter_name] = summary
        all_results["rank_predictions"][adapter_name] = {
            "predicted_rank": predicted_rank,
            "current_rank": 16,
            "agg_profile": agg_profile,
        }

        print(f"  {adapter_name}: r99={summary['mean_r99']:.0f} r95={summary['mean_r95']:.0f} "
              f"eff_rank={summary['mean_effective_rank']:.1f} SNR={summary['mean_snr']:.0f} "
              f"→ predicted_rank={predicted_rank} ({dt:.1f}s)")

    # Phase 1 summary
    mean_time = np.mean(measurement_times)
    max_time = np.max(measurement_times)
    print(f"\nSpectral profiling complete:")
    print(f"  Mean measurement time: {mean_time:.2f}s")
    print(f"  Max measurement time: {max_time:.2f}s")
    print(f"  K3: {max_time:.1f}s {'< 60s PASS' if max_time < 60 else '>= 60s KILL'}")

    # Distribution of predicted ranks
    predicted_ranks = [v["predicted_rank"] for v in all_results["rank_predictions"].values()]
    unique, counts = np.unique(predicted_ranks, return_counts=True)
    print(f"\nPredicted rank distribution:")
    for r, c in zip(unique, counts):
        print(f"  rank-{r}: {c} adapters ({100*c/len(predicted_ranks):.0f}%)")

    all_results["phase1_summary"] = {
        "mean_measurement_time_s": round(float(mean_time), 2),
        "max_measurement_time_s": round(float(max_time), 2),
        "K3_pass": max_time < 60,
        "rank_distribution": {str(int(r)): int(c) for r, c in zip(unique, counts)},
    }

    if args.skip_retrain:
        print("\n=== Skipping Phase 2 (--skip-retrain) ===")
    else:
        # Phase 2: Retrain at predicted ranks
        print("\n=== Phase 2: Retrain at predicted ranks ===")

        # Select domains to retrain: pick diverse predicted ranks
        # Group adapters by predicted rank, pick one from each group
        rank_groups = {}
        for adapter_name, pred in all_results["rank_predictions"].items():
            r = pred["predicted_rank"]
            if r not in rank_groups:
                rank_groups[r] = []
            rank_groups[r].append(adapter_name)

        retrain_candidates = []
        for r in sorted(rank_groups.keys()):
            rng.shuffle(rank_groups[r])
            retrain_candidates.append((rank_groups[r][0], r))

        # If we have fewer groups than requested, add more from the largest group
        while len(retrain_candidates) < args.retrain_domains and rank_groups:
            largest_group = max(rank_groups.values(), key=len)
            if len(largest_group) > 1:
                extra = largest_group[1]
                r = all_results["rank_predictions"][extra]["predicted_rank"]
                retrain_candidates.append((extra, r))
                largest_group.remove(extra)
            else:
                break

        retrain_candidates = retrain_candidates[:args.retrain_domains]
        print(f"Retraining {len(retrain_candidates)} adapters:")
        for adapter_name, pred_rank in retrain_candidates:
            print(f"  {adapter_name}: rank-16 → rank-{pred_rank}")

        # Find data directories for each adapter
        data_base = Path("/workspace/llm/data/pilot50")
        if not data_base.exists():
            data_base = Path("/workspace/llm/data")

        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        for adapter_name, pred_rank in retrain_candidates:
            print(f"\n  Retraining {adapter_name} at rank-{pred_rank}...")

            # Find training data
            data_path = data_base / adapter_name / "train.jsonl"
            if not data_path.exists():
                # Try alternate paths
                for alt in [data_base / f"{adapter_name}.jsonl",
                            data_base / adapter_name / "data.jsonl"]:
                    if alt.exists():
                        data_path = alt
                        break

            if not data_path.exists():
                print(f"    No training data found at {data_path}, skipping")
                all_results["retrain_results"][adapter_name] = {"error": "no training data"}
                continue

            save_dir, metrics = retrain_adapter(
                BASE_MODEL, data_path, pred_rank, adapter_name,
                steps=args.retrain_steps, lr=1e-4
            )

            if metrics is None:
                continue

            # Evaluate both original (rank-16) and retrained (predicted rank)
            eval_path = data_path  # use training data for PPL (held-out would be better)

            from peft import PeftModel
            orig_adapter_path = ADAPTER_DIR / adapter_name
            try:
                orig_model = PeftModel.from_pretrained(base_model, str(orig_adapter_path))
                orig_model.eval()

                # Compute PPL for original
                total_nll = 0.0
                total_tokens = 0
                from datasets import load_dataset
                dataset = load_dataset("json", data_files=str(eval_path), split="train")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                for i, example in enumerate(dataset):
                    if i >= 100:
                        break
                    text = example.get("text", example.get("output", ""))
                    if not text:
                        continue
                    inputs = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=512
                    ).to(orig_model.device)
                    inputs["labels"] = inputs["input_ids"].clone()
                    with torch.no_grad():
                        outputs = orig_model(**inputs)
                        n_tokens = inputs["input_ids"].shape[1]
                        total_nll += outputs.loss.item() * n_tokens
                        total_tokens += n_tokens
                orig_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

                del orig_model
                torch.cuda.empty_cache()

                # Compute PPL for retrained
                retrained_model = PeftModel.from_pretrained(base_model, str(save_dir))
                retrained_model.eval()
                total_nll = 0.0
                total_tokens = 0
                for i, example in enumerate(dataset):
                    if i >= 100:
                        break
                    text = example.get("text", example.get("output", ""))
                    if not text:
                        continue
                    inputs = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=512
                    ).to(retrained_model.device)
                    inputs["labels"] = inputs["input_ids"].clone()
                    with torch.no_grad():
                        outputs = retrained_model(**inputs)
                        n_tokens = inputs["input_ids"].shape[1]
                        total_nll += outputs.loss.item() * n_tokens
                        total_tokens += n_tokens
                retrained_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

                del retrained_model
                torch.cuda.empty_cache()

                ppl_change = (retrained_ppl - orig_ppl) / orig_ppl * 100
                metrics["orig_ppl"] = round(orig_ppl, 4)
                metrics["retrained_ppl"] = round(retrained_ppl, 4)
                metrics["ppl_change_pct"] = round(ppl_change, 2)
                print(f"    orig_ppl={orig_ppl:.4f}, retrained_ppl={retrained_ppl:.4f}, change={ppl_change:+.1f}%")

            except Exception as e:
                print(f"    Eval failed: {e}")
                metrics["eval_error"] = str(e)

            all_results["retrain_results"][adapter_name] = metrics

        # Phase 2 summary
        retrained = [v for v in all_results["retrain_results"].values() if "ppl_change_pct" in v]
        if retrained:
            changes = [v["ppl_change_pct"] for v in retrained]
            mean_change = np.mean(changes)
            print(f"\n  Mean PPL change: {mean_change:+.2f}%")
            print(f"  K2: mean change = {mean_change:+.1f}% (threshold: improvement >3%) → "
                  f"{'PASS' if mean_change < -3 else 'KILL'}")

            all_results["phase2_summary"] = {
                "n_retrained": len(retrained),
                "mean_ppl_change_pct": round(float(mean_change), 2),
                "K2_pass": mean_change < -3,
            }

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
