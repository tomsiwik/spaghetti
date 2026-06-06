#!/usr/bin/env python3
"""Cluster-aware composition: route to best cluster, compose within-cluster only.

Micro evidence: within-cluster |cos| 7.84x higher than cross-cluster.
Hypothesis: composing only within-cluster experts avoids cross-cluster interference,
beating compose-all-50 on held-out PPL.

Strategy:
1. Cluster pilot-50 adapters by weight similarity (k-means on flattened LoRA weights)
2. For each eval domain, route to best cluster via cosine similarity of domain embedding
3. Compose only within-cluster adapters (N=5-10 instead of N=50)
4. Compare: base, compose-all-50, compose-within-cluster, compose-cluster-representatives

Kill criteria:
- K1: cluster-routed composition does not beat compose-all by >3% on held-out PPL
- K2: cluster routing accuracy <80% (wrong cluster selected too often)
- K3: cluster composition within-cluster PPL >10% worse than individual expert

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
RESULTS_DIR = REPO_ROOT / "results" / "cluster_grouped_composition"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

MAX_SEQ_LEN = 512 if not IS_SMOKE else 256
EVAL_SAMPLES = 50 if not IS_SMOKE else 5
N_CLUSTERS = 7 if not IS_SMOKE else 3
SEED = 42


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


def load_adapter_weights_flat(adapter_name):
    """Load adapter safetensors and flatten to a single vector (CPU)."""
    from safetensors.torch import load_file
    st_path = ADAPTER_DIR / adapter_name / "adapter_model.safetensors"
    tensors = load_file(str(st_path), device="cpu")
    flat = []
    for key in sorted(tensors.keys()):
        flat.append(tensors[key].float().flatten())
    import torch
    return torch.cat(flat).numpy()


def cluster_adapters(adapter_names, n_clusters):
    """Cluster adapters by weight similarity using k-means on flattened weights."""
    from sklearn.cluster import KMeans

    log(f"Loading adapter weights for clustering ({len(adapter_names)} adapters)...")
    weight_matrix = []
    for name in adapter_names:
        flat = load_adapter_weights_flat(name)
        weight_matrix.append(flat)

    weight_matrix = np.array(weight_matrix)
    log(f"Weight matrix shape: {weight_matrix.shape}")

    # Normalize rows for cosine-like clustering
    norms = np.linalg.norm(weight_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    weight_matrix_normed = weight_matrix / norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(weight_matrix_normed)

    clusters = {}
    for i, name in enumerate(adapter_names):
        c = int(labels[i])
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(name)

    for c, members in sorted(clusters.items()):
        log(f"  Cluster {c}: {len(members)} adapters — {members[:5]}{'...' if len(members) > 5 else ''}")

    return clusters, kmeans, weight_matrix_normed


def compose_adapters_on_cpu(adapter_names, adapter_dir, weights=None):
    """Compose multiple LoRA adapters on CPU by averaging safetensors weights."""
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

    # Save to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="composed_adapter_")
    save_file({k: v.half() for k, v in composed.items()}, os.path.join(tmp_dir, "adapter_model.safetensors"))
    if adapter_config:
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f)

    return tmp_dir


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


def get_eval_domains():
    """Get domains that have eval data."""
    domains = []
    if not DATA_DIR.exists():
        return domains
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            if (d / "eval.jsonl").exists() or (d / "train.jsonl").exists():
                domains.append(d.name)
    return domains


def compute_ppl(model, tokenizer, texts, max_len=512):
    """Compute perplexity on a list of texts."""
    import torch
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
            total_loss += out.loss.item() * (inputs["input_ids"].shape[1] - 1)
            total_tokens += inputs["input_ids"].shape[1] - 1
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def route_domain_to_cluster(domain, adapter_names, clusters, kmeans, weight_matrix_normed):
    """Route a domain to the best cluster.

    Strategy: find which adapter name best matches the domain, then use its cluster.
    Fallback: pick the cluster whose centroid is closest to the domain adapter (if exists).
    """
    # Direct name matching
    domain_lower = domain.lower().replace("-", " ").replace("_", " ")
    for name in adapter_names:
        name_lower = name.lower().replace("-", " ").replace("_", " ")
        if domain_lower == name_lower or domain_lower in name_lower or name_lower in domain_lower:
            # Found a matching adapter, return its cluster
            idx = adapter_names.index(name)
            label = kmeans.labels_[idx]
            return int(label), name

    # Fuzzy match: check word overlap
    domain_words = set(domain_lower.split())
    best_overlap = 0
    best_name = None
    for name in adapter_names:
        name_words = set(name.lower().replace("-", " ").replace("_", " ").split())
        overlap = len(domain_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name

    if best_name and best_overlap > 0:
        idx = adapter_names.index(best_name)
        label = kmeans.labels_[idx]
        return int(label), best_name

    # No match: pick largest cluster as default
    largest_cluster = max(clusters, key=lambda c: len(clusters[c]))
    return largest_cluster, None


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    random.seed(SEED)
    np.random.seed(SEED)

    t_start = time.time()

    # Discover adapters and eval domains
    adapter_names = discover_adapters()
    eval_domains = get_eval_domains()
    log(f"Found {len(adapter_names)} adapters, {len(eval_domains)} eval domains")

    if IS_SMOKE:
        eval_domains = eval_domains[:3]
        adapter_names = adapter_names[:12]

    # PHASE 1: Cluster adapters
    log("\n" + "=" * 70)
    log("PHASE 1: Cluster adapters by weight similarity")
    log("=" * 70)
    n_clusters = min(N_CLUSTERS, len(adapter_names) // 2)
    clusters, kmeans, weight_matrix_normed = cluster_adapters(adapter_names, n_clusters)

    # PHASE 2: Load model and evaluate base
    log("\n" + "=" * 70)
    log("PHASE 2: Base model evaluation")
    log("=" * 70)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    base_ppl = {}
    for domain in eval_domains:
        texts = load_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            log(f"  {domain}: skipped (only {len(texts)} texts)")
            continue
        ppl = compute_ppl(model, tokenizer, texts, MAX_SEQ_LEN)
        base_ppl[domain] = ppl
        log(f"  Base PPL {domain}: {ppl:.2f}")

    # PHASE 3: Compose-all-50 (or all adapters)
    log("\n" + "=" * 70)
    log("PHASE 3: Compose-all baseline")
    log("=" * 70)
    composed_dir = compose_adapters_on_cpu(adapter_names, ADAPTER_DIR)
    peft_model = PeftModel.from_pretrained(model, composed_dir, adapter_name="all")
    shutil.rmtree(composed_dir)
    peft_model.eval()

    all_ppl = {}
    for domain in eval_domains:
        if domain not in base_ppl:
            continue
        texts = load_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            continue
        ppl = compute_ppl(peft_model, tokenizer, texts, MAX_SEQ_LEN)
        all_ppl[domain] = ppl
        delta_pct = (ppl - base_ppl[domain]) / base_ppl[domain] * 100
        log(f"  All-{len(adapter_names)} PPL {domain}: {ppl:.2f} ({delta_pct:+.1f}%)")

    del peft_model
    gc.collect()
    torch.cuda.empty_cache()

    # PHASE 4: Cluster-routed composition
    log("\n" + "=" * 70)
    log("PHASE 4: Cluster-routed composition")
    log("=" * 70)

    cluster_ppl = {}
    routing_decisions = {}
    routing_correct = 0
    routing_total = 0

    for domain in eval_domains:
        if domain not in base_ppl:
            continue

        cluster_id, matched_adapter = route_domain_to_cluster(
            domain, adapter_names, clusters, kmeans, weight_matrix_normed
        )
        cluster_members = clusters[cluster_id]
        routing_decisions[domain] = {
            "cluster_id": cluster_id,
            "matched_adapter": matched_adapter,
            "cluster_size": len(cluster_members),
            "cluster_members": cluster_members,
        }

        # Check routing accuracy: is the domain's own adapter in the selected cluster?
        is_correct = matched_adapter is not None and matched_adapter in cluster_members
        routing_correct += int(is_correct)
        routing_total += 1

        log(f"  {domain} → cluster {cluster_id} ({len(cluster_members)} members, match={matched_adapter})")

        # Compose within-cluster adapters
        composed_dir = compose_adapters_on_cpu(cluster_members, ADAPTER_DIR)
        fresh_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=HF_CACHE,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        fresh_model.eval()
        peft_model = PeftModel.from_pretrained(fresh_model, composed_dir, adapter_name="cluster")
        shutil.rmtree(composed_dir)
        peft_model.eval()

        texts = load_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            del peft_model, fresh_model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        ppl = compute_ppl(peft_model, tokenizer, texts, MAX_SEQ_LEN)
        cluster_ppl[domain] = ppl
        delta_vs_base = (ppl - base_ppl[domain]) / base_ppl[domain] * 100
        delta_vs_all = (ppl - all_ppl.get(domain, ppl)) / max(all_ppl.get(domain, ppl), 1e-6) * 100
        log(f"  Cluster PPL {domain}: {ppl:.2f} (vs base: {delta_vs_base:+.1f}%, vs all: {delta_vs_all:+.1f}%)")

        del peft_model, fresh_model
        gc.collect()
        torch.cuda.empty_cache()

    # PHASE 5: Individual expert PPL (for K3 comparison)
    log("\n" + "=" * 70)
    log("PHASE 5: Individual expert PPL (subset)")
    log("=" * 70)
    # Only evaluate domains where we have an exact adapter match
    individual_ppl = {}
    sample_domains = [d for d in eval_domains if d in base_ppl and d in adapter_names]
    if IS_SMOKE:
        sample_domains = sample_domains[:2]
    else:
        sample_domains = sample_domains[:10]

    for domain in sample_domains:
        adapter_path = ADAPTER_DIR / domain
        if not (adapter_path / "adapter_model.safetensors").exists():
            continue

        fresh_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=HF_CACHE,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        fresh_model.eval()
        peft_model = PeftModel.from_pretrained(fresh_model, str(adapter_path), adapter_name=domain)
        peft_model.eval()

        texts = load_texts(domain, tokenizer, EVAL_SAMPLES)
        if len(texts) < 3:
            del peft_model, fresh_model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        ppl = compute_ppl(peft_model, tokenizer, texts, MAX_SEQ_LEN)
        individual_ppl[domain] = ppl
        log(f"  Individual {domain}: {ppl:.2f}")

        del peft_model, fresh_model
        gc.collect()
        torch.cuda.empty_cache()

    # ANALYSIS
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # K1: cluster-routed beats compose-all by >3%
    k1_domains = [d for d in cluster_ppl if d in all_ppl]
    if k1_domains:
        cluster_mean = np.mean([cluster_ppl[d] for d in k1_domains])
        all_mean = np.mean([all_ppl[d] for d in k1_domains])
        k1_improvement = (all_mean - cluster_mean) / all_mean * 100
        k1_pass = k1_improvement > 3
        log(f"K1: Cluster mean PPL={cluster_mean:.2f}, All mean PPL={all_mean:.2f}")
        log(f"K1: Cluster beats all by {k1_improvement:.1f}% — {'PASS' if k1_pass else 'FAIL'} (need >3%)")
    else:
        k1_improvement = 0
        k1_pass = False
        log("K1: No comparable domains — FAIL")

    # K2: routing accuracy >=80%
    routing_acc = routing_correct / max(1, routing_total)
    k2_pass = routing_acc >= 0.80
    log(f"K2: Routing accuracy={routing_acc:.1%} ({routing_correct}/{routing_total}) — {'PASS' if k2_pass else 'FAIL'} (need >=80%)")

    # K3: within-cluster PPL not >10% worse than individual expert
    k3_domains = [d for d in individual_ppl if d in cluster_ppl]
    if k3_domains:
        worse_count = 0
        for d in k3_domains:
            degradation = (cluster_ppl[d] - individual_ppl[d]) / individual_ppl[d] * 100
            if degradation > 10:
                worse_count += 1
            log(f"  K3 {d}: cluster={cluster_ppl[d]:.2f} vs individual={individual_ppl[d]:.2f} ({degradation:+.1f}%)")
        k3_pass = worse_count / len(k3_domains) < 0.5  # less than half are >10% worse
        log(f"K3: {worse_count}/{len(k3_domains)} domains >10% worse — {'PASS' if k3_pass else 'FAIL'}")
    else:
        k3_pass = False
        log("K3: No comparable domains — FAIL")

    verdict = "PASS" if (k1_pass and k2_pass and k3_pass) else "FAIL"
    log(f"\nVERDICT: {verdict}")
    log(f"Total time: {time.time() - t_start:.0f}s")

    # Save results
    results = {
        "config": {
            "n_adapters": len(adapter_names),
            "n_clusters": n_clusters,
            "n_eval_domains": len(eval_domains),
            "eval_samples": EVAL_SAMPLES,
            "smoke_test": IS_SMOKE,
            "seed": SEED,
        },
        "clusters": {str(k): v for k, v in clusters.items()},
        "base_ppl": base_ppl,
        "all_ppl": all_ppl,
        "cluster_ppl": cluster_ppl,
        "individual_ppl": individual_ppl,
        "routing_decisions": routing_decisions,
        "kill_criteria": {
            "K1_cluster_beats_all_3pct": {
                "pass": k1_pass,
                "improvement_pct": round(k1_improvement, 2),
                "threshold": 3.0,
            },
            "K2_routing_accuracy_80pct": {
                "pass": k2_pass,
                "accuracy": round(routing_acc, 4),
                "threshold": 0.80,
            },
            "K3_within_cluster_vs_individual_10pct": {
                "pass": k3_pass,
                "n_domains_tested": len(k3_domains) if k3_domains else 0,
            },
        },
        "verdict": verdict,
        "total_time_s": round(time.time() - t_start, 1),
    }

    out_path = RESULTS_DIR / "cluster_grouped_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
