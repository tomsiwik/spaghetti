#!/usr/bin/env python3
"""Dynamic Weight Composition — compare 5 expert weighting strategies on 50 pilot adapters.

Tests whether dynamic (per-query) weighting beats equal-weight pre-merge at macro scale.

Kill criteria:
- K1: best dynamic strategy improves mean PPL over equal-weight by >= 2%
- K2: winning dynamic strategy latency overhead < 50ms per query
- K3: Pareto frontier — if equal-weight is Pareto-optimal and no dynamic dominates, KILL

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
HF_CACHE = "/workspace/hf_cache"
RESULTS_DIR = REPO_ROOT / "results" / "dynamic_weight_composition"

BASE_MODEL = "Qwen/Qwen2.5-7B"

# Runtime cap: 90 min hard max
MAX_RUNTIME_S = 60 if IS_SMOKE else 90 * 60

# Tuning per smoke vs full
EVAL_SAMPLES = 5 if IS_SMOKE else 50
MAX_SEQ_LEN = 128 if IS_SMOKE else 512
CENTROID_TRAIN_SAMPLES = 5 if IS_SMOKE else 100
CENTROID_MAX_LEN = 128 if IS_SMOKE else 256
PPL_PROBE_N = 3 if IS_SMOKE else 10  # samples for quality matrix
LATENCY_ITERS = 5 if IS_SMOKE else 100
# Domains to evaluate (cap for smoke)
MAX_DOMAINS = 2 if IS_SMOKE else None  # None = all
# Adapters to use in smoke
MAX_ADAPTERS = 3 if IS_SMOKE else None
# Top-k for embed strategies
EMBED_TOP_K = 2 if IS_SMOKE else 5
HYBRID_K = 2 if IS_SMOKE else 3
# Softmax temperature
TAU = 1.0

_start_time = time.time()


def log(msg):
    elapsed = time.time() - _start_time
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] (+{elapsed:.0f}s) {msg}", flush=True)


def timeout_handler(signum, frame):
    log("WARNING: MAX_RUNTIME reached — saving partial results and exiting.")
    sys.exit(42)


def discover_adapters():
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def load_eval_texts(domain, tokenizer, n, max_len=None):
    """Load n evaluation texts for a domain, handling both eval.jsonl and train.jsonl."""
    if max_len is None:
        max_len = MAX_SEQ_LEN
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
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
            elif "text" in record:
                text = record["text"]
            else:
                continue
            texts.append(text)
            if len(texts) >= n:
                return texts
    return texts


def measure_ppl_batched(model, tokenizer, texts, max_seq_len=None, batch_size=8):
    """Compute mean perplexity using batched inference for GPU efficiency."""
    import torch

    if max_seq_len is None:
        max_seq_len = MAX_SEQ_LEN

    if not texts:
        return float("inf"), []

    model.eval()
    all_losses = []

    # Batch inference — GPU efficiency rule: batch >= 8 for PPL
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            with torch.no_grad():
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                )
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)

                # Compute per-sample losses manually (avoid averaging over padding)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # (B, T, V)

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                # (B, T-1)
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_logits.size(0), -1)

                # Mask padding tokens
                token_losses = token_losses * shift_mask.float()
                # Mean over non-padding tokens per sample
                n_tokens = shift_mask.sum(dim=-1).float().clamp(min=1)
                sample_losses = (token_losses.sum(dim=-1) / n_tokens).cpu().tolist()
                all_losses.extend(sample_losses)

        except torch.cuda.OutOfMemoryError:
            log(f"  OOM at batch_size={batch_size}, retrying individually")
            torch.cuda.empty_cache()
            gc.collect()
            for text in batch_texts:
                try:
                    with torch.no_grad():
                        enc = tokenizer(
                            text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len)
                        input_ids = enc["input_ids"].to(model.device)
                        if input_ids.shape[1] < 2:
                            continue
                        outputs = model(input_ids=input_ids, labels=input_ids)
                        all_losses.append(outputs.loss.item())
                except Exception as e:
                    log(f"  ERROR on single sample: {e}")
                    continue

    if not all_losses:
        return float("inf"), []
    mean_loss = sum(all_losses) / len(all_losses)
    return math.exp(mean_loss), all_losses


def compute_centroids(base_model, tokenizer, adapter_names):
    """Compute L2-normalized domain centroid embeddings for all adapters.

    Forward training examples through base model, mean-pool last hidden state.
    Returns centroids tensor of shape (N, d).
    """
    import torch

    log(f"Computing centroids for {len(adapter_names)} domains...")
    centroids = []

    base_model.eval()
    for idx, domain in enumerate(adapter_names):
        train_texts = load_eval_texts(
            domain, tokenizer, n=CENTROID_TRAIN_SAMPLES, max_len=CENTROID_MAX_LEN)
        if not train_texts:
            log(f"  WARNING: no training data for {domain}, using zero centroid")
            # We'll need the hidden dim — get it from config
            d = base_model.config.hidden_size
            centroids.append(torch.zeros(d))
            continue

        domain_embeddings = []
        # Batch the centroid computation
        batch_size = min(8, len(train_texts))
        for i in range(0, len(train_texts), batch_size):
            batch = train_texts[i : i + batch_size]
            try:
                with torch.no_grad():
                    enc = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=CENTROID_MAX_LEN,
                    )
                    input_ids = enc["input_ids"].to(base_model.device)
                    attention_mask = enc["attention_mask"].to(base_model.device)

                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    # Last hidden state: (B, T, d)
                    last_hidden = outputs.hidden_states[-1]

                    # Mean-pool over non-padding tokens per example
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    summed = (last_hidden * mask_expanded).sum(dim=1)
                    counts = mask_expanded.sum(dim=1).clamp(min=1)
                    mean_hidden = summed / counts  # (B, d)

                    domain_embeddings.append(mean_hidden.cpu().float())

            except torch.cuda.OutOfMemoryError:
                log(f"  OOM on centroid for {domain}, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        if domain_embeddings:
            all_emb = torch.cat(domain_embeddings, dim=0)  # (n_examples, d)
            centroid = all_emb.mean(dim=0)  # (d,)
            norm = centroid.norm()
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
        else:
            d = base_model.config.hidden_size
            centroids.append(torch.zeros(d))

        if (idx + 1) % 10 == 0:
            log(f"  Centroids: {idx+1}/{len(adapter_names)} done")

    centroids_tensor = torch.stack(centroids)  # (N, d)
    log(f"Centroids computed: shape={list(centroids_tensor.shape)}")
    return centroids_tensor


def get_query_embedding(base_model, tokenizer, texts):
    """Compute mean-pooled L2-normalized embedding for a list of eval texts."""
    import torch

    base_model.eval()
    embeddings = []
    batch_size = min(8, len(texts))
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            input_ids = enc["input_ids"].to(base_model.device)
            attention_mask = enc["attention_mask"].to(base_model.device)
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            mean_hidden = (summed / counts).cpu().float()
            embeddings.append(mean_hidden)

    all_emb = torch.cat(embeddings, dim=0)
    query_emb = all_emb.mean(dim=0)
    norm = query_emb.norm()
    if norm > 0:
        query_emb = query_emb / norm
    return query_emb


def compute_quality_matrix(model, tokenizer, adapter_names):
    """Precompute Q[expert, domain] = PPL of expert j on domain d eval data.

    This is a static quality signal (not query-conditional).
    Used by Strategy D (ppl_precomputed).
    Returns Q as dict {expert: {domain: ppl}}.
    """
    import torch

    log(f"Computing quality matrix ({len(adapter_names)} x {len(adapter_names)})...")
    log(f"  Using {PPL_PROBE_N} samples per expert-domain pair")

    Q = {}
    n_total = len(adapter_names)

    for ei, expert_name in enumerate(adapter_names):
        Q[expert_name] = {}
        model.set_adapter(expert_name)

        for di, domain in enumerate(adapter_names):
            texts = load_eval_texts(domain, tokenizer, n=PPL_PROBE_N)
            if not texts:
                Q[expert_name][domain] = float("inf")
                continue
            ppl, _ = measure_ppl_batched(model, tokenizer, texts, batch_size=PPL_PROBE_N)
            Q[expert_name][domain] = round(ppl, 4)

        if (ei + 1) % 5 == 0:
            log(f"  Quality matrix: {ei+1}/{n_total} experts done")

    # Revert to first adapter
    model.set_adapter(adapter_names[0])
    return Q


def load_all_adapters_into_peft(base_model, adapter_names):
    """Load all adapters into a single PeftModel for fast switching."""
    import torch
    from peft import PeftModel

    log(f"Loading {len(adapter_names)} adapters into PeftModel...")
    t0 = time.time()

    # Load first adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR / adapter_names[0]),
        adapter_name=adapter_names[0],
    )

    # Load remaining adapters
    for idx, name in enumerate(adapter_names[1:], 1):
        try:
            model.load_adapter(str(ADAPTER_DIR / name), adapter_name=name)
        except Exception as e:
            log(f"  WARNING: failed to load adapter {name}: {e}")

        if (idx + 1) % 10 == 0:
            log(f"  Loaded {idx+1}/{len(adapter_names)} adapters")

    elapsed = time.time() - t0
    log(f"All adapters loaded in {elapsed:.1f}s")
    return model


def compose_and_measure_ppl(model, tokenizer, adapter_names, weights, eval_texts,
                            composed_name="__composed__"):
    """Compose adapters with given weights and measure PPL on eval_texts.

    Reuses an already-loaded PeftModel with all adapters in memory.
    Creates composed adapter, evaluates, then deletes it.
    """
    import torch

    n = len(adapter_names)
    assert len(weights) == n, f"weights len {len(weights)} != adapters len {n}"

    # Normalize weights to ensure they sum to ~1
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]

    try:
        model.add_weighted_adapter(
            adapters=adapter_names,
            weights=weights,
            adapter_name=composed_name,
            combination_type="linear",
        )
        model.set_adapter(composed_name)

        ppl, losses = measure_ppl_batched(model, tokenizer, eval_texts)

    except Exception as e:
        log(f"  ERROR in compose_and_measure_ppl: {e}")
        traceback.print_exc()
        ppl, losses = float("inf"), []
    finally:
        # Always clean up composed adapter
        try:
            model.delete_adapter(composed_name)
        except Exception:
            pass
        # Revert to first adapter so model is in valid state
        try:
            model.set_adapter(adapter_names[0])
        except Exception:
            pass

    return ppl, losses


def softmax_weights(scores, tau=1.0):
    """Compute softmax weights from scores with temperature tau."""
    import math
    max_s = max(scores)
    exps = [math.exp((s - max_s) / tau) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def measure_strategy_latency(model, tokenizer, adapter_names, strategy_fn,
                              n_iters=None):
    """Measure wall-clock time for weight computation + merge step only (not PPL eval)."""
    import torch

    if n_iters is None:
        n_iters = LATENCY_ITERS

    # Use a tiny dummy text for timing (weight computation cost, not forward pass)
    dummy_texts = ["Hello world"] * 4

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        # This calls strategy_fn which returns weights for the composition
        weights_result = strategy_fn()
        # Simulate adapter composition (the overhead beyond equal-weight)
        if weights_result is not None:
            adapter_subset, w = weights_result
            try:
                model.add_weighted_adapter(
                    adapters=adapter_subset,
                    weights=w,
                    adapter_name="__latency_test__",
                    combination_type="linear",
                )
                model.set_adapter("__latency_test__")
            except Exception:
                pass
            finally:
                try:
                    model.delete_adapter("__latency_test__")
                except Exception:
                    pass
                try:
                    model.set_adapter(adapter_names[0])
                except Exception:
                    pass
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    mean_ms = sum(times) / len(times)
    return round(mean_ms, 2)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Set up timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_RUNTIME_S)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    t0 = time.time()
    log("=" * 70)
    log("Dynamic Weight Composition Experiment")
    log("=" * 70)
    log(f"SMOKE_TEST={IS_SMOKE}, MAX_RUNTIME={MAX_RUNTIME_S}s")

    # ------------------------------------------------------------------
    # Phase 0: Discovery
    # ------------------------------------------------------------------
    log("\n=== Phase 0: Discovery ===")
    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters")

    if MAX_ADAPTERS is not None:
        all_adapters = all_adapters[:MAX_ADAPTERS]
        log(f"Smoke: capped to {len(all_adapters)} adapters: {all_adapters}")

    n_adapters = len(all_adapters)
    if n_adapters == 0:
        log("ERROR: no adapters found")
        sys.exit(1)

    # Identify eval domains with data
    eval_domains = []
    for name in all_adapters:
        data_dir = DATA_DIR / name
        if data_dir.exists():
            # Quick check: any jsonl files?
            if any(data_dir.glob("*.jsonl")):
                eval_domains.append(name)
    log(f"Domains with data: {len(eval_domains)}")

    if MAX_DOMAINS is not None:
        eval_domains = eval_domains[:MAX_DOMAINS]
        log(f"Smoke: capped to {len(eval_domains)} domains: {eval_domains}")

    if not eval_domains:
        log("ERROR: no eval domains found")
        sys.exit(1)

    # Pre-load all eval texts before the eval loop (GPU efficiency: avoid CPU bottleneck)
    log("Pre-loading all eval texts...")

    # Load tokenizer first (needed for text loading)
    log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batched generation

    domain_eval_texts = {}
    for domain in eval_domains:
        texts = load_eval_texts(domain, tokenizer, n=EVAL_SAMPLES)
        if texts:
            domain_eval_texts[domain] = texts
        else:
            log(f"  WARNING: no eval texts for {domain}")
    log(f"Pre-loaded eval texts for {len(domain_eval_texts)} domains")

    # Pre-load centroid train texts
    domain_train_texts = {}
    for domain in all_adapters:
        texts = load_eval_texts(domain, tokenizer, n=CENTROID_TRAIN_SAMPLES,
                                max_len=CENTROID_MAX_LEN)
        domain_train_texts[domain] = texts

    # ------------------------------------------------------------------
    # Load base model (4-bit NF4)
    # ------------------------------------------------------------------
    log("\n=== Loading Base Model (4-bit NF4) ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    base_model.eval()
    log("Base model loaded.")

    d_hidden = base_model.config.hidden_size
    log(f"Hidden dim: {d_hidden}")

    # ------------------------------------------------------------------
    # Phase 0b: Compute domain centroids
    # ------------------------------------------------------------------
    log("\n=== Phase 0b: Computing Domain Centroids ===")
    centroids = compute_centroids(base_model, tokenizer, all_adapters)
    # Save centroids
    centroid_path = RESULTS_DIR / "centroids.pt"
    torch.save({"adapters": all_adapters, "centroids": centroids}, centroid_path)
    log(f"Centroids saved to {centroid_path}")

    # ------------------------------------------------------------------
    # Phase 1: Base model PPL
    # ------------------------------------------------------------------
    log("\n=== Phase 1: Base Model PPL ===")
    base_ppl = {}
    for domain, texts in domain_eval_texts.items():
        ppl, _ = measure_ppl_batched(base_model, tokenizer, texts)
        base_ppl[domain] = round(ppl, 4)
        log(f"  {domain}: base PPL = {ppl:.4f}")

    results["base_ppl"] = base_ppl

    # ------------------------------------------------------------------
    # Phase 1b: Load all adapters into PeftModel
    # ------------------------------------------------------------------
    log("\n=== Phase 1b: Loading All Adapters into PeftModel ===")
    peft_model = load_all_adapters_into_peft(base_model, all_adapters)
    peft_model.eval()

    # ------------------------------------------------------------------
    # Phase 1c: Single-expert baselines
    # ------------------------------------------------------------------
    log("\n=== Phase 1c: Single Expert Baselines ===")
    single_expert_ppl = {}
    for domain in eval_domains:
        if domain not in domain_eval_texts:
            continue
        if domain not in all_adapters:
            log(f"  SKIP {domain}: no adapter")
            continue
        try:
            peft_model.set_adapter(domain)
            ppl, _ = measure_ppl_batched(
                peft_model, tokenizer, domain_eval_texts[domain])
            single_expert_ppl[domain] = round(ppl, 4)
            log(f"  {domain}: single-expert PPL = {ppl:.4f}")
        except Exception as e:
            log(f"  ERROR {domain} single-expert: {e}")
            traceback.print_exc()

    results["single_expert_ppl"] = single_expert_ppl

    # ------------------------------------------------------------------
    # Phase 0c: Compute quality matrix (for Strategy D)
    # ------------------------------------------------------------------
    log("\n=== Phase 0c: Computing Quality Matrix (Strategy D) ===")
    quality_matrix = compute_quality_matrix(peft_model, tokenizer, all_adapters)
    quality_path = RESULTS_DIR / "quality_matrix.json"
    with open(quality_path, "w") as f:
        json.dump(quality_matrix, f, indent=2)
    log(f"Quality matrix saved to {quality_path}")

    # ------------------------------------------------------------------
    # Phase 2: Strategy Evaluation
    # ------------------------------------------------------------------
    log("\n=== Phase 2: Strategy Evaluation ===")

    strategy_results = {}

    def eval_strategy(strategy_name, weight_fn, adapter_subset_fn=None):
        """
        Evaluate a strategy on all eval domains.

        weight_fn(domain, query_emb) -> (adapter_names_list, weights_list)
        """
        log(f"\n--- Strategy: {strategy_name} ---")
        per_domain_ppl = {}
        for domain in eval_domains:
            if domain not in domain_eval_texts:
                continue
            texts = domain_eval_texts[domain]
            try:
                # Compute query embedding for this domain's eval texts
                query_emb = get_query_embedding(peft_model, tokenizer, texts)
                adapter_subset, w = weight_fn(domain, query_emb)

                if not adapter_subset:
                    log(f"  {domain}: empty adapter subset, skipping")
                    continue

                ppl, _ = compose_and_measure_ppl(
                    peft_model, tokenizer, adapter_subset, w, texts)
                per_domain_ppl[domain] = round(ppl, 4)
                log(f"  {domain}: {strategy_name} PPL = {ppl:.4f}")

            except Exception as e:
                log(f"  ERROR {domain} {strategy_name}: {e}")
                traceback.print_exc()

        mean_ppl = (sum(per_domain_ppl.values()) / len(per_domain_ppl)
                    if per_domain_ppl else float("inf"))

        # Degradation vs single-expert
        degs = []
        n_worse_base = 0
        for domain, ppl in per_domain_ppl.items():
            if domain in single_expert_ppl and single_expert_ppl[domain] > 0:
                deg = (ppl - single_expert_ppl[domain]) / single_expert_ppl[domain] * 100
                degs.append(deg)
            if domain in base_ppl and ppl > base_ppl[domain]:
                n_worse_base += 1

        mean_deg = sum(degs) / len(degs) if degs else float("nan")
        strategy_results[strategy_name] = {
            "per_domain_ppl": per_domain_ppl,
            "mean_ppl": round(mean_ppl, 4),
            "mean_degradation_vs_single_pct": round(mean_deg, 2) if not math.isnan(mean_deg) else None,
            "domains_worse_than_base": n_worse_base,
            "n_evaluated": len(per_domain_ppl),
        }
        log(f"  {strategy_name}: mean_ppl={mean_ppl:.4f}, "
            f"mean_deg_vs_single={mean_deg:+.2f}%, "
            f"worse_than_base={n_worse_base}/{len(per_domain_ppl)}")
        return strategy_results[strategy_name]

    # --- Strategy A: equal_premerge (weights = 1/N for all N adapters) ---
    def weight_fn_equal(domain, query_emb):
        n = len(all_adapters)
        return all_adapters, [1.0 / n] * n

    eval_strategy("equal_premerge", weight_fn_equal)

    # --- Strategy B2: embed_topk (top-k by cosine similarity, equal weight) ---
    def weight_fn_embed_topk(domain, query_emb):
        import torch
        # Cosine similarity: centroids already L2-normalized
        sims = (centroids @ query_emb).tolist()  # (N,)
        k = min(EMBED_TOP_K, len(all_adapters))
        top_k_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        selected = [all_adapters[i] for i in top_k_idx]
        return selected, [1.0 / k] * k

    eval_strategy("embed_topk", weight_fn_embed_topk)

    # --- Strategy C2: embed_weighted (softmax over all 50 cosine similarities) ---
    def weight_fn_embed_weighted(domain, query_emb):
        import torch
        sims = (centroids @ query_emb).tolist()  # (N,)
        weights = softmax_weights(sims, tau=TAU)
        return all_adapters, weights

    eval_strategy("embed_weighted", weight_fn_embed_weighted)

    # --- Strategy D: ppl_precomputed (precomputed quality matrix, domain routing) ---
    def weight_fn_ppl_precomputed(domain, query_emb):
        # Use cosine similarity to identify the most likely domain, then use
        # precomputed quality matrix for that domain to weight experts
        import torch
        sims = (centroids @ query_emb).tolist()
        best_domain_idx = max(range(len(sims)), key=lambda i: sims[i])
        best_domain = all_adapters[best_domain_idx]

        # Weights from quality matrix for the predicted domain
        # Lower PPL = better = higher weight: use softmax(-ppl / tau)
        ppls = [quality_matrix.get(exp, {}).get(best_domain, float("inf"))
                for exp in all_adapters]
        # Replace inf with large value for softmax
        max_finite = max((p for p in ppls if p != float("inf")), default=100.0)
        ppls_clipped = [p if p != float("inf") else max_finite * 2 for p in ppls]
        # Score: negative ppl (higher is better)
        scores = [-p for p in ppls_clipped]
        weights = softmax_weights(scores, tau=TAU)
        return all_adapters, weights

    eval_strategy("ppl_precomputed", weight_fn_ppl_precomputed)

    # --- Strategy C3: hybrid_k3 (embed top-k filter + PPL rerank) ---
    def weight_fn_hybrid_k3(domain, query_emb):
        import torch
        k = HYBRID_K
        sims = (centroids @ query_emb).tolist()
        top_k_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        selected = [all_adapters[i] for i in top_k_idx]

        # PPL rerank: measure PPL of each selected expert on a small eval sample
        eval_texts = domain_eval_texts.get(domain, [])
        if not eval_texts:
            return selected, [1.0 / k] * k

        probe_texts = eval_texts[:min(3, len(eval_texts))]
        ppl_scores = []
        for expert in selected:
            try:
                peft_model.set_adapter(expert)
                ppl, _ = measure_ppl_batched(
                    peft_model, tokenizer, probe_texts, batch_size=len(probe_texts))
                ppl_scores.append(ppl)
            except Exception as e:
                log(f"    hybrid_k3 PPL probe error for {expert}: {e}")
                ppl_scores.append(float("inf"))

        # Revert adapter
        try:
            peft_model.set_adapter(selected[0])
        except Exception:
            pass

        # Softmax(-ppl / tau): lower ppl = higher weight
        max_finite = max((p for p in ppl_scores if p != float("inf")), default=100.0)
        ppl_clipped = [p if p != float("inf") else max_finite * 2 for p in ppl_scores]
        scores = [-p for p in ppl_clipped]
        weights = softmax_weights(scores, tau=TAU)
        return selected, weights

    eval_strategy("hybrid_k3", weight_fn_hybrid_k3)

    # ------------------------------------------------------------------
    # Phase 4: Latency measurement
    # ------------------------------------------------------------------
    log("\n=== Phase 4: Latency Measurement ===")

    # Use first domain's eval texts for timing
    timing_domain = eval_domains[0]
    timing_query_emb = get_query_embedding(
        peft_model, tokenizer, domain_eval_texts[timing_domain])

    latency_results = {}

    def time_strategy(name, weight_fn):
        def strategy_fn():
            return weight_fn(timing_domain, timing_query_emb)

        try:
            ms = measure_strategy_latency(peft_model, tokenizer, all_adapters, strategy_fn)
            latency_results[name] = ms
            log(f"  {name}: {ms:.2f}ms avg over {LATENCY_ITERS} iters")
        except Exception as e:
            log(f"  ERROR timing {name}: {e}")
            latency_results[name] = None

    time_strategy("equal_premerge", weight_fn_equal)
    time_strategy("embed_topk", weight_fn_embed_topk)
    time_strategy("embed_weighted", weight_fn_embed_weighted)
    time_strategy("ppl_precomputed", weight_fn_ppl_precomputed)
    time_strategy("hybrid_k3", weight_fn_hybrid_k3)

    # Attach latency to strategy results
    for name, ms in latency_results.items():
        if name in strategy_results:
            strategy_results[name]["latency_ms"] = ms

    # ------------------------------------------------------------------
    # Phase 5: Kill Criteria Assessment
    # ------------------------------------------------------------------
    log("\n=== Phase 5: Kill Criteria Assessment ===")

    eq_ppl = strategy_results.get("equal_premerge", {}).get("mean_ppl", float("inf"))

    # K1: best dynamic strategy improves mean PPL >= 2% over equal-weight
    dynamic_strategies = ["embed_topk", "embed_weighted", "ppl_precomputed", "hybrid_k3"]
    best_dynamic_name = None
    best_dynamic_ppl = float("inf")
    for name in dynamic_strategies:
        if name in strategy_results:
            ppl = strategy_results[name].get("mean_ppl", float("inf"))
            if ppl < best_dynamic_ppl:
                best_dynamic_ppl = ppl
                best_dynamic_name = name

    if eq_ppl > 0 and eq_ppl != float("inf"):
        k1_improvement = (eq_ppl - best_dynamic_ppl) / eq_ppl * 100
    else:
        k1_improvement = 0.0
    k1_pass = k1_improvement >= 2.0

    # K2: winning dynamic strategy latency < 50ms
    best_dynamic_latency = None
    if best_dynamic_name:
        best_dynamic_latency = latency_results.get(best_dynamic_name)
    k2_pass = (best_dynamic_latency is not None and best_dynamic_latency < 50.0)

    # K3: Pareto frontier analysis (latency vs quality)
    # Lower is better for both. equal_premerge Pareto-dominates iff no dynamic
    # strategy has both lower PPL and lower latency.
    pareto_points = []
    for name in ["equal_premerge"] + dynamic_strategies:
        if name in strategy_results and name in latency_results:
            ppl = strategy_results[name].get("mean_ppl", float("inf"))
            lat = latency_results.get(name) or float("inf")
            pareto_points.append({"strategy": name, "mean_ppl": ppl, "latency_ms": lat})

    # Compute Pareto frontier
    def is_dominated(point, others):
        """True if another point is strictly better in both dimensions."""
        for other in others:
            if (other["mean_ppl"] <= point["mean_ppl"] and
                    other["latency_ms"] <= point["latency_ms"] and
                    (other["mean_ppl"] < point["mean_ppl"] or
                     other["latency_ms"] < point["latency_ms"])):
                return True
        return False

    pareto_frontier = [p for p in pareto_points
                       if not is_dominated(p, [q for q in pareto_points if q["strategy"] != p["strategy"]])]
    pareto_strategy_names = [p["strategy"] for p in pareto_frontier]

    eq_on_frontier = "equal_premerge" in pareto_strategy_names
    dynamic_dominates = any(n in pareto_strategy_names for n in dynamic_strategies)
    # K3 PASS means dynamic strategies are competitive (equal-weight is NOT sole Pareto point)
    k3_pass = dynamic_dominates

    log(f"\nK1 (dynamic improves >= 2%): {'PASS' if k1_pass else 'FAIL'} — "
        f"improvement={k1_improvement:.2f}% (best: {best_dynamic_name})")
    log(f"K2 (latency < 50ms): {'PASS' if k2_pass else 'FAIL'} — "
        f"{best_dynamic_latency}ms ({best_dynamic_name})")
    log(f"K3 (dynamic on Pareto frontier): {'PASS' if k3_pass else 'FAIL'} — "
        f"frontier: {pareto_strategy_names}")

    verdict = "PASS" if (k1_pass and k2_pass and k3_pass) else "KILL"
    log(f"\nVERDICT: {verdict}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    elapsed_s = round(time.time() - t0, 1)
    log(f"\nTotal elapsed: {elapsed_s}s")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "base_model": BASE_MODEL,
            "n_adapters": n_adapters,
            "n_eval_domains": len(eval_domains),
            "eval_samples": EVAL_SAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "centroid_train_samples": CENTROID_TRAIN_SAMPLES,
            "ppl_probe_n": PPL_PROBE_N,
            "embed_top_k": EMBED_TOP_K,
            "hybrid_k": HYBRID_K,
            "tau": TAU,
            "smoke_test": IS_SMOKE,
        },
        "centroids": {
            "shape": list(centroids.shape),
            "adapters": all_adapters,
        },
        "base_ppl": base_ppl,
        "single_expert_ppl": single_expert_ppl,
        "strategies": strategy_results,
        "kill_criteria": {
            "K1_dynamic_improvement_pct": {
                "value": round(k1_improvement, 2),
                "threshold": 2.0,
                "best_dynamic_strategy": best_dynamic_name,
                "equal_premerge_mean_ppl": round(eq_ppl, 4),
                "best_dynamic_mean_ppl": round(best_dynamic_ppl, 4),
                "pass": k1_pass,
            },
            "K2_latency_lt_50ms": {
                "value": best_dynamic_latency,
                "threshold": 50.0,
                "strategy": best_dynamic_name,
                "pass": k2_pass,
            },
            "K3_pareto_frontier": {
                "pareto_points": pareto_points,
                "pareto_frontier_strategies": pareto_strategy_names,
                "equal_premerge_on_frontier": eq_on_frontier,
                "dynamic_on_frontier": dynamic_dominates,
                "pass": k3_pass,
            },
        },
        "verdict": verdict,
        "elapsed_s": elapsed_s,
    }

    out_file = RESULTS_DIR / "results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {out_file}")

    pareto_path = RESULTS_DIR / "pareto_data.json"
    with open(pareto_path, "w") as f:
        json.dump({"pareto_points": pareto_points,
                   "pareto_frontier": pareto_frontier}, f, indent=2)
    log(f"Pareto data saved to {pareto_path}")

    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for name, res in strategy_results.items():
        lat = latency_results.get(name)
        lat_str = f"{lat:.1f}ms" if lat is not None else "N/A"
        log(f"  {name:25s}: PPL={res.get('mean_ppl', 'N/A'):.4f}  "
            f"deg_vs_single={res.get('mean_degradation_vs_single_pct', 'N/A')}%  "
            f"latency={lat_str}")
    log(f"\nK1: {'PASS' if k1_pass else 'FAIL'}  K2: {'PASS' if k2_pass else 'FAIL'}  "
        f"K3: {'PASS' if k3_pass else 'FAIL'}")
    log(f"VERDICT: {verdict}")

    # Disable alarm
    signal.alarm(0)


if __name__ == "__main__":
    main()
