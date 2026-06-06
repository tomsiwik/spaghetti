#!/usr/bin/env python3
"""Experiment: N-gram cache + neural expert mixing.

Kill criteria:
  K1 (id=237): N-gram mixing doesn't improve PPL > 5% -> KILL
  K2 (id=238): N-gram table > 2GB -> KILL

Tests whether mixing a statistical n-gram model with a neural model via
entropy-adaptive weighting improves perplexity. The n-gram model is FREE
(zero training, built from frequency counts of the same data).
"""

import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Memory limits
device = mx.device_info()
total = device["memory_size"]
mx.set_memory_limit(total - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# N-gram Language Model
# ============================================================================

class NgramLM:
    """N-gram language model with stupid backoff and add-delta smoothing."""

    def __init__(self, max_order: int = 5, delta: float = 0.1, backoff: float = 0.4):
        self.max_order = max_order
        self.delta = delta
        self.backoff = backoff
        self.vocab_size = None
        # counts[n] maps (context_tuple) -> {token: count}
        self.counts = {}
        # context_totals[n] maps (context_tuple) -> total_count
        self.context_totals = {}
        self._memory_bytes = 0

    def fit(self, sequences: list[list[int]], vocab_size: int):
        """Build n-gram tables from token sequences."""
        self.vocab_size = vocab_size
        self.counts = {n: defaultdict(lambda: defaultdict(int))
                       for n in range(1, self.max_order + 1)}
        self.context_totals = {n: defaultdict(int)
                               for n in range(1, self.max_order + 1)}

        for seq in sequences:
            for n in range(1, self.max_order + 1):
                for i in range(n - 1, len(seq)):
                    ctx = tuple(seq[max(0, i - n + 1):i]) if n > 1 else ()
                    token = seq[i]
                    self.counts[n][ctx][token] += 1
                    self.context_totals[n][ctx] += 1

        # Compute memory usage
        self._compute_memory()

    def _compute_memory(self):
        """Estimate memory usage of n-gram tables in bytes."""
        total = 0
        for n in range(1, self.max_order + 1):
            for ctx, token_counts in self.counts[n].items():
                # Context tuple: ~56 bytes base + 8 per element
                total += 56 + 8 * len(ctx)
                # Token counts dict: ~232 bytes base + 72 per entry
                total += 232 + 72 * len(token_counts)
            # Context totals dict entry
            total += 72 * len(self.context_totals[n])
        self._memory_bytes = total

    @property
    def memory_mb(self):
        return self._memory_bytes / (1024 * 1024)

    @property
    def memory_gb(self):
        return self._memory_bytes / (1024 ** 3)

    def stats(self):
        """Return table statistics."""
        info = {}
        for n in range(1, self.max_order + 1):
            n_contexts = len(self.counts[n])
            n_entries = sum(len(v) for v in self.counts[n].values())
            info[f"{n}-gram"] = {"contexts": n_contexts, "entries": n_entries}
        info["total_memory_mb"] = self.memory_mb
        return info

    def predict(self, context: list[int]) -> list[float]:
        """Return probability distribution over vocab given context.

        Uses stupid backoff: try highest order first, back off with discount.
        Returns smoothed probabilities (always valid distribution).
        """
        V = self.vocab_size
        scores = [0.0] * V

        # Try each n-gram order from highest to lowest
        for token in range(V):
            scores[token] = self._backoff_score(context, token)

        # Normalize to probabilities
        total = sum(scores)
        if total > 0:
            return [s / total for s in scores]
        # Fallback: uniform
        return [1.0 / V] * V

    def _backoff_score(self, context: list[int], token: int) -> float:
        """Compute stupid backoff score for a single token.

        Accumulates a backoff multiplier (gamma_bo) each time we fall through
        to a lower-order n-gram, per Brants et al. (2007).
        """
        backoff_weight = 1.0
        for n in range(self.max_order, 0, -1):
            if n > 1:
                ctx = tuple(context[-(n-1):]) if len(context) >= n-1 else tuple(context)
                if len(ctx) < n - 1:
                    backoff_weight *= self.backoff
                    continue
            else:
                ctx = ()

            if ctx in self.counts[n]:
                count = self.counts[n][ctx].get(token, 0)
                ctx_total = self.context_totals[n][ctx]
                if count > 0:
                    return backoff_weight * count / ctx_total
                # Context exists but token unseen: back off with penalty
                backoff_weight *= self.backoff
            else:
                backoff_weight *= self.backoff
                continue

        # Unigram fallback with smoothing
        ctx = ()
        if ctx in self.counts[1]:
            count = self.counts[1][ctx].get(token, 0)
            ctx_total = self.context_totals[1][ctx]
            return backoff_weight * (count + self.delta) / (ctx_total + self.delta * self.vocab_size)

        return backoff_weight * self.delta / (self.delta * self.vocab_size)

    def entropy(self, context: list[int]) -> float:
        """Compute Shannon entropy of the n-gram distribution."""
        probs = self.predict(context)
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log(p)
        return h

    def normalized_entropy(self, context: list[int]) -> float:
        """Entropy normalized to [0, 1] range."""
        h = self.entropy(context)
        h_max = math.log(self.vocab_size) if self.vocab_size > 1 else 1.0
        return h / h_max


# ============================================================================
# Entropy-Adaptive Mixer
# ============================================================================

class EntropyAdaptiveMixer:
    """Mix n-gram and neural predictions based on n-gram entropy."""

    def __init__(self, ngram_lm: NgramLM, tau: float = 0.7):
        self.ngram_lm = ngram_lm
        self.tau = tau

    def mix(self, context: list[int], neural_probs: list[float]) -> tuple[list[float], float]:
        """Mix n-gram and neural probabilities.

        Returns (mixed_probs, alpha) where alpha is the n-gram weight used.
        """
        ngram_probs = self.ngram_lm.predict(context)
        h_norm = self.ngram_lm.normalized_entropy(context)

        # Entropy-adaptive weight
        alpha = max(0.0, 1.0 - h_norm / self.tau)

        V = len(neural_probs)
        mixed = [alpha * ngram_probs[i] + (1.0 - alpha) * neural_probs[i]
                 for i in range(V)]
        return mixed, alpha


# ============================================================================
# Phase 1: Load and prepare data
# ============================================================================

def phase_load_data():
    """Load names data and split into domains."""
    from experiments.data import load_names, CharTokenizer, domain_split, train_val_split

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    domains = domain_split(docs, method="quintary")

    domain_data = {}
    for name, domain_docs in domains.items():
        train_docs, val_docs = train_val_split(domain_docs, seed=42)
        # Convert to token sequences for n-gram model
        train_seqs = []
        for doc in train_docs:
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            train_seqs.append(tokens)
        val_seqs = []
        for doc in val_docs:
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            val_seqs.append(tokens)
        domain_data[name] = {
            "train_seqs": train_seqs,
            "val_seqs": val_seqs,
            "train_docs": train_docs,
            "val_docs": val_docs,
        }

    # Also prepare global (all domains combined) data
    all_train_seqs = []
    all_val_seqs = []
    for dd in domain_data.values():
        all_train_seqs.extend(dd["train_seqs"])
        all_val_seqs.extend(dd["val_seqs"])

    print(f"  Loaded {len(docs)} names, vocab_size={tokenizer.vocab_size}")
    print(f"  Domains: {list(domain_data.keys())}")
    for name, dd in domain_data.items():
        print(f"    {name}: {len(dd['train_seqs'])} train, {len(dd['val_seqs'])} val")

    return {
        "tokenizer": tokenizer,
        "domain_data": domain_data,
        "all_train_seqs": all_train_seqs,
        "all_val_seqs": all_val_seqs,
    }


# ============================================================================
# Phase 2: Build n-gram models and evaluate standalone
# ============================================================================

def phase_ngram_standalone(data):
    """Build n-gram models and evaluate them standalone (no neural model)."""
    tokenizer = data["tokenizer"]
    V = tokenizer.vocab_size

    results = {}

    # Test different n-gram orders
    for max_order in [2, 3, 4, 5]:
        print(f"\n--- N-gram order {max_order} ---")

        # Global model
        global_lm = NgramLM(max_order=max_order)
        global_lm.fit(data["all_train_seqs"], V)
        stats = global_lm.stats()
        print(f"  Global model stats: {json.dumps(stats, indent=2)}")

        # Evaluate on all validation sequences
        total_log_prob = 0.0
        total_tokens = 0
        alpha_sum = 0.0
        alpha_count = 0

        for seq in data["all_val_seqs"]:
            for i in range(1, len(seq)):
                ctx = seq[:i]
                probs = global_lm.predict(ctx)
                target = seq[i]
                prob = probs[target]
                if prob > 0:
                    total_log_prob += math.log(prob)
                else:
                    total_log_prob += math.log(1e-10)
                total_tokens += 1

                # Track entropy
                h_norm = global_lm.normalized_entropy(ctx)
                alpha = max(0.0, 1.0 - h_norm / 0.7)
                alpha_sum += alpha
                alpha_count += 1

        avg_loss = -total_log_prob / total_tokens
        ppl = math.exp(avg_loss)
        bpb = avg_loss / math.log(2)
        avg_alpha = alpha_sum / alpha_count if alpha_count > 0 else 0

        print(f"  Global {max_order}-gram: PPL={ppl:.2f}, BPB={bpb:.4f}, "
              f"avg_alpha={avg_alpha:.3f}, memory={global_lm.memory_mb:.2f}MB")

        # Per-domain models
        domain_results = {}
        for dname, dd in data["domain_data"].items():
            domain_lm = NgramLM(max_order=max_order)
            domain_lm.fit(dd["train_seqs"], V)

            total_lp = 0.0
            total_t = 0
            for seq in dd["val_seqs"]:
                for i in range(1, len(seq)):
                    ctx = seq[:i]
                    probs = domain_lm.predict(ctx)
                    target = seq[i]
                    prob = probs[target]
                    total_lp += math.log(max(prob, 1e-10))
                    total_t += 1

            d_loss = -total_lp / total_t
            d_ppl = math.exp(d_loss)
            domain_results[dname] = {
                "ppl": round(d_ppl, 2),
                "loss": round(d_loss, 4),
                "memory_mb": round(domain_lm.memory_mb, 3),
            }
            print(f"    {dname}: PPL={d_ppl:.2f}, memory={domain_lm.memory_mb:.3f}MB")

        results[f"ngram_{max_order}"] = {
            "global_ppl": round(ppl, 2),
            "global_bpb": round(bpb, 4),
            "global_loss": round(avg_loss, 4),
            "avg_alpha_tau07": round(avg_alpha, 3),
            "memory_mb": round(global_lm.memory_mb, 3),
            "stats": stats,
            "per_domain": domain_results,
        }

    return results


# ============================================================================
# Phase 3: Train neural baseline and evaluate
# ============================================================================

def phase_neural_baseline(data):
    """Train a small GPT model and evaluate per-token probabilities."""
    from experiments.data import CharDataset
    from experiments.models.gpt import GPT

    tokenizer = data["tokenizer"]
    V = tokenizer.vocab_size

    # Flatten all train/val docs
    all_train_docs = []
    all_val_docs = []
    for dd in data["domain_data"].values():
        all_train_docs.extend(dd["train_docs"])
        all_val_docs.extend(dd["val_docs"])

    train_ds = CharDataset(all_train_docs, tokenizer, block_size=32)
    val_ds = CharDataset(all_val_docs, tokenizer, block_size=32)

    # Train a small GPT
    model = GPT(vocab_size=V, block_size=32, n_embd=64, n_head=4, n_layer=4)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"\n--- Neural baseline: GPT ({n_params:,} params) ---")

    optimizer = optim.Adam(learning_rate=3e-3)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    rng = random.Random(42)
    steps = 1000
    gc.disable()
    t0 = time.time()
    for step in range(1, steps + 1):
        inputs, targets = train_ds.get_batch(32, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 200 == 0 or step == steps:
            print(f"  step {step}/{steps} | loss={loss.item():.4f}")
    gc.enable()
    gc.collect()
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")

    # Evaluate: get per-token loss on val sequences
    # Standard batch evaluation for PPL
    eval_rng = random.Random(0)
    total_val_loss = 0.0
    n_batches = 50
    for _ in range(n_batches):
        inputs, targets = val_ds.get_batch(32, eval_rng)
        logits = model(inputs)
        B, T, Vl = logits.shape
        val_loss = nn.losses.cross_entropy(
            logits.reshape(B * T, Vl),
            targets.reshape(B * T),
            reduction="mean",
        )
        mx.eval(val_loss)
        total_val_loss += val_loss.item()
        del logits, val_loss

    neural_ppl_batch = math.exp(total_val_loss / n_batches)
    neural_loss_batch = total_val_loss / n_batches
    print(f"  Neural batch val PPL: {neural_ppl_batch:.2f} (loss={neural_loss_batch:.4f})")

    # Per-token evaluation on individual sequences for mixing
    # We need actual per-token neural probabilities to mix with n-gram
    print("  Computing per-token neural probabilities on val sequences...")
    neural_token_probs = {}  # seq_idx -> list of (context, target, neural_prob_dist)

    # Process val sequences one at a time
    # Process ALL val sequences so per-domain eval covers every domain (Fix 4)
    n_eval_seqs = len(data["all_val_seqs"])
    all_neural_log_probs = 0.0
    all_neural_tokens = 0

    for seq_idx in range(n_eval_seqs):
        seq = data["all_val_seqs"][seq_idx]
        seq_probs = []

        # Process the sequence through the model
        if len(seq) < 2:
            continue

        # Feed full sequence, get logits for all positions
        seq_len = min(len(seq), 33)  # block_size + 1
        input_tokens = seq[:seq_len - 1]
        # Pad to block_size
        padded = input_tokens + [0] * (32 - len(input_tokens))
        x = mx.array([padded])
        logits = model(x)
        mx.eval(logits)
        # logits shape: (1, 32, V)
        logits_np = logits.tolist()[0]

        for i in range(min(len(input_tokens), len(seq) - 1)):
            target = seq[i + 1]
            # Softmax
            logit_row = logits_np[i]
            max_logit = max(logit_row)
            exp_logits = [math.exp(l - max_logit) for l in logit_row]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]

            seq_probs.append({
                "context": seq[:i + 1],
                "target": target,
                "neural_probs": probs,
            })

            all_neural_log_probs += math.log(max(probs[target], 1e-10))
            all_neural_tokens += 1

        neural_token_probs[seq_idx] = seq_probs
        del logits

        if (seq_idx + 1) % 100 == 0:
            ppl_so_far = math.exp(-all_neural_log_probs / all_neural_tokens)
            print(f"    Processed {seq_idx + 1}/{n_eval_seqs} seqs, PPL so far: {ppl_so_far:.2f}")

    neural_ppl_seq = math.exp(-all_neural_log_probs / all_neural_tokens)
    neural_loss_seq = -all_neural_log_probs / all_neural_tokens
    print(f"  Neural per-token val PPL: {neural_ppl_seq:.2f} (loss={neural_loss_seq:.4f})")
    print(f"  Total tokens evaluated: {all_neural_tokens}")

    # --- Padding asymmetry analysis (Fix 3) ---
    # Quantify how zero-padding (without attention masking) affects neural PPL
    # by splitting sequences into short (heavy padding) vs long (minimal padding)
    print("\n  --- Padding asymmetry analysis ---")
    short_lp, short_t = 0.0, 0
    long_lp, long_t = 0.0, 0
    # Median sequence length as threshold
    seq_lengths = [len(data["all_val_seqs"][i]) for i in range(n_eval_seqs)]
    median_len = sorted(seq_lengths)[len(seq_lengths) // 2]
    for seq_idx in range(n_eval_seqs):
        seq = data["all_val_seqs"][seq_idx]
        if seq_idx not in neural_token_probs:
            continue
        for tp in neural_token_probs[seq_idx]:
            target = tp["target"]
            prob = max(tp["neural_probs"][target], 1e-10)
            lp = math.log(prob)
            if len(seq) <= median_len:
                short_lp += lp
                short_t += 1
            else:
                long_lp += lp
                long_t += 1
    padding_analysis = {}
    if short_t > 0:
        short_ppl = math.exp(-short_lp / short_t)
        padding_analysis["short_seqs"] = {
            "ppl": round(short_ppl, 2), "n_tokens": short_t,
            "max_len": median_len, "padding_frac": round(1.0 - median_len / 33, 2),
        }
        print(f"    Short seqs (len<={median_len}): PPL={short_ppl:.2f}, n_tokens={short_t}")
    if long_t > 0:
        long_ppl = math.exp(-long_lp / long_t)
        padding_analysis["long_seqs"] = {
            "ppl": round(long_ppl, 2), "n_tokens": long_t,
            "min_len": median_len + 1,
        }
        print(f"    Long seqs (len>{median_len}): PPL={long_ppl:.2f}, n_tokens={long_t}")
    if short_t > 0 and long_t > 0:
        padding_analysis["ppl_gap_pct"] = round(
            (short_ppl - long_ppl) / long_ppl * 100, 2
        )
        print(f"    PPL gap (short-long)/long: {padding_analysis['ppl_gap_pct']:+.2f}%")

    log_memory("post-neural")

    result = {
        "n_params": n_params,
        "train_time_s": round(elapsed, 1),
        "neural_ppl_batch": round(neural_ppl_batch, 2),
        "neural_loss_batch": round(neural_loss_batch, 4),
        "neural_ppl_seq": round(neural_ppl_seq, 2),
        "neural_loss_seq": round(neural_loss_seq, 4),
        "n_eval_seqs": n_eval_seqs,
        "n_eval_tokens": all_neural_tokens,
        "padding_asymmetry": padding_analysis,
    }

    # Clean up model but keep token probs
    cleanup(model, optimizer)

    return result, neural_token_probs


# ============================================================================
# Phase 4: Mix n-gram with neural and evaluate
# ============================================================================

def phase_mixing(data, neural_token_probs, neural_loss_seq):
    """Mix n-gram predictions with neural predictions and measure improvement."""
    tokenizer = data["tokenizer"]
    V = tokenizer.vocab_size

    results = {}

    # Test different n-gram orders and tau values
    for max_order in [2, 3, 4, 5]:
        # Build global n-gram model
        global_lm = NgramLM(max_order=max_order)
        global_lm.fit(data["all_train_seqs"], V)

        for tau in [0.3, 0.5, 0.7, 0.9, 1.0]:
            mixer = EntropyAdaptiveMixer(global_lm, tau=tau)

            total_mixed_log_prob = 0.0
            total_neural_log_prob = 0.0
            total_ngram_log_prob = 0.0
            total_tokens = 0
            alpha_sum = 0.0
            n_ngram_wins = 0
            n_neural_wins = 0

            for seq_idx, seq_probs in neural_token_probs.items():
                for tp in seq_probs:
                    ctx = tp["context"]
                    target = tp["target"]
                    neural_probs = tp["neural_probs"]

                    # Mix
                    mixed_probs, alpha = mixer.mix(ctx, neural_probs)

                    # N-gram standalone
                    ngram_probs = global_lm.predict(ctx)

                    # Log probs
                    total_mixed_log_prob += math.log(max(mixed_probs[target], 1e-10))
                    total_neural_log_prob += math.log(max(neural_probs[target], 1e-10))
                    total_ngram_log_prob += math.log(max(ngram_probs[target], 1e-10))
                    total_tokens += 1
                    alpha_sum += alpha

                    # Who was better?
                    if ngram_probs[target] > neural_probs[target]:
                        n_ngram_wins += 1
                    else:
                        n_neural_wins += 1

            mixed_loss = -total_mixed_log_prob / total_tokens
            neural_loss = -total_neural_log_prob / total_tokens
            ngram_loss = -total_ngram_log_prob / total_tokens
            mixed_ppl = math.exp(mixed_loss)
            neural_ppl = math.exp(neural_loss)
            ngram_ppl = math.exp(ngram_loss)
            avg_alpha = alpha_sum / total_tokens

            improvement = (neural_ppl - mixed_ppl) / neural_ppl * 100
            ngram_vs_neural = (neural_ppl - ngram_ppl) / neural_ppl * 100

            key = f"mix_{max_order}gram_tau{tau}"
            results[key] = {
                "max_order": max_order,
                "tau": tau,
                "mixed_ppl": round(mixed_ppl, 4),
                "neural_ppl": round(neural_ppl, 4),
                "ngram_ppl": round(ngram_ppl, 4),
                "mixed_loss": round(mixed_loss, 6),
                "neural_loss": round(neural_loss, 6),
                "ngram_loss": round(ngram_loss, 6),
                "improvement_pct": round(improvement, 2),
                "ngram_vs_neural_pct": round(ngram_vs_neural, 2),
                "avg_alpha": round(avg_alpha, 4),
                "frac_ngram_wins": round(n_ngram_wins / total_tokens, 4),
                "total_tokens": total_tokens,
            }

            print(f"  {key}: mixed_PPL={mixed_ppl:.4f}, neural_PPL={neural_ppl:.4f}, "
                  f"ngram_PPL={ngram_ppl:.4f}, improvement={improvement:+.2f}%, "
                  f"avg_alpha={avg_alpha:.3f}")

    # Also test fixed-weight mixing (not entropy-adaptive) for comparison
    print("\n--- Fixed-weight mixing (ablation) ---")
    global_lm = NgramLM(max_order=5)
    global_lm.fit(data["all_train_seqs"], V)

    for fixed_alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        total_mixed_log_prob = 0.0
        total_tokens = 0

        for seq_idx, seq_probs in neural_token_probs.items():
            for tp in seq_probs:
                ctx = tp["context"]
                target = tp["target"]
                neural_probs = tp["neural_probs"]
                ngram_probs = global_lm.predict(ctx)

                mixed_prob = fixed_alpha * ngram_probs[target] + (1 - fixed_alpha) * neural_probs[target]
                total_mixed_log_prob += math.log(max(mixed_prob, 1e-10))
                total_tokens += 1

        mixed_loss = -total_mixed_log_prob / total_tokens
        mixed_ppl = math.exp(mixed_loss)
        neural_ppl_ref = math.exp(neural_loss_seq)
        improvement = (neural_ppl_ref - mixed_ppl) / neural_ppl_ref * 100

        key = f"fixed_alpha_{fixed_alpha}"
        results[key] = {
            "method": "fixed_weight",
            "alpha": fixed_alpha,
            "mixed_ppl": round(mixed_ppl, 4),
            "improvement_pct": round(improvement, 2),
        }
        print(f"  fixed alpha={fixed_alpha}: mixed_PPL={mixed_ppl:.4f}, "
              f"improvement={improvement:+.2f}%")

    # Per-domain n-gram mixing (domain-specific tables)
    print("\n--- Per-domain n-gram mixing (5-gram, tau=0.7) ---")
    domain_lms = {}
    total_domain_memory_mb = 0
    for dname, dd in data["domain_data"].items():
        lm = NgramLM(max_order=5)
        lm.fit(dd["train_seqs"], V)
        domain_lms[dname] = lm
        total_domain_memory_mb += lm.memory_mb

    print(f"  Total per-domain table memory: {total_domain_memory_mb:.2f}MB")

    # For per-domain eval, we need to know which domain each val seq belongs to
    # We re-evaluate domain-specifically
    domain_seq_ranges = {}
    idx = 0
    for dname, dd in data["domain_data"].items():
        n_val = len(dd["val_seqs"])
        domain_seq_ranges[dname] = (idx, idx + n_val)
        idx += n_val

    domain_mixing_results = {}
    for dname, (start, end) in domain_seq_ranges.items():
        lm = domain_lms[dname]
        mixer = EntropyAdaptiveMixer(lm, tau=0.7)

        total_mixed_lp = 0.0
        total_neural_lp = 0.0
        total_t = 0

        for seq_idx in range(start, min(end, max(neural_token_probs.keys()) + 1)):
            if seq_idx not in neural_token_probs:
                continue
            for tp in neural_token_probs[seq_idx]:
                ctx = tp["context"]
                target = tp["target"]
                neural_probs = tp["neural_probs"]
                mixed_probs, alpha = mixer.mix(ctx, neural_probs)
                total_mixed_lp += math.log(max(mixed_probs[target], 1e-10))
                total_neural_lp += math.log(max(neural_probs[target], 1e-10))
                total_t += 1

        if total_t > 0:
            d_mixed_ppl = math.exp(-total_mixed_lp / total_t)
            d_neural_ppl = math.exp(-total_neural_lp / total_t)
            d_improvement = (d_neural_ppl - d_mixed_ppl) / d_neural_ppl * 100
            domain_mixing_results[dname] = {
                "mixed_ppl": round(d_mixed_ppl, 4),
                "neural_ppl": round(d_neural_ppl, 4),
                "improvement_pct": round(d_improvement, 2),
                "n_tokens": total_t,
            }
            print(f"    {dname}: mixed={d_mixed_ppl:.4f}, neural={d_neural_ppl:.4f}, "
                  f"improvement={d_improvement:+.2f}%")

    results["per_domain_5gram_tau07"] = domain_mixing_results
    results["per_domain_total_memory_mb"] = round(total_domain_memory_mb, 3)

    return results


# ============================================================================
# Phase 5: Evaluate kill criteria
# ============================================================================

def evaluate_kill_criteria(ngram_results, mixing_results, neural_result):
    """Check K1 and K2."""
    print("\n" + "=" * 60)
    print("KILL CRITERIA EVALUATION")
    print("=" * 60)

    # K1: Does mixing improve PPL > 5%?
    # Find the best mixing configuration
    best_improvement = -999
    best_config = None
    for key, res in mixing_results.items():
        if isinstance(res, dict) and "improvement_pct" in res:
            imp = res["improvement_pct"]
            if imp > best_improvement:
                best_improvement = imp
                best_config = key

    k1_pass = best_improvement > 5.0
    print(f"\nK1: N-gram mixing improvement > 5%")
    print(f"  Best config: {best_config}")
    print(f"  Best improvement: {best_improvement:+.2f}%")
    print(f"  Verdict: {'PASS' if k1_pass else 'FAIL (KILL)'}")

    # K2: N-gram table < 2GB
    max_memory_mb = 0
    for key, res in ngram_results.items():
        if "memory_mb" in res:
            max_memory_mb = max(max_memory_mb, res["memory_mb"])
    max_memory_gb = max_memory_mb / 1024

    k2_pass = max_memory_gb < 2.0
    print(f"\nK2: N-gram table < 2GB")
    print(f"  Largest table: {max_memory_mb:.2f}MB ({max_memory_gb:.4f}GB)")
    print(f"  Verdict: {'PASS' if k2_pass else 'FAIL (KILL)'}")

    return {
        "k1_best_improvement_pct": round(best_improvement, 2),
        "k1_best_config": best_config,
        "k1_pass": k1_pass,
        "k2_max_memory_mb": round(max_memory_mb, 2),
        "k2_max_memory_gb": round(max_memory_gb, 4),
        "k2_pass": k2_pass,
        "overall_pass": k1_pass and k2_pass,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log_memory("start")

    # Phase 1: Load data
    print("\n" + "=" * 60)
    print("PHASE 1: Load and prepare data")
    print("=" * 60)
    data = phase_load_data()

    # Phase 2: N-gram standalone evaluation
    print("\n" + "=" * 60)
    print("PHASE 2: N-gram standalone evaluation")
    print("=" * 60)
    ngram_results = phase_ngram_standalone(data)
    log_memory("after-ngram")

    # Phase 3: Neural baseline
    print("\n" + "=" * 60)
    print("PHASE 3: Train neural baseline")
    print("=" * 60)
    neural_result, neural_token_probs = phase_neural_baseline(data)
    log_memory("after-neural")

    # Phase 4: Mixing
    print("\n" + "=" * 60)
    print("PHASE 4: N-gram + neural mixing")
    print("=" * 60)
    mixing_results = phase_mixing(data, neural_token_probs, neural_result["neural_loss_seq"])
    log_memory("after-mixing")

    # Phase 5: Kill criteria
    kill_results = evaluate_kill_criteria(ngram_results, mixing_results, neural_result)

    # Save results
    total_time = round(time.time() - t0, 1)
    all_results = {
        "experiment": "ngram_expert_mixing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": total_time,
        "ngram_standalone": ngram_results,
        "neural_baseline": neural_result,
        "mixing": mixing_results,
        "kill_criteria": kill_results,
    }

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {total_time}s")

    return all_results


if __name__ == "__main__":
    main()
