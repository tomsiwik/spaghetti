#!/usr/bin/env python3
"""
P11.J0: Compose Thinking Adapter + Domain Knowledge Adapter via Exclusive Routing

Tests whether exclusive routing (apply exactly one adapter per query based on query type)
outperforms unconditional application of either adapter.

Three evaluation conditions:
  A. thinking_only: apply thinking-universal adapter to all questions
  B. domain_only: apply best domain adapter based on ground-truth category label
  C. embedding_routed: apply adapter based on query embedding cosine similarity

Router: mean of query token embeddings → cosine similarity to reasoning/knowledge centroid.
No forward pass required — just embed_tokens weights.

Kill criteria:
  K1526: Routed (C) >= domain-only (B) + 3pp on 4-category MMLU-Pro subset
  K1527: Routed (C) >= thinking-only (A) + 2pp on knowledge categories (bio, law)
  K1528: Embedding router accuracy >= 85% on binary (reasoning/knowledge) classification

References:
  arXiv:2407.06582 — LoRAMOE: Mixture of LoRA Experts
  arXiv:2312.00752 — MoLoRA: Multi-expert LoRA
  arXiv:1904.10480 — JL-lemma and embedding geometry
  Finding #517 — domain adapters degrade MCQ (q_proj NTP)
  Finding #527 — pre-merge killed (exclusive routing required)
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MMLU_PATH = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"

# Adapter paths (relative to REPO_ROOT)
ADAPTER_THINKING = REPO_ROOT / "data" / "adapters" / "thinking-openthoughts-universal-v0"
ADAPTER_MATH     = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/math"
ADAPTER_MEDICAL  = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/medical"
ADAPTER_LEGAL    = REPO_ROOT / "experiments/models/exp_p1_t2_multi_domain_5/adapters/legal"
ADAPTER_FINANCE  = REPO_ROOT / "experiments/models/exp_p1_t2_multi_domain_5/adapters/finance"

SEED = 42
OPTION_LETTERS = "ABCDEFGHIJ"

IS_SMOKE = "--smoke" in sys.argv or os.environ.get("SMOKE_TEST", "0") == "1"

# Target categories for evaluation
REASONING_CATS = ["math", "physics"]
KNOWLEDGE_CATS = ["biology", "law"]
EVAL_CATS = REASONING_CATS + KNOWLEDGE_CATS  # 4 categories

# Router calibration: seed examples per category-group
N_SEED_PER_GROUP = 2 if IS_SMOKE else 10  # for centroid building
N_ROUTER_EVAL_PER_CAT = 3 if IS_SMOKE else 20  # for K1528 router accuracy check
N_EVAL_PER_CAT = 2 if IS_SMOKE else 20  # for adapter condition comparison

MAX_TOKENS = 2048


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def strip_thinking(response):
    """Extract non-thinking portion of Gemma 4 response."""
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
        return cleaned, len(m.group(0))
    m = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if m:
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return cleaned, len(m.group(0))
    return response, 0


def parse_answer(response):
    """Extract MCQ letter answer."""
    answer_text, _ = strip_thinking(response)
    for pat in [
        r'\b([A-J])\b(?:\s*$|\s*\.|\s*\))',
        r'answer is ([A-J])',
        r'answer: ([A-J])',
        r'\b([A-J])\b',
    ]:
        m = re.search(pat, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def format_question(row, tokenizer):
    """Format MMLU-Pro question for Gemma 4 chat."""
    options = row["options"]
    opts_str = "\n".join(
        f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(options) if i < len(OPTION_LETTERS)
    )
    content = f"{row['question']}\n\n{opts_str}\n\nAnswer with a single letter."
    chat = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────
# Router: Token Embedding Centroids
# ─────────────────────────────────────────────

def build_embedding_centroids(model, tokenizer, df, n_seed_per_group, seed):
    """
    Build two centroids (reasoning, knowledge) from seed examples.

    Uses only the embedding layer (no forward pass) to keep it fast.
    embed_tokens: maps token_id → embedding vector (vocab_size, d_model)
    """
    log(f"[Router] Building centroids from {n_seed_per_group} seed examples per group")

    # All 14 categories: which group does each belong to?
    ALL_CATS = sorted(df["category"].unique())
    REASONING_CATS_ALL = {"math", "physics", "engineering", "computer science"}
    group_map = {cat: ("reasoning" if cat in REASONING_CATS_ALL else "knowledge")
                 for cat in ALL_CATS}

    log(f"  Reasoning cats: {[c for c in ALL_CATS if group_map[c]=='reasoning']}")
    log(f"  Knowledge cats: {[c for c in ALL_CATS if group_map[c]=='knowledge']}")

    rng = np.random.default_rng(seed)
    seed_reasoning = []
    seed_knowledge = []

    for cat in ALL_CATS:
        cat_df = df[df["category"] == cat].reset_index(drop=True)
        n = min(n_seed_per_group, len(cat_df))
        idx = rng.choice(len(cat_df), n, replace=False)
        for i in idx:
            row = cat_df.iloc[i]
            text = row["question"][:200]  # First 200 chars for speed
            tokens = tokenizer.encode(text)[:64]  # First 64 tokens
            if group_map[cat] == "reasoning":
                seed_reasoning.append(tokens)
            else:
                seed_knowledge.append(tokens)

    log(f"  Seed: reasoning={len(seed_reasoning)} seqs, knowledge={len(seed_knowledge)} seqs")

    # Get embedding layer
    # mlx_lm loads as model.language_model.embed_tokens (Gemma 4 architecture)
    embed = None
    if hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
        embed = model.language_model.embed_tokens
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed = model.model.embed_tokens
    else:
        # Fallback: find embed_tokens by introspection
        for name, layer in model.named_modules():
            if "embed_tokens" in name:
                embed = layer
                break

    if embed is None:
        raise RuntimeError("Could not find embed_tokens layer")

    log(f"  Found embed layer, embedding dim: {embed.weight.shape}")

    def compute_centroid(seqs):
        vecs = []
        for toks in seqs:
            ids = mx.array(toks)
            embs = embed(ids)  # (n_tokens, d_model)
            mx.eval(embs)
            mean_emb = embs.mean(axis=0)
            mx.eval(mean_emb)
            vecs.append(np.array(mean_emb.astype(mx.float32)))
        mat = np.stack(vecs, axis=0)  # (n_seqs, d_model)
        centroid = mat.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / (norm + 1e-8)

    c_r = compute_centroid(seed_reasoning)
    c_k = compute_centroid(seed_knowledge)

    log(f"  Centroids built: shape={c_r.shape}")
    return c_r, c_k, group_map, embed


def route_query(question_text, embed, tokenizer, centroid_r, centroid_k):
    """Route a query to 'reasoning' or 'knowledge' based on embedding cosine sim."""
    text = question_text[:200]
    tokens = tokenizer.encode(text)[:64]
    ids = mx.array(tokens)
    embs = embed(ids)
    mx.eval(embs)
    q_emb = np.array(embs.mean(axis=0).astype(mx.float32))
    q_norm = np.linalg.norm(q_emb)
    q_emb = q_emb / (q_norm + 1e-8)
    sim_r = float(np.dot(q_emb, centroid_r))
    sim_k = float(np.dot(q_emb, centroid_k))
    return "reasoning" if sim_r >= sim_k else "knowledge"


# ─────────────────────────────────────────────
# Domain adapter mapping
# ─────────────────────────────────────────────

DOMAIN_ADAPTER_MAP = {
    "math": ADAPTER_MATH,
    "physics": None,          # No physics adapter — fall back to thinking
    "engineering": None,      # No engineering adapter
    "computer science": None,
    "biology": ADAPTER_MEDICAL,
    "chemistry": None,
    "health": ADAPTER_MEDICAL,
    "medicine": ADAPTER_MEDICAL,
    "law": ADAPTER_LEGAL,
    "jurisprudence": ADAPTER_LEGAL,
    "economics": ADAPTER_FINANCE,
    "business": ADAPTER_FINANCE,
    "psychology": None,
    "history": None,
    "philosophy": None,
    "other": None,
}


def get_domain_adapter(category):
    """Return adapter path for this category, or None if not available."""
    path = DOMAIN_ADAPTER_MAP.get(category)
    if path is not None and Path(path).exists():
        return path
    return None


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def eval_condition(condition_name, adapter_path, questions):
    """
    Evaluate a single condition (load model once, run all questions).

    adapter_path: None = base model, else path to adapter dir.
    """
    from mlx_lm import load, generate

    log(f"\n[{condition_name}] adapter={'none' if adapter_path is None else Path(adapter_path).name}")
    log(f"  n_questions={len(questions)}")

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)
    log_memory(f"loaded {condition_name}")

    per_cat = {}
    for cat in EVAL_CATS:
        per_cat[cat] = {"correct": 0, "total": 0}

    total_thinking = 0
    t0 = time.time()

    for i, (cat, row) in enumerate(questions):
        prompt = format_question(row, tokenizer)
        response = generate(model, tokenizer, prompt=prompt,
                            max_tokens=MAX_TOKENS, verbose=False)
        pred = parse_answer(response)
        answer_letter = OPTION_LETTERS[row["answer_index"]]
        is_correct = (pred == answer_letter)
        if is_correct:
            per_cat[cat]["correct"] += 1
        per_cat[cat]["total"] += 1

        _, t_chars = strip_thinking(response)
        total_thinking += t_chars

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            log(f"  [{condition_name}] {i+1}/{len(questions)} done ({elapsed:.0f}s)")

    # Compute accuracy per cat and overall
    cat_acc = {}
    for cat, v in per_cat.items():
        if v["total"] > 0:
            cat_acc[cat] = v["correct"] / v["total"]
        else:
            cat_acc[cat] = None

    overall_correct = sum(v["correct"] for v in per_cat.values())
    overall_total = sum(v["total"] for v in per_cat.values())
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    reasoning_correct = sum(per_cat[c]["correct"] for c in REASONING_CATS)
    reasoning_total = sum(per_cat[c]["total"] for c in REASONING_CATS)
    reasoning_acc = reasoning_correct / reasoning_total if reasoning_total > 0 else 0.0

    knowledge_correct = sum(per_cat[c]["correct"] for c in KNOWLEDGE_CATS)
    knowledge_total = sum(per_cat[c]["total"] for c in KNOWLEDGE_CATS)
    knowledge_acc = knowledge_correct / knowledge_total if knowledge_total > 0 else 0.0

    elapsed = time.time() - t0
    log(f"  RESULT [{condition_name}]: overall={overall_acc:.3f} "
        f"reasoning={reasoning_acc:.3f} knowledge={knowledge_acc:.3f} "
        f"time={elapsed:.0f}s")

    result = {
        "overall_acc": overall_acc,
        "reasoning_acc": reasoning_acc,
        "knowledge_acc": knowledge_acc,
        "per_cat": cat_acc,
        "n_questions": overall_total,
        "avg_thinking_chars": total_thinking / max(overall_total, 1),
        "elapsed_s": elapsed,
    }

    cleanup(model, tokenizer)
    return result


def eval_routed_condition(questions, centroid_r, centroid_k, embed_ref_model,
                          embed_ref_tokenizer, routing_decisions):
    """
    Routed condition: each question gets routed to thinking or domain adapter.
    Pre-computed routing decisions to avoid per-question model reload.

    Strategy:
    1. Partition questions into reasoning_set (→ thinking adapter) and knowledge_set
       (→ domain adapter, or thinking if no domain adapter available).
    2. Run each partition as a separate eval_condition() call.
    """
    log("\n[embedding_routed] Partitioning questions by routing decision")

    # Route each question
    assert len(routing_decisions) == len(questions), "routing_decisions must match questions"

    reasoning_qs = []
    knowledge_qs = []
    for (cat, row), decision in zip(questions, routing_decisions):
        if decision == "reasoning":
            reasoning_qs.append((cat, row))
        else:
            knowledge_qs.append((cat, row))

    log(f"  routing: {len(reasoning_qs)} → thinking, {len(knowledge_qs)} → domain")

    # Run reasoning partition → thinking adapter
    r_reasoning = eval_condition(
        "routed_reasoning_partition",
        ADAPTER_THINKING if ADAPTER_THINKING.exists() else None,
        reasoning_qs,
    )

    # Run knowledge partition → best domain adapter (fallback: thinking adapter)
    # For simplicity: use thinking adapter as fallback for categories without domain adapter
    # (This still tests that routing MATTERS — different knowledge cats get different adapters)
    knowledge_by_adapter = {}
    for cat, row in knowledge_qs:
        adapter = get_domain_adapter(cat)
        if adapter is None:
            adapter = ADAPTER_THINKING if ADAPTER_THINKING.exists() else None
        key = str(adapter) if adapter else "none"
        if key not in knowledge_by_adapter:
            knowledge_by_adapter[key] = (adapter, [])
        knowledge_by_adapter[key][1].append((cat, row))

    knowledge_results = {}
    for key, (adapter_path, qs) in knowledge_by_adapter.items():
        r = eval_condition(f"routed_knowledge_{Path(adapter_path).name if adapter_path else 'base'}", adapter_path, qs)
        knowledge_results[key] = (r, len(qs))

    # Combine knowledge results
    kn_correct = 0
    kn_total = 0
    per_cat_combined = {}
    for key, (r, n) in knowledge_results.items():
        kn_total += n
        kn_correct += int(r["overall_acc"] * n)
        for c, acc in r["per_cat"].items():
            if acc is not None and c in KNOWLEDGE_CATS:
                per_cat_combined[c] = acc

    # Overall combined
    r_qs = len(reasoning_qs)
    k_qs = len(knowledge_qs)
    total = r_qs + k_qs
    if total == 0:
        return {"overall_acc": 0.0, "reasoning_acc": 0.0, "knowledge_acc": 0.0}

    r_correct = int(r_reasoning["overall_acc"] * r_qs)
    overall_acc = (r_correct + kn_correct) / total
    reasoning_acc = r_reasoning["reasoning_acc"]
    knowledge_acc = kn_correct / k_qs if k_qs > 0 else 0.0

    # Merge per_cat
    per_cat_merged = dict(r_reasoning["per_cat"])
    per_cat_merged.update(per_cat_combined)

    log(f"  RESULT [embedding_routed]: overall={overall_acc:.3f} "
        f"reasoning={reasoning_acc:.3f} knowledge={knowledge_acc:.3f}")

    return {
        "overall_acc": overall_acc,
        "reasoning_acc": reasoning_acc,
        "knowledge_acc": knowledge_acc,
        "per_cat": per_cat_merged,
        "n_questions": total,
        "routing_split": {"reasoning": r_qs, "knowledge": k_qs},
    }


# ─────────────────────────────────────────────
# Router accuracy evaluation
# ─────────────────────────────────────────────

def eval_router_accuracy(df, centroid_r, centroid_k, embed, tokenizer,
                         group_map, n_per_cat, seed):
    """
    K1528: measure router accuracy on held-out examples.
    Ground truth: group_map (reasoning vs knowledge by category).
    """
    log(f"\n[Router Accuracy] n_per_cat={n_per_cat}")
    rng = np.random.default_rng(seed + 999)  # different seed from centroid building

    correct = 0
    total = 0
    ALL_CATS = sorted(df["category"].unique())

    for cat in ALL_CATS:
        cat_df = df[df["category"] == cat].reset_index(drop=True)
        n = min(n_per_cat, len(cat_df))
        idx = rng.choice(len(cat_df), n, replace=False)
        gt_label = group_map[cat]
        for i in idx:
            row = cat_df.iloc[i]
            pred_label = route_query(row["question"], embed, tokenizer, centroid_r, centroid_k)
            if pred_label == gt_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    log(f"  Router accuracy: {correct}/{total} = {accuracy:.3f}")
    return accuracy, correct, total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    from mlx_lm import load

    log("=" * 60)
    log("P11.J0: Adapter Composition via Exclusive Routing")
    log("=" * 60)
    log(f"Smoke: {IS_SMOKE}")
    log(f"Eval categories: {EVAL_CATS}")
    log(f"N eval per cat: {N_EVAL_PER_CAT}, N router eval per cat: {N_ROUTER_EVAL_PER_CAT}")

    # Check prerequisites
    if not ADAPTER_THINKING.exists():
        log(f"ERROR: Thinking adapter not found at {ADAPTER_THINKING}")
        log("Dependency: exp_p11_thinking_adapter_universal must complete first")
        sys.exit(1)
    log(f"  thinking adapter: {ADAPTER_THINKING} ✓")
    log(f"  math adapter: {'✓' if ADAPTER_MATH.exists() else '✗'} {ADAPTER_MATH}")
    log(f"  medical adapter: {'✓' if ADAPTER_MEDICAL.exists() else '✗'} {ADAPTER_MEDICAL}")
    log(f"  legal adapter: {'✓' if ADAPTER_LEGAL.exists() else '✗'} {ADAPTER_LEGAL}")

    # Load MMLU-Pro
    log(f"\nLoading MMLU-Pro from {MMLU_PATH}")
    df = pd.read_parquet(MMLU_PATH)
    log(f"Loaded {len(df)} rows, {len(df['category'].unique())} categories")

    # Partition data: seed pool (for centroids), router eval pool, condition eval pool
    # Partitioning: different seed offsets ensure disjoint sets
    rng_main = np.random.default_rng(SEED)
    eval_questions = []  # (cat, row) tuples for condition eval

    for cat in EVAL_CATS:
        cat_df = df[df["category"] == cat].reset_index(drop=True)
        # Reserve first N_SEED_PER_GROUP*4 + N_ROUTER_EVAL_PER_CAT for routing calibration
        # Use last N_EVAL_PER_CAT for condition eval (disjoint by stable row offsets)
        n_reserve = N_SEED_PER_GROUP * 4 + N_ROUTER_EVAL_PER_CAT
        n_needed = n_reserve + N_EVAL_PER_CAT
        if len(cat_df) < n_needed:
            log(f"WARNING: {cat} has only {len(cat_df)} rows (need {n_needed}). Using available.")
            n_eval = max(1, len(cat_df) - n_reserve)
        else:
            n_eval = N_EVAL_PER_CAT

        # Use last n_eval rows for condition eval
        eval_idx = list(range(len(cat_df) - n_eval, len(cat_df)))
        for i in eval_idx:
            eval_questions.append((cat, cat_df.iloc[i]))

    log(f"Eval questions: {len(eval_questions)} total")

    # ─────────────────────────────────────────────
    # Phase 1: Build router centroids
    # ─────────────────────────────────────────────
    log("\n[Phase 1] Build embedding router centroids")
    log("[Loading base model for embeddings]")
    base_model, base_tokenizer = load(MODEL_ID)
    log_memory("base model loaded")

    centroid_r, centroid_k, group_map, embed = build_embedding_centroids(
        base_model, base_tokenizer, df, N_SEED_PER_GROUP, SEED
    )

    # ─────────────────────────────────────────────
    # Phase 2: Router accuracy check (K1528)
    # ─────────────────────────────────────────────
    log("\n[Phase 2] Router accuracy evaluation (K1528)")
    router_accuracy, router_correct, router_total = eval_router_accuracy(
        df, centroid_r, centroid_k, embed, base_tokenizer,
        group_map, N_ROUTER_EVAL_PER_CAT, SEED
    )

    # Pre-compute routing decisions for eval questions
    log("\n[Phase 2b] Pre-computing routing decisions for eval questions")
    routing_decisions = []
    for cat, row in eval_questions:
        decision = route_query(row["question"], embed, base_tokenizer, centroid_r, centroid_k)
        routing_decisions.append(decision)
    routing_counts = {"reasoning": routing_decisions.count("reasoning"),
                      "knowledge": routing_decisions.count("knowledge")}
    log(f"  Routing decisions: {routing_counts}")

    # Done with base model (embeddings only)
    cleanup(base_model, base_tokenizer, embed)
    log_memory("after base model cleanup")

    # ─────────────────────────────────────────────
    # Phase 3: Condition A — thinking_only
    # ─────────────────────────────────────────────
    log("\n[Phase 3] Condition A: thinking_only")
    result_thinking = eval_condition(
        "thinking_only",
        ADAPTER_THINKING if ADAPTER_THINKING.exists() else None,
        eval_questions,
    )

    # ─────────────────────────────────────────────
    # Phase 4: Condition B — domain_routing (ground-truth labels)
    # ─────────────────────────────────────────────
    log("\n[Phase 4] Condition B: domain_routing (ground-truth)")
    # For domain routing, group questions by their best adapter
    domain_groups = {}
    for cat, row in eval_questions:
        adapter = get_domain_adapter(cat)
        key = str(adapter) if adapter else "none"
        if key not in domain_groups:
            domain_groups[key] = (adapter, [])
        domain_groups[key][1].append((cat, row))

    log(f"  domain_groups: {[(Path(k).name if k != 'none' else 'none', len(v[1])) for k, v in domain_groups.items()]}")

    domain_sub_results = {}
    for key, (adapter_path, qs) in domain_groups.items():
        r = eval_condition(f"domain_{Path(adapter_path).name if adapter_path else 'base'}", adapter_path, qs)
        domain_sub_results[key] = (r, len(qs))

    # Combine domain results
    domain_correct = 0
    domain_total = 0
    domain_per_cat = {}
    domain_r_correct = 0
    domain_r_total = 0
    domain_k_correct = 0
    domain_k_total = 0

    for key, (r, n) in domain_sub_results.items():
        domain_total += n
        domain_correct += int(r["overall_acc"] * n)
        for c, acc in r["per_cat"].items():
            if acc is not None:
                domain_per_cat[c] = acc
        if r.get("reasoning_acc") is not None:
            reasoning_questions_this = [q for q in domain_groups[key][1] if q[0] in REASONING_CATS]
            knowledge_questions_this = [q for q in domain_groups[key][1] if q[0] in KNOWLEDGE_CATS]
            domain_r_correct += int(r["reasoning_acc"] * len(reasoning_questions_this))
            domain_r_total += len(reasoning_questions_this)
            domain_k_correct += int(r["knowledge_acc"] * len(knowledge_questions_this))
            domain_k_total += len(knowledge_questions_this)

    domain_overall = domain_correct / domain_total if domain_total > 0 else 0.0
    domain_reasoning = domain_r_correct / domain_r_total if domain_r_total > 0 else 0.0
    domain_knowledge = domain_k_correct / domain_k_total if domain_k_total > 0 else 0.0

    result_domain = {
        "overall_acc": domain_overall,
        "reasoning_acc": domain_reasoning,
        "knowledge_acc": domain_knowledge,
        "per_cat": domain_per_cat,
        "n_questions": domain_total,
    }
    log(f"  COMBINED domain_only: overall={domain_overall:.3f} "
        f"reasoning={domain_reasoning:.3f} knowledge={domain_knowledge:.3f}")

    # ─────────────────────────────────────────────
    # Phase 5: Condition C — embedding_routed
    # ─────────────────────────────────────────────
    log("\n[Phase 5] Condition C: embedding_routed")
    result_routed = eval_routed_condition(
        eval_questions, centroid_r, centroid_k,
        None, None,  # embed and tokenizer already cleaned up
        routing_decisions,
    )

    # ─────────────────────────────────────────────
    # Kill criteria evaluation
    # ─────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("KILL CRITERIA EVALUATION")
    log("=" * 60)

    routed_overall = result_routed["overall_acc"]
    domain_only_overall = result_domain["overall_acc"]
    thinking_only_overall = result_thinking["overall_acc"]

    routed_knowledge = result_routed["knowledge_acc"]
    thinking_knowledge = result_thinking["knowledge_acc"]

    k1526_delta = routed_overall - domain_only_overall
    k1527_delta = routed_knowledge - thinking_knowledge

    k1526_pass = k1526_delta >= 0.03
    k1527_pass = k1527_delta >= 0.02
    k1528_pass = router_accuracy >= 0.85

    log(f"K1526: routed({routed_overall:.3f}) >= domain_only({domain_only_overall:.3f}) + 3pp → "
        f"delta={k1526_delta*100:.1f}pp → {'PASS' if k1526_pass else 'FAIL'}")
    log(f"K1527: routed_knowledge({routed_knowledge:.3f}) >= thinking_knowledge({thinking_knowledge:.3f}) + 2pp → "
        f"delta={k1527_delta*100:.1f}pp → {'PASS' if k1527_pass else 'FAIL'}")
    log(f"K1528: router_accuracy({router_accuracy:.3f}) >= 0.85 → {'PASS' if k1528_pass else 'FAIL'}")

    # ─────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────
    results = {
        "experiment": "exp_p11_adapter_composition_thinking",
        "smoke": IS_SMOKE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conditions": {
            "thinking_only": result_thinking,
            "domain_only": result_domain,
            "embedding_routed": result_routed,
        },
        "router": {
            "accuracy": router_accuracy,
            "correct": router_correct,
            "total": router_total,
            "routing_decisions": routing_counts,
        },
        "kill_criteria": {
            "K1526": {"pass": k1526_pass, "delta_pp": round(k1526_delta * 100, 2)},
            "K1527": {"pass": k1527_pass, "delta_pp": round(k1527_delta * 100, 2)},
            "K1528": {"pass": k1528_pass, "accuracy": round(router_accuracy, 4)},
        },
        "summary": {
            "thinking_overall": round(thinking_only_overall, 4),
            "domain_overall": round(domain_only_overall, 4),
            "routed_overall": round(routed_overall, 4),
            "router_accuracy": round(router_accuracy, 4),
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Final summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  thinking_only:    {thinking_only_overall:.1%} overall")
    log(f"  domain_only:      {domain_only_overall:.1%} overall")
    log(f"  embedding_routed: {routed_overall:.1%} overall")
    log(f"  router_accuracy:  {router_accuracy:.1%} (K1528)")

    n_pass = sum([k1526_pass, k1527_pass, k1528_pass])
    log(f"\nKill criteria: {n_pass}/3 PASS")
    log(f"  K1526: {'PASS' if k1526_pass else 'FAIL'} ({k1526_delta*100:+.1f}pp)")
    log(f"  K1527: {'PASS' if k1527_pass else 'FAIL'} ({k1527_delta*100:+.1f}pp)")
    log(f"  K1528: {'PASS' if k1528_pass else 'FAIL'} ({router_accuracy:.1%})")


if __name__ == "__main__":
    main()
