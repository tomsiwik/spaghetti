#!/usr/bin/env python3
"""SVD solidification: extract experts from merged adapters, measure quality vs raw LoRA.

FlexMoRE shows SVD extraction preserves 93-107% quality (5/6 experts improved).
This experiment: compute LoRA delta = scale * B^T @ A^T, SVD extract at various ranks.

Eckart-Young theorem guarantees:
  - At rank >= 16 (LoRA rank), SVD is LOSSLESS (PPL ratio = 1.000)
  - Quality degrades monotonically with decreasing rank
  - SVD is the OPTIMAL rank-r approximation (no better truncation exists)

Kill criteria:
  K834: SVD expert at best rank > 2x worse PPL than raw LoRA
  K835: SVD composition degrades MMLU more than raw LoRA composition

Predictions:
  P1: SVD at rank=16 reconstruction error = 0, PPL ratio = 1.000
  P2: SVD at rank>16 also lossless (delta is rank 16)
  P3: PPL degrades monotonically: ratio(r=4) > ratio(r=8) > ratio(r=16)
  P4: Best rank mean PPL ratio < 2.0 (K834 PASS)

Memory strategy: Process one (domain, rank) at a time. Never hold all deltas
simultaneously. Each delta is ~14.5 GB in float32; all 5 = 72 GB = OOM.
"""

import gc, json, math, os, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre.bench import Experiment, ppl, cleanup
from pierre import load_adapter, load_frozen_A, attach_adapter
from mlx_lm import load

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Use Pierre Pro (Qwen3) if available, else Pierre Tiny (BitNet)
PRO_ADAPTERS = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
TINY_ADAPTERS = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
PRO_SKELETON = EXPERIMENT_DIR.parent / "pro_grassmannian_init" / "grassmannian_skeleton_n5.npz"
TINY_SKELETON = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters" / "grassmannian_skeleton.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

LORA_SCALE = 20.0
LORA_RANK = 16
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
SVD_RANKS = [4, 8, 16, 32, 64, 128]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Same 50 MMLU questions used in pro_composition_mmlu for consistency
MMLU_QUESTIONS = [
    ("physics", "A 2 kg object at 3 m/s collides with a stationary 1 kg object in a perfectly inelastic collision. What is the speed after collision?", "A) 1 m/s\nB) 2 m/s\nC) 3 m/s\nD) 6 m/s", "B"),
    ("physics", "What is the SI unit of electrical resistance?", "A) Volt\nB) Ampere\nC) Ohm\nD) Watt", "C"),
    ("physics", "According to Newton's second law, force equals:", "A) mass times velocity\nB) mass times acceleration\nC) mass times distance\nD) mass times time", "B"),
    ("physics", "What is the speed of light in a vacuum?", "A) 3 x 10^6 m/s\nB) 3 x 10^7 m/s\nC) 3 x 10^8 m/s\nD) 3 x 10^9 m/s", "C"),
    ("physics", "What is the unit of frequency?", "A) Watt\nB) Joule\nC) Hertz\nD) Pascal", "C"),
    ("physics", "A ball is dropped from rest. After 2 seconds of free fall (g=10 m/s^2), its speed is:", "A) 10 m/s\nB) 20 m/s\nC) 30 m/s\nD) 40 m/s", "B"),
    ("chemistry", "What is the molecular formula of glucose?", "A) C6H12O6\nB) C12H22O11\nC) CH3COOH\nD) C2H5OH", "A"),
    ("chemistry", "Which element has the highest electronegativity?", "A) Oxygen\nB) Chlorine\nC) Fluorine\nD) Nitrogen", "C"),
    ("chemistry", "What is the pH of pure water at 25 degrees Celsius?", "A) 0\nB) 1\nC) 7\nD) 14", "C"),
    ("chemistry", "What is the atomic number of carbon?", "A) 4\nB) 6\nC) 8\nD) 12", "B"),
    ("biology", "Which organelle produces ATP in eukaryotic cells?", "A) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi apparatus", "C"),
    ("biology", "What type of bond holds DNA strands together?", "A) Covalent bonds\nB) Ionic bonds\nC) Hydrogen bonds\nD) Metallic bonds", "C"),
    ("biology", "What is the powerhouse of the cell?", "A) Nucleus\nB) Mitochondria\nC) Chloroplast\nD) Endoplasmic reticulum", "B"),
    ("biology", "Which molecule carries amino acids to the ribosome during translation?", "A) mRNA\nB) rRNA\nC) tRNA\nD) DNA", "C"),
    ("math", "What is the derivative of x^3?", "A) x^2\nB) 3x^2\nC) 3x\nD) x^3", "B"),
    ("math", "If log base 2 of x equals 5, what is x?", "A) 10\nB) 25\nC) 32\nD) 64", "C"),
    ("math", "What is the sum of the interior angles of a hexagon?", "A) 360 degrees\nB) 540 degrees\nC) 720 degrees\nD) 900 degrees", "C"),
    ("math", "If f(x) = 2x + 3, what is f(f(1))?", "A) 7\nB) 13\nC) 11\nD) 9", "B"),
    ("math", "What is the value of pi to two decimal places?", "A) 3.12\nB) 3.14\nC) 3.16\nD) 3.18", "B"),
    ("computer_science", "What is the time complexity of binary search?", "A) O(1)\nB) O(n)\nC) O(log n)\nD) O(n log n)", "C"),
    ("computer_science", "Which data structure uses FIFO ordering?", "A) Stack\nB) Queue\nC) Binary tree\nD) Hash table", "B"),
    ("computer_science", "What does SQL stand for?", "A) Structured Query Language\nB) Sequential Query Logic\nC) Standard Query Library\nD) System Query Language", "A"),
    ("computer_science", "In a binary search tree, worst-case search time is:", "A) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)", "C"),
    ("history", "In what year did World War II end?", "A) 1943\nB) 1944\nC) 1945\nD) 1946", "C"),
    ("history", "Who was the first President of the United States?", "A) Thomas Jefferson\nB) John Adams\nC) Benjamin Franklin\nD) George Washington", "D"),
    ("history", "The French Revolution began in:", "A) 1776\nB) 1789\nC) 1799\nD) 1804", "B"),
    ("history", "The Berlin Wall fell in:", "A) 1987\nB) 1988\nC) 1989\nD) 1990", "C"),
    ("history", "Who discovered penicillin?", "A) Louis Pasteur\nB) Alexander Fleming\nC) Joseph Lister\nD) Robert Koch", "B"),
    ("philosophy", "Who wrote 'The Republic'?", "A) Aristotle\nB) Socrates\nC) Plato\nD) Epicurus", "C"),
    ("philosophy", "The categorical imperative is associated with:", "A) John Stuart Mill\nB) Immanuel Kant\nC) David Hume\nD) Friedrich Nietzsche", "B"),
    ("philosophy", "Cogito ergo sum was stated by:", "A) Descartes\nB) Locke\nC) Spinoza\nD) Leibniz", "A"),
    ("literature", "Who wrote Romeo and Juliet?", "A) Charles Dickens\nB) William Shakespeare\nC) Jane Austen\nD) Mark Twain", "B"),
    ("literature", "Who wrote 1984?", "A) Aldous Huxley\nB) George Orwell\nC) Ray Bradbury\nD) H.G. Wells", "B"),
    ("literature", "In which century was Don Quixote first published?", "A) 15th\nB) 16th\nC) 17th\nD) 18th", "C"),
    ("economics", "What does GDP stand for?", "A) General Domestic Product\nB) Gross Domestic Product\nC) Gross Domestic Profit\nD) General Domestic Profit", "B"),
    ("economics", "According to the law of demand, as price increases:", "A) quantity demanded increases\nB) quantity demanded decreases\nC) supply increases\nD) supply decreases", "B"),
    ("economics", "Inflation is defined as:", "A) A decrease in the general price level\nB) An increase in the general price level\nC) A decrease in unemployment\nD) An increase in GDP", "B"),
    ("psychology", "Who is the father of psychoanalysis?", "A) Carl Jung\nB) B.F. Skinner\nC) Sigmund Freud\nD) Ivan Pavlov", "C"),
    ("psychology", "Classical conditioning was discovered by:", "A) B.F. Skinner\nB) Ivan Pavlov\nC) John Watson\nD) Albert Bandura", "B"),
    ("psychology", "Maslow's hierarchy places which need at the base?", "A) Self-actualization\nB) Esteem\nC) Safety\nD) Physiological", "D"),
    ("geography", "What is the largest ocean on Earth?", "A) Atlantic\nB) Indian\nC) Arctic\nD) Pacific", "D"),
    ("geography", "Which continent has the most countries?", "A) Asia\nB) Europe\nC) Africa\nD) South America", "C"),
    ("geography", "What is the longest river in the world?", "A) Amazon\nB) Nile\nC) Mississippi\nD) Yangtze", "B"),
    ("law", "Habeas corpus protects against:", "A) Double jeopardy\nB) Unlawful detention\nC) Self-incrimination\nD) Cruel punishment", "B"),
    ("medicine", "What organ produces insulin?", "A) Liver\nB) Kidney\nC) Pancreas\nD) Spleen", "C"),
    ("medicine", "Normal resting heart rate for adults (bpm)?", "A) 40-60\nB) 60-100\nC) 100-120\nD) 120-140", "B"),
    ("engineering", "Ohm's law is:", "A) V = IR\nB) F = ma\nC) E = mc^2\nD) P = IV", "A"),
    ("astronomy", "Which planet is the Red Planet?", "A) Venus\nB) Mars\nC) Jupiter\nD) Saturn", "B"),
    ("astronomy", "How many planets in our solar system?", "A) 7\nB) 8\nC) 9\nD) 10", "B"),
    ("nutrition", "Which vitamin is produced by sunlight exposure?", "A) Vitamin A\nB) Vitamin B12\nC) Vitamin C\nD) Vitamin D", "D"),
]


def log(m): print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    c = mx.get_cache_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB peak={p:.2f}GB")


def load_data(domain, split="valid", n=None):
    texts = []
    with open(DATA_DIR / domain / f"{split}.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if n and len(texts) >= n:
                break
    return texts


# ── SVD extraction core ─────────────────────────────────────────────────────
# Key insight: do NOT materialize full delta as (out, in) matrix.
# Instead, compute SVD efficiently from the rank-16 factors directly.
# For a rank-r matrix delta = scale * B^T @ A^T where B is (r, out), A is (in, r):
#   delta = (scale * B^T) @ A^T  = M @ N^T  where M=(out,r), N=(in,r)
# The SVD can be computed from the (r, r) inner product: N^T @ M @ ... no,
# we need the SVD of the (out, in) product. But since rank=16 and dimensions
# are in thousands, the key is: SVD of an (m,n) rank-k matrix costs O(mn*k).
#
# Even better: we can avoid the full (out,in) matrix entirely by working with
# the (16,16) cross-product. The thin SVD of M @ N^T can be obtained from:
#   1. QR of M: M = Q_M R_M, Q_M (out,r), R_M (r,r)
#   2. QR of N: N = Q_N R_N, Q_N (in,r), R_N (r,r)
#   3. SVD of R_M @ R_N^T: this is (r,r) = tiny
#   4. U = Q_M @ U_small, V = Q_N @ V_small


def compute_delta_factors(frozen_A, adapter_b, domain_idx, n_layers, scale):
    """Compute LoRA delta factors WITHOUT materializing full (out, in) matrix.

    Returns dict of module_key -> (M, N) where delta = M @ N^T.
    M = scale * B^T  (out_features, rank)
    N = A             (in_features, rank)
    """
    factors = {}
    for li in range(n_layers):
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in frozen_A:
                continue
            A = mx.array(frozen_A[ak]).astype(mx.float32)   # (in, rank)
            B = adapter_b[bk].astype(mx.float32)             # (rank, out)
            M = scale * B.T   # (out, rank)
            N = A             # (in, rank)
            mx.eval(M, N)
            factors[f"layer_{li}_{key}"] = (M, N)
    return factors


def svd_extract_from_factors(factors, rank):
    """SVD extraction from factored form: delta = M @ N^T, without full matrix.

    For each module, computes efficient thin SVD via (r_lora, r_lora) inner product,
    then truncates to the desired rank.

    Returns:
      extracted: dict of module_key -> (A_svd, B_svd) in bf16
                 A_svd (in, rank_svd), B_svd (rank_svd, out)
      rel_error: relative reconstruction error
      spectral:  singular values for first module (for analysis)
    """
    extracted = {}
    total_error_sq = 0.0
    total_energy_sq = 0.0
    first_svs = None

    for key, (M, N) in factors.items():
        # M (out, r_lora), N (in, r_lora), delta = M @ N^T
        # QR factorization to get orthonormal bases
        Q_m, R_m = mx.linalg.qr(M, stream=mx.cpu)   # Q_m (out, r), R_m (r, r)
        Q_n, R_n = mx.linalg.qr(N, stream=mx.cpu)   # Q_n (in, r), R_n (r, r)
        mx.eval(Q_m, R_m, Q_n, R_n)

        # SVD of the small (r, r) matrix
        small = R_m @ R_n.T    # (r, r)
        U_s, S, Vt_s = mx.linalg.svd(small, stream=mx.cpu)  # U_s (r,r), S (r,), Vt_s (r,r)
        mx.eval(U_s, S, Vt_s)

        if first_svs is None:
            first_svs = S.tolist()

        # Full thin SVD: delta = (Q_m @ U_s) @ diag(S) @ (Vt_s @ Q_n^T)
        # Truncate to desired rank
        r = min(rank, S.shape[0])
        U_full = Q_m @ U_s[:, :r]     # (out, r)
        Vt_full = Vt_s[:r, :] @ Q_n.T  # (r, in)
        S_r = S[:r]

        # Split SVs symmetrically into activation-space factors:
        # A_svd (in, r) and B_svd (r, out)
        # such that x @ A_svd @ B_svd approximates x @ delta^T
        sqrt_S = mx.sqrt(S_r)
        A_svd = (Vt_full.T * sqrt_S[None, :])    # (in, r)
        B_svd = (U_full * sqrt_S[None, :]).T       # (r, out)
        mx.eval(A_svd, B_svd)

        # Reconstruction error from discarded singular values
        energy_sq = mx.sum(S * S).item()
        retained_sq = mx.sum(S_r * S_r).item()
        error_sq = energy_sq - retained_sq
        total_error_sq += error_sq
        total_energy_sq += energy_sq

        extracted[key] = (A_svd.astype(mx.bfloat16), B_svd.astype(mx.bfloat16))
        del Q_m, R_m, Q_n, R_n, small, U_s, Vt_s, U_full, Vt_full, S, S_r, sqrt_S

    rel_error = math.sqrt(total_error_sq / (total_energy_sq + 1e-30))
    return extracted, rel_error, first_svs


class SVDExpertLayer(nn.Module):
    """Applies SVD-extracted expert as a side-path: y = base(x) + x @ A_svd @ B_svd"""
    def __init__(self, base, A_svd, B_svd):
        super().__init__()
        self.base = base
        self.A_svd = A_svd  # (in, r)
        self.B_svd = B_svd  # (r, out)
        self.freeze()

    def __call__(self, x):
        y = self.base(x)
        expert_out = (x @ self.A_svd) @ self.B_svd
        return y + expert_out.astype(y.dtype)


def inject_svd_expert(model, extracted_expert):
    """Inject SVD expert into model as side-path."""
    count = 0
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        updates = []
        for key in TARGET_KEYS:
            ek = f"layer_{li}_{key}"
            if ek not in extracted_expert:
                continue
            A_svd, B_svd = extracted_expert[ek]
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            wrapped = SVDExpertLayer(m, A_svd, B_svd)
            updates.append((key, wrapped))
            count += 1
        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


# ── MMLU evaluation ─────────────────────────────────────────────────────────


def eval_mmlu(model, tokenizer, max_seq=512):
    """Logit-based MMLU evaluation (same 50Q as pro_composition_mmlu)."""
    answer_tokens = {}
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(f" {letter}")
        answer_tokens[letter] = ids[-1]

    correct = 0
    total = 0
    for subject, question, choices, answer in MMLU_QUESTIONS:
        prompt = f"Question: {question}\n{choices}\nAnswer: The correct answer is"
        tokens = tokenizer.encode(prompt)[:max_seq]
        x = mx.array(tokens)[None, :]
        logits = model(x)
        mx.eval(logits)
        last = logits[0, -1]
        preds = {k: last[v].item() for k, v in answer_tokens.items()}
        predicted = max(preds, key=preds.get)
        if predicted == answer:
            correct += 1
        total += 1
        del logits, x
    return correct, total


# ── Phase functions (each self-contained, cleanup after) ────────────────────


def phase_select_model():
    """Determine which model/adapters to use."""
    if PRO_ADAPTERS.exists() and (PRO_ADAPTERS / "medical" / "adapter.npz").exists():
        adapter_dir = PRO_ADAPTERS
        skeleton_path = PRO_SKELETON
        model_id = "mlx-community/Qwen3-4B-4bit"
        tag = "pro"
    else:
        adapter_dir = TINY_ADAPTERS
        skeleton_path = TINY_SKELETON
        model_id = "microsoft/BitNet-b1.58-2B-4T"
        tag = "tiny"

    log(f"Using {tag} ({model_id})")
    return model_id, adapter_dir, skeleton_path, tag


def phase_raw_lora_ppl(model_id, adapter_dir, skeleton_path, val_data):
    """Phase 1: Raw LoRA PPL baseline for each domain."""
    log("\n" + "=" * 60)
    log("Phase 1: Raw LoRA PPL baseline (scale=20)")
    log("=" * 60)

    frozen_A = load_frozen_A(str(skeleton_path))
    raw_ppls = {}
    for di, domain in enumerate(DOMAINS):
        model, tok = load(model_id)
        adapter = load_adapter(str(adapter_dir / domain / "adapter.npz"))
        attach_adapter(model, frozen_A, adapter, di, LORA_SCALE)
        p = round(ppl(model, tok, val_data[domain]), 3)
        raw_ppls[domain] = p
        log(f"  raw_lora/{domain}: PPL={p}")
        cleanup(model, tok, adapter)
    del frozen_A
    gc.collect()
    log_memory("post-raw-lora")
    return raw_ppls


def phase_svd_quality_sweep(model_id, adapter_dir, skeleton_path, val_data, raw_ppls):
    """Phase 2+3: For each (domain, rank), compute delta -> SVD -> PPL.

    Memory-efficient: only one domain's factors + one model in memory at a time.
    """
    log("\n" + "=" * 60)
    log("Phase 2+3: SVD quality sweep (memory-efficient)")
    log("=" * 60)

    frozen_A = load_frozen_A(str(skeleton_path))

    # Get n_layers from model
    model, tok = load(model_id)
    n_layers = len(model.model.layers)
    log(f"  Model has {n_layers} layers")
    cleanup(model, tok)

    rank_results = {r: {"ppls": {}, "errors": {}} for r in SVD_RANKS}
    spectral_analysis = {}

    for di, domain in enumerate(DOMAINS):
        log(f"\n  --- Domain: {domain} ---")
        adapter = load_adapter(str(adapter_dir / domain / "adapter.npz"))
        factors = compute_delta_factors(frozen_A, adapter, di, n_layers, LORA_SCALE)
        log(f"    Computed {len(factors)} factor pairs")
        log_memory(f"factors-{domain}")
        del adapter

        for rank in SVD_RANKS:
            extracted, rel_error, svs = svd_extract_from_factors(factors, rank)

            # Save spectral info from first extraction
            if rank == SVD_RANKS[0]:
                spectral_analysis[domain] = {
                    "singular_values": [round(s, 4) for s in svs[:20]] if svs else [],
                    "n_svs": len(svs) if svs else 0,
                }
                if svs:
                    total_e = sum(s ** 2 for s in svs)
                    for rk in [1, 2, 4, 8, 16]:
                        if rk <= len(svs):
                            e = sum(s ** 2 for s in svs[:rk])
                            spectral_analysis[domain][f"energy_at_r{rk}"] = round(e / total_e, 6)

            model, tok = load(model_id)
            inject_svd_expert(model, extracted)
            p = round(ppl(model, tok, val_data[domain]), 3)
            ratio = round(p / raw_ppls[domain], 4) if raw_ppls[domain] > 0 else float('inf')
            log(f"    rank={rank}: PPL={p} ratio={ratio}x recon_err={rel_error:.6f}")
            rank_results[rank]["ppls"][domain] = p
            rank_results[rank]["errors"][domain] = round(rel_error, 6)
            cleanup(model, tok)
            del extracted

        del factors
        gc.collect()
        mx.clear_cache()
        log_memory(f"post-{domain}")

    del frozen_A
    gc.collect()

    # Compute summary stats
    for rank in SVD_RANKS:
        ppls = rank_results[rank]["ppls"]
        ratios = [ppls[d] / raw_ppls[d] for d in DOMAINS]
        rank_results[rank]["mean_ratio"] = round(float(np.mean(ratios)), 4)
        rank_results[rank]["worst_ratio"] = round(float(max(ratios)), 4)

    return rank_results, spectral_analysis


def phase_mmlu_comparison(model_id, adapter_dir, skeleton_path, best_rank):
    """Phase 4: MMLU comparison between base, raw LoRA, and SVD expert.

    Tests K835: does SVD expert preserve MMLU better/same as raw LoRA?
    Uses single medical adapter at scale=20 (worst case from Finding #320).
    """
    log("\n" + "=" * 60)
    log("Phase 4: MMLU comparison (K835)")
    log("=" * 60)

    results = {}

    # 4a: Base model MMLU
    log("  4a: Base model MMLU")
    model, tok = load(model_id)
    base_correct, base_total = eval_mmlu(model, tok)
    base_acc = base_correct / base_total
    results["base"] = {"correct": base_correct, "total": base_total,
                       "accuracy": round(base_acc, 4)}
    log(f"    Base: {base_correct}/{base_total} = {base_acc:.1%}")
    cleanup(model, tok)

    # 4b: Raw LoRA single medical adapter at scale=20
    log("  4b: Raw LoRA medical (scale=20) MMLU")
    frozen_A = load_frozen_A(str(skeleton_path))
    model, tok = load(model_id)
    adapter = load_adapter(str(adapter_dir / "medical" / "adapter.npz"))
    attach_adapter(model, frozen_A, adapter, 0, LORA_SCALE)
    raw_correct, raw_total = eval_mmlu(model, tok)
    raw_acc = raw_correct / raw_total
    raw_degradation_pp = round((raw_acc - base_acc) * 100, 1)
    results["raw_lora_medical_s20"] = {
        "correct": raw_correct, "total": raw_total,
        "accuracy": round(raw_acc, 4),
        "degradation_pp": raw_degradation_pp,
    }
    log(f"    Raw LoRA: {raw_correct}/{raw_total} = {raw_acc:.1%} (degradation: {raw_degradation_pp}pp)")
    cleanup(model, tok, adapter)

    # 4c: SVD experts at rank=16 (lossless) and at low ranks (truncated)
    n_layers_ref = [0]

    def get_n_layers():
        if n_layers_ref[0] == 0:
            m, t = load(model_id)
            n_layers_ref[0] = len(m.model.layers)
            cleanup(m, t)
        return n_layers_ref[0]

    for test_rank in [4, 8, best_rank]:
        if test_rank in [r["rank"] for r in results.values() if isinstance(r, dict) and "rank" in r]:
            continue
        log(f"  4c: SVD expert (rank={test_rank}) medical MMLU")
        adapter = load_adapter(str(adapter_dir / "medical" / "adapter.npz"))
        factors = compute_delta_factors(frozen_A, adapter, 0, get_n_layers(), LORA_SCALE)
        extracted, rel_error, _ = svd_extract_from_factors(factors, test_rank)
        model, tok = load(model_id)
        inject_svd_expert(model, extracted)
        svd_correct, svd_total = eval_mmlu(model, tok)
        svd_acc = svd_correct / svd_total
        svd_degradation_pp = round((svd_acc - base_acc) * 100, 1)
        results[f"svd_r{test_rank}_medical"] = {
            "correct": svd_correct, "total": svd_total,
            "accuracy": round(svd_acc, 4),
            "degradation_pp": svd_degradation_pp,
            "rank": test_rank,
        }
        log(f"    SVD r{test_rank}: {svd_correct}/{svd_total} = {svd_acc:.1%} (degradation: {svd_degradation_pp}pp)")
        cleanup(model, tok)
        del adapter, factors, extracted

    del frozen_A
    gc.collect()
    mx.clear_cache()
    log_memory("post-mmlu")
    return results


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    exp = Experiment("svd_extraction_quality", dir=str(EXPERIMENT_DIR))
    mx.random.seed(SEED)

    # Select model and adapters
    model_id, adapter_dir, skeleton_path, tag = phase_select_model()
    exp.metric("model", model_id, display=False)
    exp.metric("tag", tag, display=False)

    # Load validation data (tiny: 20 texts per domain)
    val_data = {d: load_data(d, "valid", 20) for d in DOMAINS}

    # Phase 1: Raw LoRA PPL baseline
    with exp.phase("raw_lora_baseline"):
        raw_ppls = phase_raw_lora_ppl(model_id, adapter_dir, skeleton_path, val_data)
        exp.metric("raw_lora_ppls", raw_ppls)

    # Phase 2+3: SVD quality sweep (memory-efficient: one domain at a time)
    with exp.phase("svd_quality_sweep"):
        rank_results, spectral_analysis = phase_svd_quality_sweep(
            model_id, adapter_dir, skeleton_path, val_data, raw_ppls)
        for rank in SVD_RANKS:
            exp.metric(f"svd_r{rank}_mean_ratio", rank_results[rank]["mean_ratio"])

    # Find best rank
    best_rank = min(rank_results.keys(), key=lambda r: rank_results[r]["mean_ratio"])
    exp.metric("best_rank", best_rank)
    exp.metric("best_rank_mean_ratio", rank_results[best_rank]["mean_ratio"])
    exp.metric("best_rank_worst_ratio", rank_results[best_rank]["worst_ratio"])
    log(f"\nBest rank: {best_rank} (mean_ratio={rank_results[best_rank]['mean_ratio']}, "
        f"worst_ratio={rank_results[best_rank]['worst_ratio']})")

    # Phase 4: MMLU comparison (K835)
    with exp.phase("mmlu_comparison"):
        mmlu_results = phase_mmlu_comparison(model_id, adapter_dir, skeleton_path, best_rank)
        exp.metric("base_mmlu_accuracy", mmlu_results["base"]["accuracy"])
        exp.metric("raw_lora_mmlu_degradation_pp",
                   mmlu_results["raw_lora_medical_s20"]["degradation_pp"])
        svd_key = f"svd_r{best_rank}_medical"
        if svd_key in mmlu_results:
            exp.metric("svd_best_mmlu_degradation_pp",
                       mmlu_results[svd_key]["degradation_pp"])

    # Prediction verification
    log("\n" + "=" * 60)
    log("PREDICTION VERIFICATION")
    log("=" * 60)

    predictions = {}

    # P1: SVD at rank=16 is lossless (ratio = 1.000)
    r16 = rank_results.get(16, {})
    r16_ratio = r16.get("mean_ratio")
    r16_errors = r16.get("errors", {})
    p1_pass = r16_ratio is not None and abs(r16_ratio - 1.0) < 0.02
    predictions["P1_rank16_lossless"] = {
        "pass": p1_pass, "ratio": r16_ratio, "errors": r16_errors}
    log(f"  P1 (rank=16 lossless): ratio={r16_ratio}, "
        f"max_err={max(r16_errors.values()) if r16_errors else 'N/A'}")
    log(f"      {'CONFIRMED' if p1_pass else 'REFUTED'}")

    # P2: SVD at rank>16 also lossless
    r32 = rank_results.get(32, {})
    r32_ratio = r32.get("mean_ratio")
    p2_pass = r32_ratio is not None and abs(r32_ratio - 1.0) < 0.02
    predictions["P2_rank_gt16_lossless"] = {"pass": p2_pass, "ratio": r32_ratio}
    log(f"  P2 (rank>16 lossless): r32_ratio={r32_ratio}")
    log(f"      {'CONFIRMED' if p2_pass else 'REFUTED'}")

    # P3: Monotonic degradation (ratio increases as rank decreases)
    ratios_by_rank = [(r, rank_results[r]["mean_ratio"]) for r in sorted(SVD_RANKS) if r <= 16]
    p3_pass = all(ratios_by_rank[i][1] >= ratios_by_rank[i + 1][1]
                  for i in range(len(ratios_by_rank) - 1))
    predictions["P3_monotonic"] = {"pass": p3_pass, "rank_ratios": ratios_by_rank}
    log(f"  P3 (monotonic degradation): {ratios_by_rank}")
    log(f"      {'CONFIRMED' if p3_pass else 'REFUTED'}")

    # P4: Best rank ratio < 2.0 (K834)
    p4_pass = rank_results[best_rank]["mean_ratio"] < 2.0
    predictions["P4_best_under_2x"] = {
        "pass": p4_pass, "ratio": rank_results[best_rank]["mean_ratio"]}
    log(f"  P4 (best rank < 2x): ratio={rank_results[best_rank]['mean_ratio']}")
    log(f"      {'CONFIRMED' if p4_pass else 'REFUTED'}")

    # Store full results
    exp.results["rank_results"] = {str(k): v for k, v in rank_results.items()}
    exp.results["mmlu_results"] = mmlu_results
    exp.results["spectral_analysis"] = spectral_analysis
    exp.results["prediction_verification"] = predictions

    # Kill criteria
    exp.kill_if("best_rank_mean_ratio", ">", 2.0, kid="K834")

    # K835: SVD at best rank degrades MMLU more than raw LoRA?
    svd_key = f"svd_r{best_rank}_medical"
    if svd_key in mmlu_results:
        svd_degrad = abs(mmlu_results[svd_key]["degradation_pp"])
        raw_degrad = abs(mmlu_results["raw_lora_medical_s20"]["degradation_pp"])
        exp.metric("k835_svd_abs_degradation_pp", svd_degrad)
        exp.metric("k835_raw_abs_degradation_pp", raw_degrad)
        # KILL if SVD is strictly worse than raw LoRA
        exp.kill_if("k835_svd_abs_degradation_pp", ">", raw_degrad + 0.5, kid="K835")

    # Save SVD experts at best rank for downstream experiments
    log(f"\nSaving SVD experts at rank {best_rank}...")
    frozen_A = load_frozen_A(str(skeleton_path))
    model, tok = load(model_id)
    n_layers = len(model.model.layers)
    cleanup(model, tok)
    for di, domain in enumerate(DOMAINS):
        adapter = load_adapter(str(adapter_dir / domain / "adapter.npz"))
        factors = compute_delta_factors(frozen_A, adapter, di, n_layers, LORA_SCALE)
        extracted, _, _ = svd_extract_from_factors(factors, best_rank)
        save_dir = EXPERIMENT_DIR / "svd_experts" / domain
        save_dir.mkdir(parents=True, exist_ok=True)
        save_data = {}
        for key, (A, B) in extracted.items():
            save_data[f"{key}_A"] = A
            save_data[f"{key}_B"] = B
        mx.savez(str(save_dir / f"expert_r{best_rank}.npz"), **save_data)
        del adapter, factors, extracted, save_data
        gc.collect()
    del frozen_A
    gc.collect()
    mx.clear_cache()
    log(f"  Saved SVD experts for all {len(DOMAINS)} domains")

    exp.save()


if __name__ == "__main__":
    main()
