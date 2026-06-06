#!/usr/bin/env python3
"""Solidified expert composition: does SVD extraction fix the scale=20 MMLU catastrophe?

Finding #320: raw LoRA at scale=20 destroys MMLU by -60pp (single), -44pp (N=5 composed).
Finding #326: SVD improvement is magnitude reduction, not directional selection.

This experiment tests whether SVD-truncated composition preserves MMLU better than
raw composition, and whether scale reduction is equivalent (Theorem 3 in MATH.md).

Configurations:
  1. Base Qwen3-4B MMLU (control: should be ~92%)
  2. Raw LoRA N=5 scale=20 (replicate: should be ~-44pp)
  3. SVD rank=4 N=5 scale=20 (key test: predicted -25 to -35pp)
  4. SVD rank=1 N=5 scale=20 (aggressive: predicted -17 to -27pp)
  5. Full-rank N=5 scale=13 (energy-match: should match SVD rank=4 within 5pp)
  6. Full-rank N=5 scale=5 (safe control: should be 0 to -2pp)

Kill criteria:
  K837: MMLU degradation > 15pp (SVD composition worse than single SVD)
  K838: Domain quality < 50% of raw LoRA (SVD destroys domain expertise)
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

from pierre.bench import Experiment, mmlu_eval, ppl, cleanup
from pierre import load_adapter, load_frozen_A, compose_adapters
from mlx_lm import load

EXPERIMENT_DIR = Path(__file__).parent
SVD_DIR = EXPERIMENT_DIR.parent / "svd_extraction_quality"
ADAPTER_DIR = EXPERIMENT_DIR.parent / "pro_sft_5_adapters" / "adapters"
SKELETON_PATH = EXPERIMENT_DIR.parent / "pro_grassmannian_init" / "grassmannian_skeleton_n5.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

MODEL_ID = "mlx-community/Qwen3-4B-4bit"
LORA_RANK = 16
LORA_SCALE = 20.0
SEED = 42
MAX_SEQ = 512

DOMAINS = ["medical", "code", "math", "legal", "finance"]

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Same 50 MMLU questions as pro_composition_mmlu for consistency
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


def log(m):
    print(m, flush=True)


def log_memory(label=""):
    a = mx.get_active_memory() / 1e9
    p = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={a:.2f}GB peak={p:.2f}GB")


# ── LoRA attachment (same as pro_composition_mmlu) ─────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, base_module, rank=16, scale=20.0, a_init=None):
        super().__init__()
        self.base = base_module
        in_f = base_module.in_features if hasattr(base_module, 'in_features') else base_module.weight.shape[-1]
        out_f = base_module.out_features if hasattr(base_module, 'out_features') else base_module.weight.shape[0]
        self.lora_a = a_init if a_init is not None else mx.random.normal(shape=(in_f, rank)) * (1.0 / math.sqrt(in_f))
        self.lora_b = mx.zeros((rank, out_f))
        self.scale = scale
        self.base.freeze()
        self.freeze(keys=["base", "lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.base(x)
        return base_out + ((x @ self.lora_a) @ self.lora_b * self.scale).astype(base_out.dtype)


def attach_adapter(model, skeleton, adapter_b, domain_idx, scale):
    """Attach a single adapter's B-weights using Grassmannian skeleton A-matrices."""
    count = 0
    for li, layer in enumerate(model.model.layers):
        updates = []
        for key in TARGET_KEYS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
            if m is None:
                continue
            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            lora = LoRALinear(m, rank=LORA_RANK, scale=scale, a_init=A)
            lora.lora_b = adapter_b[bk].astype(mx.bfloat16)
            updates.append((key, lora))
            count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


def attach_composed_adapter(model, skeleton, composed_b, scale):
    """Attach a COMPOSED adapter. Uses domain_0's A-matrices (NRE averages B only)."""
    return attach_adapter(model, skeleton, composed_b, 0, scale)


# ── SVD expert injection (same as svd_extraction_quality) ──────────────────

class SVDExpertLayer(nn.Module):
    """Applies SVD-extracted expert: y = base(x) + x @ A_svd @ B_svd"""
    def __init__(self, base, A_svd, B_svd):
        super().__init__()
        self.base = base
        self.A_svd = A_svd
        self.B_svd = B_svd
        self.freeze()

    def __call__(self, x):
        y = self.base(x)
        return y + ((x @ self.A_svd) @ self.B_svd).astype(y.dtype)


def load_svd_expert(domain, svd_dir, rank_suffix="r4"):
    """Load SVD-extracted expert from saved npz."""
    expert_path = svd_dir / "svd_experts" / domain / f"expert_{rank_suffix}.npz"
    if not expert_path.exists():
        return None
    data = dict(mx.load(str(expert_path)))
    expert = {}
    keys = set(k.rsplit("_", 1)[0] for k in data.keys() if k.endswith("_A") or k.endswith("_B"))
    for k in keys:
        if f"{k}_A" in data and f"{k}_B" in data:
            expert[k] = (data[f"{k}_A"], data[f"{k}_B"])
    return expert


def inject_svd_expert(model, expert):
    """Inject SVD expert into model as side-path."""
    count = 0
    for li in range(len(model.model.layers)):
        updates = []
        for key in TARGET_KEYS:
            ek = f"layer_{li}_{key}"
            if ek not in expert:
                continue
            A_svd, B_svd = expert[ek]
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
            if m is None:
                continue
            updates.append((key, SVDExpertLayer(m, A_svd, B_svd)))
            count += 1
        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    return count


def compose_svd_experts(experts_list):
    """NRE-merge SVD experts: average A_svd and B_svd with norm rescaling.

    Each expert is a dict of module_key -> (A_svd, B_svd).
    We average the A and B factors separately with norm preservation.
    """
    if len(experts_list) == 1:
        return experts_list[0]

    all_keys = set()
    for e in experts_list:
        all_keys.update(e.keys())

    composed = {}
    for k in all_keys:
        As = [e[k][0] for e in experts_list if k in e]
        Bs = [e[k][1] for e in experts_list if k in e]
        n = len(As)
        # Average and norm-rescale A
        mean_A = sum(a.astype(mx.float32) for a in As) / n
        src_norm_A = mx.mean(mx.stack([mx.linalg.norm(a.reshape(-1).astype(mx.float32)) for a in As]))
        mean_norm_A = mx.linalg.norm(mean_A.reshape(-1))
        mx.eval(src_norm_A, mean_norm_A)
        if mean_norm_A.item() > 1e-8:
            mean_A = mean_A * (src_norm_A / mean_norm_A)

        # Average and norm-rescale B
        mean_B = sum(b.astype(mx.float32) for b in Bs) / n
        src_norm_B = mx.mean(mx.stack([mx.linalg.norm(b.reshape(-1).astype(mx.float32)) for b in Bs]))
        mean_norm_B = mx.linalg.norm(mean_B.reshape(-1))
        mx.eval(src_norm_B, mean_norm_B)
        if mean_norm_B.item() > 1e-8:
            mean_B = mean_B * (src_norm_B / mean_norm_B)

        composed[k] = (mean_A.astype(mx.bfloat16), mean_B.astype(mx.bfloat16))

    return composed


def load_data(domain, split="valid", n=None):
    texts = []
    with open(DATA_DIR / domain / f"{split}.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if n and len(texts) >= n:
                break
    return texts


# ── Phase functions ────────────────────────────────────────────────────────

def phase_base_mmlu():
    """Phase 1: Base model MMLU (50Q)."""
    log("\n" + "=" * 60)
    log("Phase 1: Base model MMLU")
    log("=" * 60)
    model, tok = load(MODEL_ID)
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    acc = correct / total if total else 0
    log(f"  Base MMLU: {acc:.1%} ({correct}/{total})")
    log_memory("post-base-mmlu")
    cleanup(model, tok)
    return acc, correct, total


def phase_raw_composition_mmlu(skeleton, base_acc, scale):
    """Phase 2: Raw LoRA composition N=5 at given scale."""
    log("\n" + "=" * 60)
    log(f"Phase 2: Raw LoRA composition N=5 at scale={scale}")
    log("=" * 60)

    adapter_bs = []
    loaded = []
    for domain in DOMAINS:
        path = ADAPTER_DIR / domain / "adapter.npz"
        if path.exists():
            adapter_bs.append(dict(mx.load(str(path))))
            loaded.append(domain)
    log(f"  Loaded {len(adapter_bs)} adapters: {loaded}")

    if len(adapter_bs) < 2:
        log("  ERROR: need >= 2 adapters")
        return None

    composed = compose_adapters(adapter_bs)
    # Free individual adapters
    for ab in adapter_bs:
        del ab
    gc.collect()

    model, tok = load(MODEL_ID)
    n_mod = attach_composed_adapter(model, skeleton, composed, scale)
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    acc = correct / total if total else 0
    deg = (acc - base_acc) * 100
    log(f"  Raw N=5 scale={scale}: {acc:.1%} ({deg:+.1f}pp vs base, {n_mod} modules)")
    log_memory("post-raw-composition")
    cleanup(model, tok, composed)
    return {"accuracy": acc, "correct": correct, "total": total,
            "degradation_pp": round(deg, 2), "n_adapters": len(loaded),
            "per_subject": per_subject}


def phase_svd_composition_mmlu(base_acc, rank_suffix="r4"):
    """Phase 3: SVD-truncated composition N=5."""
    log("\n" + "=" * 60)
    log(f"Phase 3: SVD {rank_suffix} composition N=5")
    log("=" * 60)

    experts = {}
    for domain in DOMAINS:
        expert = load_svd_expert(domain, SVD_DIR, rank_suffix)
        if expert:
            experts[domain] = expert
            log(f"  Loaded SVD expert: {domain} ({len(expert)} modules)")
        else:
            log(f"  MISSING SVD expert: {domain}")

    if len(experts) < 2:
        log("  ERROR: need >= 2 SVD experts")
        return None

    available = [experts[d] for d in DOMAINS if d in experts]
    composed = compose_svd_experts(available)

    # Measure composed perturbation norm (sample)
    sample_keys = list(composed.keys())[:5]
    norms = []
    for k in sample_keys:
        A, B = composed[k]
        n = mx.linalg.norm((A.astype(mx.float32) @ B.astype(mx.float32)).reshape(-1))
        mx.eval(n)
        norms.append(n.item())
    mean_norm = sum(norms) / len(norms) if norms else 0
    log(f"  Composed SVD delta norm (first 5 modules): {mean_norm:.4f}")

    model, tok = load(MODEL_ID)
    n_mod = inject_svd_expert(model, composed)
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    acc = correct / total if total else 0
    deg = (acc - base_acc) * 100
    log(f"  SVD {rank_suffix} composed N=5: {acc:.1%} ({deg:+.1f}pp vs base, {n_mod} modules)")
    log_memory("post-svd-composition")
    cleanup(model, tok, composed)
    experts.clear()
    gc.collect()
    mx.clear_cache()
    return {"accuracy": acc, "correct": correct, "total": total,
            "degradation_pp": round(deg, 2), "n_experts": len(available),
            "delta_norm_sample": round(mean_norm, 6),
            "per_subject": per_subject}


def phase_svd_rank1_composition(base_acc, skeleton):
    """Phase 4: SVD rank=1 composition -- compute on-the-fly from raw adapters.

    Since rank=1 experts are not pre-saved, we extract them from raw adapters
    and skeleton, then compose via NRE averaging of SVD factors.
    """
    log("\n" + "=" * 60)
    log("Phase 4: SVD rank=1 composition N=5 (on-the-fly extraction)")
    log("=" * 60)

    # We need to extract SVD rank=1 for each adapter
    experts = []
    for di, domain in enumerate(DOMAINS):
        adapter_path = ADAPTER_DIR / domain / "adapter.npz"
        if not adapter_path.exists():
            log(f"  SKIP {domain}: no adapter")
            continue

        adapter_b = dict(mx.load(str(adapter_path)))
        expert = {}
        for li in range(36):  # Qwen3-4B has 36 layers
            for key in TARGET_KEYS:
                bk = f"model.layers.{li}.{key}.lora_b"
                ak = f"layer_{li}_{key}_domain_{di}"
                if bk not in adapter_b or ak not in skeleton:
                    continue
                A = mx.array(skeleton[ak]).astype(mx.float32)  # (in, rank)
                B = adapter_b[bk].astype(mx.float32)           # (rank, out)
                M = LORA_SCALE * B.T  # (out, lora_rank)
                N = A                  # (in, lora_rank)

                # Efficient SVD via QR + small SVD
                Q_m, R_m = mx.linalg.qr(M, stream=mx.cpu)
                Q_n, R_n = mx.linalg.qr(N, stream=mx.cpu)
                mx.eval(Q_m, R_m, Q_n, R_n)
                small = R_m @ R_n.T
                U_s, S, Vt_s = mx.linalg.svd(small, stream=mx.cpu)
                mx.eval(U_s, S, Vt_s)

                # Rank 1 truncation
                r = 1
                U_full = Q_m @ U_s[:, :r]
                Vt_full = Vt_s[:r, :] @ Q_n.T
                S_r = S[:r]
                sqrt_S = mx.sqrt(S_r)
                A_svd = (Vt_full.T * sqrt_S[None, :])
                B_svd = (U_full * sqrt_S[None, :]).T
                mx.eval(A_svd, B_svd)
                expert[f"layer_{li}_{key}"] = (A_svd.astype(mx.bfloat16), B_svd.astype(mx.bfloat16))
                del Q_m, R_m, Q_n, R_n, small, U_s, S, Vt_s, U_full, Vt_full, S_r, sqrt_S, A_svd, B_svd

        experts.append(expert)
        log(f"  Extracted SVD rank=1 for {domain} ({len(expert)} modules)")
        del adapter_b
        gc.collect()
        mx.clear_cache()

    if len(experts) < 2:
        log("  ERROR: need >= 2 experts")
        return None

    composed = compose_svd_experts(experts)

    model, tok = load(MODEL_ID)
    n_mod = inject_svd_expert(model, composed)
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    acc = correct / total if total else 0
    deg = (acc - base_acc) * 100
    log(f"  SVD rank=1 composed N=5: {acc:.1%} ({deg:+.1f}pp vs base, {n_mod} modules)")
    log_memory("post-svd-r1-composition")
    cleanup(model, tok, composed)
    for e in experts:
        del e
    gc.collect()
    mx.clear_cache()
    return {"accuracy": acc, "correct": correct, "total": total,
            "degradation_pp": round(deg, 2), "n_experts": len(experts),
            "per_subject": per_subject}


def phase_scale_reduced_composition(skeleton, base_acc, scale):
    """Phase 5/6: Full-rank composition at reduced scale."""
    log("\n" + "=" * 60)
    log(f"Phase: Full-rank N=5 at scale={scale}")
    log("=" * 60)

    adapter_bs = []
    loaded = []
    for domain in DOMAINS:
        path = ADAPTER_DIR / domain / "adapter.npz"
        if path.exists():
            adapter_bs.append(dict(mx.load(str(path))))
            loaded.append(domain)
    log(f"  Loaded {len(adapter_bs)} adapters: {loaded}")

    if len(adapter_bs) < 2:
        return None

    composed = compose_adapters(adapter_bs)
    for ab in adapter_bs:
        del ab
    gc.collect()

    model, tok = load(MODEL_ID)
    n_mod = attach_composed_adapter(model, skeleton, composed, scale)
    correct, total, per_subject = mmlu_eval(model, tok, MMLU_QUESTIONS)
    acc = correct / total if total else 0
    deg = (acc - base_acc) * 100
    log(f"  Full-rank N=5 scale={scale}: {acc:.1%} ({deg:+.1f}pp vs base, {n_mod} modules)")
    log_memory(f"post-fullrank-scale{scale}")
    cleanup(model, tok, composed)
    return {"accuracy": acc, "correct": correct, "total": total,
            "degradation_pp": round(deg, 2), "n_adapters": len(loaded),
            "per_subject": per_subject}


def phase_domain_ppl(skeleton, base_acc, configs):
    """Phase 7: Domain PPL for key configurations to check K838."""
    log("\n" + "=" * 60)
    log("Phase 7: Domain PPL for composed configurations")
    log("=" * 60)

    val_texts = {}
    for d in DOMAINS:
        try:
            val_texts[d] = load_data(d, "valid", 10)
        except FileNotFoundError:
            log(f"  No validation data for {d}")

    results = {}
    for config_name, config_fn in configs:
        log(f"\n  --- {config_name} ---")
        ppl_results = {}

        if config_name == "base":
            model, tok = load(MODEL_ID)
        elif config_name == "raw_scale20":
            adapter_bs = []
            for domain in DOMAINS:
                path = ADAPTER_DIR / domain / "adapter.npz"
                if path.exists():
                    adapter_bs.append(dict(mx.load(str(path))))
            composed = compose_adapters(adapter_bs)
            for ab in adapter_bs:
                del ab
            model, tok = load(MODEL_ID)
            attach_composed_adapter(model, skeleton, composed, LORA_SCALE)
            del composed
        elif config_name == "svd_r4":
            experts = []
            for domain in DOMAINS:
                e = load_svd_expert(domain, SVD_DIR, "r4")
                if e:
                    experts.append(e)
            composed = compose_svd_experts(experts)
            model, tok = load(MODEL_ID)
            inject_svd_expert(model, composed)
            del composed
            for e in experts:
                del e
        else:
            continue

        for d in DOMAINS:
            if d not in val_texts:
                continue
            domain_ppl = ppl(model, tok, val_texts[d])
            ppl_results[d] = round(domain_ppl, 3)
            log(f"    {d}: {domain_ppl:.3f}")

        cleanup(model, tok)
        results[config_name] = ppl_results

    return results


# ── Main ───────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if hasattr(o, 'item'): return o.item()
        return super().default(o)


def main():
    t0 = time.time()
    exp = Experiment("solidified_composition_mmlu", dir=str(EXPERIMENT_DIR))
    mx.random.seed(SEED)

    exp.metric("model_id", MODEL_ID, display=True)

    # Load skeleton (needed for raw LoRA and rank=1 SVD)
    skeleton = dict(np.load(str(SKELETON_PATH)))
    log(f"Skeleton loaded: {len(skeleton)} A-matrices")

    # Phase 1: Base MMLU
    base_acc, base_c, base_t = phase_base_mmlu()
    exp.metric("base_mmlu", base_acc)
    exp.metric("base_correct", base_c)

    # Phase 2: Raw LoRA composition N=5 at scale=20 (replication of Finding #320)
    raw_20 = phase_raw_composition_mmlu(skeleton, base_acc, LORA_SCALE)
    if raw_20:
        exp.metric("raw_scale20_mmlu", raw_20["accuracy"])
        exp.metric("raw_scale20_degradation_pp", raw_20["degradation_pp"])

    # Phase 3: SVD rank=4 composition N=5 (THE KEY MEASUREMENT)
    svd_r4 = phase_svd_composition_mmlu(base_acc, "r4")
    if svd_r4:
        exp.metric("svd_r4_composed_mmlu", svd_r4["accuracy"])
        exp.metric("svd_r4_composed_degradation_pp", svd_r4["degradation_pp"])

    # Phase 4: SVD rank=1 composition N=5 (most aggressive)
    svd_r1 = phase_svd_rank1_composition(base_acc, skeleton)
    if svd_r1:
        exp.metric("svd_r1_composed_mmlu", svd_r1["accuracy"])
        exp.metric("svd_r1_composed_degradation_pp", svd_r1["degradation_pp"])

    # Phase 5: Full-rank N=5 at scale=13 (energy-matched control, Finding #326)
    full_13 = phase_scale_reduced_composition(skeleton, base_acc, 13.0)
    if full_13:
        exp.metric("full_scale13_mmlu", full_13["accuracy"])
        exp.metric("full_scale13_degradation_pp", full_13["degradation_pp"])

    # Phase 6: Full-rank N=5 at scale=5 (known safe control)
    full_5 = phase_scale_reduced_composition(skeleton, base_acc, 5.0)
    if full_5:
        exp.metric("full_scale5_mmlu", full_5["accuracy"])
        exp.metric("full_scale5_degradation_pp", full_5["degradation_pp"])

    # Phase 7: Domain PPL for key configs (K838 check)
    ppl_configs = [("base", None), ("raw_scale20", None), ("svd_r4", None)]
    ppl_results = phase_domain_ppl(skeleton, base_acc, ppl_configs)
    for config_name, ppls in ppl_results.items():
        for d, v in ppls.items():
            exp.metric(f"ppl_{config_name}_{d}", v)

    # ── Kill criteria assessment ──────────────────────────────────────────

    # K837: MMLU degradation > 15pp
    # For SVD composition: compare degradation to single SVD adapter (-30pp)
    # The kill is if SVD COMPOSITION is worse than we're trying to solve
    svd_r4_deg = svd_r4["degradation_pp"] if svd_r4 else -999
    # K837 checks: is the SVD composition degradation worse than 15pp?
    # Note: degradation is negative (e.g., -30pp), so we check < -15
    k837_pass = svd_r4_deg > -15.0  # degradation better than -15pp
    exp.results["kill_criteria"]["K837"] = {
        "pass": k837_pass,
        "detail": f"SVD r4 composed degradation: {svd_r4_deg:.1f}pp (threshold: >-15pp)",
    }
    log(f"\nK837: SVD r4 composed degradation {svd_r4_deg:.1f}pp {'PASS' if k837_pass else 'FAIL'}")

    # K838: Domain quality < 50% of raw LoRA
    # Compare PPL: SVD composed PPL should not be >2x worse than raw composed PPL
    k838_pass = True
    k838_detail = []
    if "raw_scale20" in ppl_results and "svd_r4" in ppl_results:
        for d in DOMAINS:
            raw_ppl = ppl_results["raw_scale20"].get(d, 999)
            svd_ppl = ppl_results["svd_r4"].get(d, 999)
            ratio = svd_ppl / raw_ppl if raw_ppl > 0 else 999
            k838_detail.append(f"{d}: svd={svd_ppl:.1f} raw={raw_ppl:.1f} ratio={ratio:.2f}")
            if ratio > 2.0:
                k838_pass = False
    else:
        k838_detail.append("PPL comparison incomplete")

    exp.results["kill_criteria"]["K838"] = {
        "pass": k838_pass,
        "detail": "; ".join(k838_detail),
    }
    log(f"K838: Domain quality check {'PASS' if k838_pass else 'FAIL'}")
    for d in k838_detail:
        log(f"  {d}")

    # ── Summary table ─────────────────────────────────────────────────────

    log("\n" + "=" * 60)
    log("SUMMARY: MMLU Degradation Under Composition")
    log("=" * 60)
    log(f"{'Configuration':<35} {'MMLU':>6} {'Degrad':>8} {'vs Pred':>10}")
    log("-" * 65)
    log(f"{'Base Qwen3-4B':<35} {base_acc:>5.1%} {'0pp':>8} {'(control)':>10}")
    if raw_20:
        log(f"{'Raw LoRA N=5 scale=20':<35} {raw_20['accuracy']:>5.1%} {raw_20['degradation_pp']:>+7.1f}pp {'~-44pp':>10}")
    if svd_r4:
        log(f"{'SVD r=4 composed N=5':<35} {svd_r4['accuracy']:>5.1%} {svd_r4['degradation_pp']:>+7.1f}pp {'-25 to -35':>10}")
    if svd_r1:
        log(f"{'SVD r=1 composed N=5':<35} {svd_r1['accuracy']:>5.1%} {svd_r1['degradation_pp']:>+7.1f}pp {'-17 to -27':>10}")
    if full_13:
        log(f"{'Full-rank N=5 scale=13':<35} {full_13['accuracy']:>5.1%} {full_13['degradation_pp']:>+7.1f}pp {'-25 to -35':>10}")
    if full_5:
        log(f"{'Full-rank N=5 scale=5':<35} {full_5['accuracy']:>5.1%} {full_5['degradation_pp']:>+7.1f}pp {'0 to -2':>10}")
    log("-" * 65)

    # S83 check
    best_svd_deg = min(
        abs(svd_r4["degradation_pp"]) if svd_r4 else 999,
        abs(svd_r1["degradation_pp"]) if svd_r1 else 999,
    )
    s83_pass = best_svd_deg < 5.0
    log(f"\nS83: Best SVD degradation {best_svd_deg:.1f}pp {'PASS (<5pp)' if s83_pass else 'FAIL (>=5pp)'}")

    # Check Theorem 3: scale-reduced ~= SVD composition
    if svd_r4 and full_13:
        diff = abs(svd_r4["degradation_pp"] - full_13["degradation_pp"])
        log(f"Theorem 3 check: SVD r4 vs scale=13 gap = {diff:.1f}pp (predicted <5pp)")

    # Save detailed results
    exp.results["configurations"] = {
        "base": {"accuracy": base_acc, "correct": base_c, "total": base_t},
        "raw_scale20": raw_20,
        "svd_r4_composed": svd_r4,
        "svd_r1_composed": svd_r1,
        "full_scale13": full_13,
        "full_scale5": full_5,
    }
    exp.results["ppl_comparison"] = ppl_results
    exp.results["total_time_s"] = round(time.time() - t0, 1)

    exp.save()
    log(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
