#!/usr/bin/env python3
"""
exp_p3_b0_medical_oe_adapter

Format-register aligned medical adapter: train on medalpaca/medical_meadow_wikidoc
(open-ended explanatory responses) and verify behavioral improvement > 80%.

Root cause of Finding #457 kill: MCQ training format shifts adapter toward concise
Q/A register. Open-ended wikidoc training matches the evaluation register, producing
positive delta_D via Format-Register Alignment (MATH.md Theorem 1).

Kill criteria:
  K1169: improvement_rate (vocabulary rubric) > 80% (vs 60% with MCQ, Finding #457)
  K1170: adapted_mean_vocab >= base_mean_vocab × 1.5
  K1171: MMLU medical regression < 5pp (quality preserved)

References:
  Finding #457 (MCQ format kills behavioral delta)
  Finding #217 (medical s=20 gives +17.9% behavioral in v1)
  medalpaca/medical_meadow_wikidoc (clinical explanatory responses)
  arxiv 2106.09685 (LoRA)
"""

import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR / "lora_adapter"
DATA_DIR = EXPERIMENT_DIR / "lora_data"
LORA_CONFIG = EXPERIMENT_DIR / "lora_config.yaml"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 6
LORA_SCALE = 6.0

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

N_TRAIN = 50 if IS_SMOKE else 500
N_VALID = 10 if IS_SMOKE else 50
N_EVAL = 5 if IS_SMOKE else 20
N_MMLU = 5 if IS_SMOKE else 20
TRAIN_ITERS = 50 if IS_SMOKE else 500

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)


# ── Vocabulary rubric ────────────────────────────────────────────────────────
# Exact 30-term glossary from Finding #457 (exp_p1_p0_behavioral_e2e) for
# direct comparison: base_mean≈1.4, MCQ adapter mean≈2.1, improvement_rate=60%.
MEDICAL_GLOSSARY = [
    "mechanism", "inhibitor", "receptor", "pharmacology", "clinical", "therapy",
    "diagnosis", "treatment", "pathophysiology", "enzyme", "protein",
    "antibody", "immune", "inflammation", "vascular", "cardiac", "neural",
    "medication", "dose", "adverse", "contraindicated", "efficacy", "etiology",
    "prognosis", "cytokine", "antibiotics", "prophylaxis", "comorbidity",
]

MEDICAL_QUERIES = [
    "Explain how ACE inhibitors work to treat hypertension.",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "Describe how beta-blockers work and their clinical uses.",
    "What are the main symptoms of myocardial infarction?",
    "Explain the role of the immune system in fighting bacterial infections.",
    "What is the significance of the blood-brain barrier?",
    "Describe how diuretics work to treat edema.",
    "What is the pathophysiology of asthma?",
    "Explain how vaccines provide immunity.",
    "What are the stages of wound healing?",
    "How does aspirin inhibit platelet aggregation?",
    "Describe the mechanism of action of statins.",
    "What is the difference between bacterial and viral infections?",
    "Explain how insulin resistance develops in Type 2 diabetes.",
    "Describe the pathophysiology of heart failure.",
    "What are the adverse effects of corticosteroids?",
    "Explain the mechanism of action of beta-lactam antibiotics.",
    "What is the role of cytokines in inflammation?",
    "Describe the clinical presentation of pneumonia.",
    "Explain how antihypertensive medications lower blood pressure.",
]

MMLU_MEDICAL_QUESTIONS = [
    ("A patient with chest pain and ST elevation in leads II, III, aVF. Most likely diagnosis?",
     "Inferior myocardial infarction"),
    ("Which enzyme is deficient in PKU (phenylketonuria)?",
     "Phenylalanine hydroxylase"),
    ("First-line treatment for uncomplicated UTI in women?",
     "Trimethoprim-sulfamethoxazole or nitrofurantoin"),
    ("Which cells produce insulin?",
     "Beta cells of the islets of Langerhans"),
    ("Mechanism of action of metformin?",
     "Inhibits hepatic gluconeogenesis, activates AMPK"),
    ("Classic triad of Meniere's disease?",
     "Episodic vertigo, tinnitus, sensorineural hearing loss"),
    ("Which vitamin deficiency causes Wernicke encephalopathy?",
     "Thiamine (Vitamin B1)"),
    ("First-line treatment for H. pylori eradication?",
     "Triple therapy: PPI + clarithromycin + amoxicillin"),
    ("The most common cause of community-acquired pneumonia?",
     "Streptococcus pneumoniae"),
    ("Mechanism of action of loop diuretics?",
     "Inhibit Na-K-2Cl cotransporter in thick ascending limb of loop of Henle"),
    ("What is the pathophysiology of type 1 diabetes mellitus?",
     "Autoimmune destruction of pancreatic beta cells leading to insulin deficiency"),
    ("How does heparin work as an anticoagulant?",
     "Activates antithrombin III, inhibiting thrombin and factor Xa"),
    ("Classic ECG finding in hyperkalemia?",
     "Peaked T waves, widened QRS, eventually sine wave pattern"),
    ("Mechanism of action of ACE inhibitors?",
     "Inhibit angiotensin-converting enzyme, preventing angiotensin I to II conversion"),
    ("What causes the symptoms of myasthenia gravis?",
     "Autoimmune antibodies against nicotinic acetylcholine receptors at NMJ"),
    ("First-line treatment for anaphylaxis?",
     "Epinephrine (adrenaline) intramuscular injection"),
    ("Which cancer is associated with asbestos exposure?",
     "Mesothelioma"),
    ("Mechanism of action of SSRIs?",
     "Selectively inhibit serotonin reuptake, increasing synaptic serotonin"),
    ("Classic presentation of appendicitis?",
     "Periumbilical pain migrating to right iliac fossa, nausea, fever, rebound tenderness"),
    ("What is the mechanism of resistance to penicillin in MRSA?",
     "Altered penicillin-binding protein (PBP2a) encoded by mecA gene"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"  [MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def score_vocabulary(text: str) -> int:
    """Count medical glossary terms in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for term in MEDICAL_GLOSSARY if term.lower() in text_lower)


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 300) -> str:
    """Generate a response using mlx_lm.generate."""
    from mlx_lm import generate
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)


def parse_response_for_eval(text: str) -> str:
    """Strip Gemma 4 thinking blocks and return clean response."""
    clean = re.sub(r"<\|channel>thought.*?</?\|channel>thought>", "", text,
                   flags=re.DOTALL | re.IGNORECASE)
    return clean.strip()


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_wikidoc_data(data_dir: Path, n_train: int, n_valid: int) -> None:
    """Load medalpaca wikidoc and format as instruction-following JSONL."""
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    test_path = data_dir / "test.jsonl"

    log(f"Loading medalpaca/medical_meadow_wikidoc (n_train={n_train}, n_valid={n_valid})...")
    ds = load_dataset("medalpaca/medical_meadow_wikidoc", split="train", streaming=True)

    # Collect examples: filter for explanatory responses (len > 100)
    # Use messages format so mask_prompt=True works correctly (masks the user turn)
    examples = []
    for row in ds:
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if len(out) > 100 and len(inp) > 10:
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Answer this medical question with a clinical explanation:\n\n{inp}"},
                    {"role": "assistant", "content": out},
                ]
            })
        if len(examples) >= n_train + n_valid + 50:
            break

    if len(examples) < n_train + n_valid:
        log(f"WARNING: only {len(examples)} examples found (requested {n_train + n_valid})")
        n_train = max(len(examples) - n_valid, 10)
        n_valid = min(n_valid, len(examples) - n_train)

    import random
    random.seed(42)
    random.shuffle(examples)

    train_examples = examples[:n_train]
    valid_examples = examples[n_train:n_train + n_valid]

    # Sanity check: count glossary density in training data
    all_text = " ".join(
        e["messages"][1]["content"] for e in train_examples[:20]
    )
    density = score_vocabulary(all_text) / max(len(all_text.split()), 1)
    log(f"  Training data glossary density: {density:.4f} (terms/word)")
    log(f"  Expected G_density >= 0.10 for positive delta_D signal")

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(valid_path, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")
    # Test file is same as valid (not used for training)
    with open(test_path, "w") as f:
        for ex in valid_examples[:5]:
            f.write(json.dumps(ex) + "\n")

    log(f"  Wrote {len(train_examples)} train, {len(valid_examples)} valid examples")


def write_lora_config(config_path: Path, data_dir: Path, adapter_dir: Path) -> None:
    """Write LoRA training config yaml."""
    config = f"""model: {MODEL_ID}
train: true
data: {data_dir}
adapter_path: {adapter_dir}
seed: 42
lora_parameters:
  rank: {LORA_RANK}
  scale: {LORA_SCALE}
  dropout: 0.0
  keys:
    - self_attn.q_proj
num_layers: -1
batch_size: 2
iters: {TRAIN_ITERS}
val_batches: 5
steps_per_report: 100
steps_per_eval: {TRAIN_ITERS}
save_every: {TRAIN_ITERS}
max_seq_length: 512
grad_checkpoint: true
mask_prompt: true
learning_rate: 1e-4
"""
    config_path.write_text(config)


# ── Training ─────────────────────────────────────────────────────────────────

def train_adapter(config_path: Path, adapter_dir: Path) -> dict:
    """Run mlx_lm.lora training via subprocess."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cmd = ["uv", "run", "python", "-m", "mlx_lm.lora", "--config", str(config_path)]
    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=EXPERIMENT_DIR.parent.parent.parent)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  ERROR: lora training failed (exit {result.returncode})")
        log(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"LoRA training failed: {result.stderr[-500:]}")

    # Parse final train loss from stdout
    train_loss = None
    val_loss = None
    for line in result.stdout.split("\n"):
        if "Train loss" in line or "train_loss" in line.lower():
            try:
                train_loss = float(re.search(r"[\d.]+", line.split(":")[-1]).group())
            except Exception:
                pass
        if "Val loss" in line or "val_loss" in line.lower():
            try:
                val_loss = float(re.search(r"[\d.]+", line.split(":")[-1]).group())
            except Exception:
                pass

    log(f"  Training complete: {elapsed:.1f}s, train_loss={train_loss}, val_loss={val_loss}")
    return {"train_loss": train_loss, "val_loss": val_loss, "elapsed_s": elapsed}


# ── Evaluation ───────────────────────────────────────────────────────────────

def eval_vocabulary_improvement(
    model_base, model_adapted, tokenizer, queries: list, max_tokens: int = 300
) -> dict:
    """
    Evaluate vocabulary improvement: compare base vs adapted on open-ended medical queries.
    Returns improvement_rate and mean vocab scores.
    """
    results = []
    for i, query in enumerate(queries):
        log(f"  Query {i+1}/{len(queries)}: {query[:60]}...")
        t0 = time.time()
        resp_base = generate_response(model_base, tokenizer, query, max_tokens)
        t1 = time.time()
        resp_adapted = generate_response(model_adapted, tokenizer, query, max_tokens)
        t2 = time.time()

        score_base = score_vocabulary(resp_base)
        score_adapted = score_vocabulary(resp_adapted)
        improved = score_adapted > score_base

        log(f"    Base: {score_base} terms | Adapted: {score_adapted} terms | Improved: {improved}")
        log(f"    Latency: base={t1-t0:.2f}s adapted={t2-t1:.2f}s")

        results.append({
            "query": query,
            "score_base": score_base,
            "score_adapted": score_adapted,
            "improved": improved,
            "resp_base_excerpt": resp_base[:200],
            "resp_adapted_excerpt": resp_adapted[:200],
        })

    improvement_rate = sum(r["improved"] for r in results) / len(results)
    mean_base = sum(r["score_base"] for r in results) / len(results)
    mean_adapted = sum(r["score_adapted"] for r in results) / len(results)
    vocab_ratio = mean_adapted / max(mean_base, 0.1)

    return {
        "improvement_rate": improvement_rate,
        "mean_score_base": mean_base,
        "mean_score_adapted": mean_adapted,
        "vocab_ratio": vocab_ratio,
        "results": results,
    }


def eval_mmlu_regression(
    model_base, model_adapted, tokenizer, n_questions: int
) -> dict:
    """Check MMLU-style medical accuracy regression."""
    questions = MMLU_MEDICAL_QUESTIONS[:n_questions]
    base_correct = 0
    adapted_correct = 0

    for question, expected_key in questions:
        prompt = f"Answer this medical question briefly: {question}"
        resp_base = generate_response(model_base, tokenizer, prompt, max_tokens=100)
        resp_adapted = generate_response(model_adapted, tokenizer, prompt, max_tokens=100)

        # Simple keyword check: does response contain key terms from expected answer?
        expected_terms = [t.lower() for t in expected_key.split() if len(t) > 3]
        base_hits = sum(1 for t in expected_terms if t in resp_base.lower())
        adapted_hits = sum(1 for t in expected_terms if t in resp_adapted.lower())

        if base_hits >= len(expected_terms) // 2:
            base_correct += 1
        if adapted_hits >= len(expected_terms) // 2:
            adapted_correct += 1

    base_acc = base_correct / len(questions)
    adapted_acc = adapted_correct / len(questions)
    regression = base_acc - adapted_acc  # positive = regression, negative = improvement

    log(f"  MMLU: base={base_acc:.1%} adapted={adapted_acc:.1%} regression={regression:+.1%}")
    return {
        "base_acc": base_acc,
        "adapted_acc": adapted_acc,
        "regression_pp": regression * 100,
        "n_questions": len(questions),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log(f"\n{'='*60}")
    log(f"exp_p3_b0_medical_oe_adapter")
    log(f"IS_SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_MMLU={N_MMLU}")
    log(f"{'='*60}")

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "train_iters": TRAIN_ITERS,
    }

    # ── Phase 1: Prepare training data ───────────────────────────────────────
    log("\n=== Phase 1: Prepare wikidoc training data ===")
    prepare_wikidoc_data(DATA_DIR, N_TRAIN, N_VALID)
    write_lora_config(LORA_CONFIG, DATA_DIR, ADAPTER_DIR)
    log("Data preparation complete.")

    # ── Phase 2: Train OE adapter ─────────────────────────────────────────────
    log("\n=== Phase 2: Train medical OE adapter ===")
    train_result = train_adapter(LORA_CONFIG, ADAPTER_DIR)
    results["training"] = train_result

    # ── Phase 3: Load models ──────────────────────────────────────────────────
    log("\n=== Phase 3: Load base and adapted models ===")
    from mlx_lm import load

    log("Loading base model...")
    log_memory("before-base")
    model_base, tokenizer = load(MODEL_ID)
    mx.eval(model_base.parameters())
    log_memory("after-base")

    log("Loading adapted model...")
    model_adapted, _ = load(MODEL_ID, adapter_path=str(ADAPTER_DIR))
    mx.eval(model_adapted.parameters())
    log_memory("after-adapted")

    # ── Phase 4: Evaluate vocabulary improvement ──────────────────────────────
    log("\n=== Phase 4: Evaluate vocabulary improvement ===")
    queries = MEDICAL_QUERIES[:N_EVAL]
    vocab_result = eval_vocabulary_improvement(
        model_base, model_adapted, tokenizer, queries
    )
    results["vocabulary"] = vocab_result

    improvement_rate = vocab_result["improvement_rate"]
    mean_base = vocab_result["mean_score_base"]
    mean_adapted = vocab_result["mean_score_adapted"]
    vocab_ratio = vocab_result["vocab_ratio"]

    log(f"\nVocabulary Results:")
    log(f"  improvement_rate = {improvement_rate:.1%} (K1169 threshold: > 80%)")
    log(f"  mean_base = {mean_base:.2f}, mean_adapted = {mean_adapted:.2f}")
    log(f"  vocab_ratio = {vocab_ratio:.3f} (K1170 threshold: >= 1.5)")

    k1169_pass = improvement_rate > 0.80
    k1170_pass = vocab_ratio >= 1.5

    log(f"  K1169 ({'PASS' if k1169_pass else 'FAIL'}): improvement_rate={improvement_rate:.1%}")
    log(f"  K1170 ({'PASS' if k1170_pass else 'FAIL'}): vocab_ratio={vocab_ratio:.3f}")

    # ── Phase 5: MMLU regression check ───────────────────────────────────────
    log("\n=== Phase 5: MMLU regression check ===")
    mmlu_result = eval_mmlu_regression(model_base, model_adapted, tokenizer, N_MMLU)
    results["mmlu"] = mmlu_result

    regression = mmlu_result["regression_pp"]
    k1171_pass = regression < 5.0

    log(f"  K1171 ({'PASS' if k1171_pass else 'FAIL'}): MMLU regression={regression:.1f}pp (threshold: < 5pp)")

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("KILL CRITERIA SUMMARY")
    log(f"{'='*60}")
    log(f"K1169: improvement_rate={improvement_rate:.1%} (>80%) — {'PASS' if k1169_pass else 'FAIL'}")
    log(f"K1170: vocab_ratio={vocab_ratio:.3f} (>=1.5) — {'PASS' if k1170_pass else 'FAIL'}")
    log(f"K1171: MMLU_regression={regression:.1f}pp (<5pp) — {'PASS' if k1171_pass else 'FAIL'}")

    overall = "SUPPORTED" if (k1169_pass and k1170_pass and k1171_pass) else "KILLED"
    log(f"\nOverall: {overall}")

    results.update({
        "k1169_pass": k1169_pass,
        "k1170_pass": k1170_pass,
        "k1171_pass": k1171_pass,
        "overall": overall,
    })

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()
