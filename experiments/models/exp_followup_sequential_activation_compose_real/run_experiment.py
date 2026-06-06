#!/usr/bin/env python3
"""
exp_followup_sequential_activation_compose_real

Model-level sequential generation pipeline:
    h(x) = personal_forward( domain_forward( base_forward(x) ) )

with base_forward = identity (user prompt passes through unchanged to
stage 1). Stage 1 uses the math (domain) adapter; stage 2 uses the
personal (style) adapter and is prompted with the original question
plus stage-1's reply as "Initial reply" context.

See MATH.md §Theorem 1 for why weight/activation-space sequential on
q_proj is architecturally infeasible and why this is the only sound
interpretation of the title.

KCs (locked pre-run):
  K1563a: pipeline_style_rate >= 24.0%  (additive baseline, parent §1)
  K1563b: pipeline_mcq_acc    >= 15.0%  (additive baseline, parent §1)
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

MATH_ADAPTER_DIR = (
    EXPERIMENT_DIR.parent
    / "exp_p1_t2_single_domain_training"
    / "adapters"
    / "math"
)
PERSONAL_ADAPTER_DIR = (
    EXPERIMENT_DIR.parent
    / "exp_p1_t5_user_local_training"
    / "personal_adapter"
)

IS_SMOKE = os.environ.get("SMOKE_TEST", "1") == "1"  # default SMOKE
N_STYLE = 5 if IS_SMOKE else 25
N_MATH = 5 if IS_SMOKE else 20
SEED = 42

MAX_TOK_STAGE1 = 128
MAX_TOK_STAGE2 = 128

OPTION_LETTERS = ["A", "B", "C", "D"]
PREFERENCE_MARKER = "Hope that helps, friend!"

STYLE_QUESTIONS = [
    "What is gravity?",
    "How do computers process information?",
    "What is electricity?",
    "How does sound travel?",
    "What is chemistry?",
    "How do magnets work?",
    "What is the Big Bang?",
    "How does the brain work?",
    "What is renewable energy?",
    "How do tides affect marine life?",
    "What is a virus?",
    "How does temperature affect matter?",
    "What is atmospheric pressure?",
    "How does a telescope work?",
    "What is genetic engineering?",
    "How do crystals form?",
    "What is electrical resistance?",
    "How does radar work?",
    "What is ocean acidification?",
    "How does a microchip work?",
    "What is radioactivity?",
    "How do ecosystems balance themselves?",
    "What is the ozone layer?",
    "How do languages evolve?",
    "What is thermodynamics?",
]


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
        mx.reset_peak_memory()
    except Exception:
        pass


def log_memory(label=""):
    try:
        import mlx.core as mx

        active = mx.get_active_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)
    except Exception:
        pass


def strip_thinking_block(text: str) -> str:
    stripped = re.sub(r"<\|channel\>thought.*?</\|channel\>thought>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def extract_letter(clean_text: str) -> str | None:
    clean = clean_text.upper()
    for letter in OPTION_LETTERS:
        if clean.startswith(letter):
            return letter
    m = re.search(r"\b([ABCD])\b", clean)
    return m.group(1) if m else None


# ─── Phase 0: shape verification (Thm 1) ─────────────────────────────────────

def verify_shapes() -> dict:
    """Assert dim mismatch d_out(q_proj)=2048 ≠ d_in(q_proj)=2560.

    Requires at least one adapter to be readable. If the parent math
    adapter artefact is missing (K_vacate condition — artefact gitignored,
    not recoverable without retraining), verify Thm 1 from the personal
    adapter alone, which lives on the same projection (q_proj) and has
    the same d_in, d_out.
    """
    import safetensors.numpy as stn

    math_path = MATH_ADAPTER_DIR / "adapters.safetensors"
    pers_path = PERSONAL_ADAPTER_DIR / "adapters.safetensors"

    math_available = math_path.exists()
    pers_available = pers_path.exists()

    key_a = "language_model.model.layers.30.self_attn.q_proj.lora_a"
    key_b = "language_model.model.layers.30.self_attn.q_proj.lora_b"

    info: dict = {
        "math_adapter_available": math_available,
        "personal_adapter_available": pers_available,
    }

    if pers_available:
        pers_d = stn.load_file(str(pers_path))
        la_P = pers_d[key_a]
        lb_P = pers_d[key_b]
        d_in_P, r_P = la_P.shape
        r_P2, d_out_P = lb_P.shape
        assert r_P == r_P2
        assert d_in_P == 2560, f"Gemma 4 E4B hidden_size = 2560, got {d_in_P}"
        assert d_out_P == 2048, f"Gemma 4 E4B q_proj out = 2048, got {d_out_P}"
        assert d_out_P != d_in_P
        info.update({
            "d_in": int(d_in_P),
            "d_out": int(d_out_P),
            "r_P": int(r_P),
            "thm1_verified_from_personal": True,
        })
        print(
            f"Thm 1 verified via personal adapter: "
            f"d_in={d_in_P}, d_out={d_out_P} → d_out != d_in → "
            f"weight/activation sequential infeasible",
            flush=True,
        )

    if math_available:
        math_d = stn.load_file(str(math_path))
        la_D = math_d[key_a]
        lb_D = math_d[key_b]
        d_in_D, r_D = la_D.shape
        _, d_out_D = lb_D.shape
        info.update({
            "r_D": int(r_D),
            "thm1_verified_from_math": True,
        })

    info["thm1_verified"] = pers_available  # personal alone suffices
    return info


# ─── Generation helpers ──────────────────────────────────────────────────────

def format_chat(tokenizer, content: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def format_mcq_prompt(ex: dict) -> str:
    formatted_q = (
        f"{ex['question']}\n"
        + "\n".join(
            f"({OPTION_LETTERS[k]}) {ex['choices'][k]}"
            for k in range(len(ex["choices"]))
        )
    )
    return (
        "Answer this multiple choice question. "
        "Respond with only the letter (A/B/C/D).\n\n" + formatted_q
    )


# ─── Phase 1 baselines ──────────────────────────────────────────────────────

def eval_style_single_stage(adapter_path: Path, n_eval: int, label: str) -> float:
    from mlx_lm import generate, load

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory(f"style-{label}")

    compliant = 0
    for i, q in enumerate(STYLE_QUESTIONS[:n_eval]):
        prompt = format_chat(tokenizer, q)
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOK_STAGE2, verbose=False)
        hit = PREFERENCE_MARKER in resp
        if hit:
            compliant += 1
        if i < 3:
            preview = resp[:80].replace("\n", " ")
            print(f"  q{i}: {'✓' if hit else '✗'} '{preview}...'", flush=True)

    rate = compliant / n_eval * 100
    print(f"Style[{label}]: {compliant}/{n_eval} = {rate:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return rate


def eval_mcq_single_stage(adapter_path: Path, n_eval: int, label: str) -> float:
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory(f"mcq-{label}")

    correct = 0
    for i, ex in enumerate(ds):
        prompt = format_chat(tokenizer, format_mcq_prompt(ex))
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOK_STAGE2, verbose=False)
        clean = strip_thinking_block(resp)
        pred = extract_letter(clean)
        gt = OPTION_LETTERS[ex["answer"]]
        ok = pred == gt
        if ok:
            correct += 1
        if i < 3:
            print(f"  q{i}: gt={gt}, pred={pred}, {'✓' if ok else '✗'}", flush=True)

    acc = correct / len(ds) * 100
    print(f"MCQ[{label}]: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


# ─── Phase 2: sequential pipeline ───────────────────────────────────────────

def run_sequential_pipeline_style(n_eval: int) -> tuple[float, list]:
    """h(x) = personal_forward(domain_forward(base_forward(x))) for style task."""
    from mlx_lm import generate, load

    questions = STYLE_QUESTIONS[:n_eval]

    # Stage 1: domain forward
    print("\n  Stage 1: domain_forward(base_forward(x))", flush=True)
    model, tok = load(MODEL_ID, adapter_path=str(MATH_ADAPTER_DIR))
    log_memory("seq-style-stage1")
    stage1_responses = []
    for i, q in enumerate(questions):
        prompt = format_chat(tok, q)
        resp = generate(model, tok, prompt=prompt, max_tokens=MAX_TOK_STAGE1, verbose=False)
        stage1_responses.append(resp.strip())
        if i < 2:
            preview = resp[:60].replace("\n", " ")
            print(f"    stage1[{i}]: '{preview}...'", flush=True)
    cleanup(model, tok)

    # Stage 2: personal forward on stage-1 outputs
    print("  Stage 2: personal_forward(stage1_output)", flush=True)
    model, tok = load(MODEL_ID, adapter_path=str(PERSONAL_ADAPTER_DIR))
    log_memory("seq-style-stage2")
    compliant = 0
    records = []
    for i, (q, r1) in enumerate(zip(questions, stage1_responses)):
        refined_prompt_body = (
            f"{q}\n\nInitial reply: {r1}\n\nRefined reply:"
        )
        prompt = format_chat(tok, refined_prompt_body)
        resp = generate(model, tok, prompt=prompt, max_tokens=MAX_TOK_STAGE2, verbose=False)
        hit = PREFERENCE_MARKER in resp
        if hit:
            compliant += 1
        records.append({
            "q": q,
            "stage1": r1[:200],
            "stage2": resp[:200],
            "hit": hit,
        })
        if i < 2:
            preview = resp[:60].replace("\n", " ")
            print(f"    stage2[{i}]: {'✓' if hit else '✗'} '{preview}...'", flush=True)
    cleanup(model, tok)

    rate = compliant / n_eval * 100
    print(f"Pipeline style: {compliant}/{n_eval} = {rate:.1f}%", flush=True)
    return rate, records


def run_sequential_pipeline_mcq(n_eval: int) -> tuple[float, list]:
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))
    ex_list = [{"q": ex["question"], "choices": ex["choices"], "answer": ex["answer"]} for ex in ds]

    print("\n  Stage 1: domain_forward(base_forward(x))", flush=True)
    model, tok = load(MODEL_ID, adapter_path=str(MATH_ADAPTER_DIR))
    log_memory("seq-mcq-stage1")
    stage1 = []
    for i, ex in enumerate(ex_list):
        prompt = format_chat(tok, format_mcq_prompt(ex))
        resp = generate(model, tok, prompt=prompt, max_tokens=MAX_TOK_STAGE1, verbose=False)
        stage1.append(resp.strip())
        if i < 2:
            clean = strip_thinking_block(resp)[:80].replace("\n", " ")
            print(f"    stage1[{i}]: '{clean}...'", flush=True)
    cleanup(model, tok)

    print("  Stage 2: personal_forward(stage1_output)", flush=True)
    model, tok = load(MODEL_ID, adapter_path=str(PERSONAL_ADAPTER_DIR))
    log_memory("seq-mcq-stage2")
    correct = 0
    records = []
    for i, (ex, r1) in enumerate(zip(ex_list, stage1)):
        refined = (
            f"{format_mcq_prompt(ex)}\n\nInitial reply: {r1}\n\n"
            "Refined reply (letter only):"
        )
        prompt = format_chat(tok, refined)
        resp = generate(model, tok, prompt=prompt, max_tokens=MAX_TOK_STAGE2, verbose=False)
        clean = strip_thinking_block(resp)
        pred = extract_letter(clean)
        gt = OPTION_LETTERS[ex["answer"]]
        ok = pred == gt
        if ok:
            correct += 1
        records.append({
            "q": ex["q"][:120],
            "stage1": r1[:120],
            "stage2": resp[:120],
            "pred": pred,
            "gt": gt,
            "ok": ok,
        })
        if i < 2:
            print(f"    stage2[{i}]: gt={gt}, pred={pred}, {'✓' if ok else '✗'}", flush=True)
    cleanup(model, tok)

    acc = correct / len(ex_list) * 100
    print(f"Pipeline MCQ: {correct}/{len(ex_list)} = {acc:.1f}%", flush=True)
    return acc, records


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    import mlx.core as mx

    mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    t0 = time.time()
    results: dict = {
        "is_smoke": IS_SMOKE,
        "n_style": N_STYLE,
        "n_math": N_MATH,
        "seed": SEED,
    }

    print(
        f"=== exp_followup_sequential_activation_compose_real "
        f"({'SMOKE' if IS_SMOKE else 'FULL'}, N_style={N_STYLE}, N_math={N_MATH}) ===",
        flush=True,
    )

    # Phase 0
    print("\n--- Phase 0: Thm 1 shape verification ---", flush=True)
    shape_info = verify_shapes()
    results["phase0_shapes"] = shape_info

    math_available = shape_info.get("math_adapter_available", False)
    pers_available = shape_info.get("personal_adapter_available", False)

    STYLE_BASELINE_ADD = 24.0  # parent PAPER.md §1 additive composition
    MCQ_BASELINE_ADD = 15.0

    if not math_available:
        # K_vacate: parent math adapter artefact missing on disk.
        # Cannot evaluate behavioural KCs K1563a/K1563b. Thm 1 (Phase 0)
        # stands on its own as structural evidence.
        print(
            "\n*** K_vacate triggered: parent math adapter artefact missing "
            f"at {MATH_ADAPTER_DIR / 'adapters.safetensors'}. "
            "Thm 1 (structural infeasibility of weight/activation sequential "
            "on q_proj) verified; behavioral KCs K1563a/K1563b cannot be "
            "evaluated. Reporting as PROVISIONAL with K_vacate=true. ***",
            flush=True,
        )
        elapsed = time.time() - t0
        verdict = "PROVISIONAL"
        results["k1563a_style"] = {
            "status": "VACATE",
            "reason": "parent_math_adapter_missing",
            "threshold": STYLE_BASELINE_ADD,
            "pass": False,
        }
        results["k1563b_mcq"] = {
            "status": "VACATE",
            "reason": "parent_math_adapter_missing",
            "threshold": MCQ_BASELINE_ADD,
            "pass": False,
        }
        results["summary"] = {
            "all_pass": False,
            "k1563a_pass": False,
            "k1563b_pass": False,
            "k_vacate_active": True,
            "thm1_structural_pass": bool(shape_info.get("thm1_verified", False)),
            "verdict": verdict,
            "elapsed_s": round(elapsed, 1),
            "vacate_reason": (
                f"parent adapter {MATH_ADAPTER_DIR / 'adapters.safetensors'} "
                "not on disk (gitignored); sibling pattern to "
                "exp_followup_hypernetwork_residual (see .ralph/current_direction.md)"
            ),
        }
        results["verdict"] = verdict

        out_path = EXPERIMENT_DIR / "results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results → {out_path} (K_vacate)", flush=True)
        return

    # Phase 1 baselines (quantify what each adapter alone does on the *opposite* task)
    print("\n--- Phase 1a: domain-only on style task ---", flush=True)
    dom_style = eval_style_single_stage(MATH_ADAPTER_DIR, N_STYLE, "domain-only")
    print("\n--- Phase 1b: personal-only on MCQ task ---", flush=True)
    pers_mcq = eval_mcq_single_stage(PERSONAL_ADAPTER_DIR, N_MATH, "personal-only")

    results["phase1_baselines"] = {
        "domain_only_style_rate": round(float(dom_style), 1),
        "personal_only_mcq_acc": round(float(pers_mcq), 1),
    }

    # Phase 2 sequential
    print("\n--- Phase 2: sequential pipeline on style ---", flush=True)
    seq_style, style_records = run_sequential_pipeline_style(N_STYLE)
    print("\n--- Phase 2: sequential pipeline on MCQ ---", flush=True)
    seq_mcq, mcq_records = run_sequential_pipeline_mcq(N_MATH)

    results["phase2_pipeline"] = {
        "style_rate": round(float(seq_style), 1),
        "mcq_acc": round(float(seq_mcq), 1),
    }
    results["style_records"] = style_records
    results["mcq_records"] = mcq_records

    k1563a_pass = bool(seq_style >= STYLE_BASELINE_ADD)
    k1563b_pass = bool(seq_mcq >= MCQ_BASELINE_ADD)
    all_pass = bool(k1563a_pass and k1563b_pass)

    results["k1563a_style"] = {
        "pipeline_rate": round(float(seq_style), 1),
        "additive_baseline": STYLE_BASELINE_ADD,
        "threshold": STYLE_BASELINE_ADD,
        "pass": k1563a_pass,
    }
    results["k1563b_mcq"] = {
        "pipeline_acc": round(float(seq_mcq), 1),
        "additive_baseline": MCQ_BASELINE_ADD,
        "threshold": MCQ_BASELINE_ADD,
        "pass": k1563b_pass,
    }

    # Verdict: smoke → provisional always (PLAN.md §1 rule 4)
    if IS_SMOKE:
        verdict = "PROVISIONAL"
    else:
        verdict = "SUPPORTED" if all_pass else "KILLED"

    elapsed = time.time() - t0
    results["summary"] = {
        "all_pass": all_pass,
        "k1563a_pass": k1563a_pass,
        "k1563b_pass": k1563b_pass,
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }
    results["verdict"] = verdict

    print("\n=== RESULTS ===", flush=True)
    print(
        f"K1563a (style >= {STYLE_BASELINE_ADD}%): "
        f"pipeline={seq_style:.1f}%, domain_alone={dom_style:.1f}% "
        f"→ {'PASS' if k1563a_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"K1563b (mcq   >= {MCQ_BASELINE_ADD}%): "
        f"pipeline={seq_mcq:.1f}%, personal_alone={pers_mcq:.1f}% "
        f"→ {'PASS' if k1563b_pass else 'FAIL'}",
        flush=True,
    )
    print(f"Verdict: {verdict}  (is_smoke={IS_SMOKE}, all_pass={all_pass})", flush=True)
    print(f"elapsed={elapsed:.0f}s", flush=True)

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {out_path}", flush=True)


if __name__ == "__main__":
    main()
