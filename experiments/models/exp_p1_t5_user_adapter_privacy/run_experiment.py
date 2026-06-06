#!/usr/bin/env python3
"""
T5.4: User adapter does not leak other users' data (isolation proof).

Kill criteria:
  K1110: User A's adapter gives 0 improvement on User B's tasks (behavioral isolation)
  K1111: MIA delta < 20pp — adapter doesn't distinguish training from held-out samples
  K1112: User A and User B subspaces geometrically isolated (max|cos| < 0.50)

Phases:
  Phase 0: Train User B adapter (50 iters, style "Best regards, colleague.") via subprocess
  Phase 1 (CPU): K1112 — Grassmannian isolation between User A and User B lora_a matrices
  Phase 2 (User A adapter): K1110 — apply User A's adapter to User B's queries, measure B-style compliance
  Phase 3 (same model):     K1111 — MIA test: member vs non-member compliance delta

References:
  MATH.md — Theorems 1-3 with formal proofs
  T5.1 (Finding #436): User A adapter, 76pp gain, "Hope that helps, friend!"
  T3.7 (Finding #430): exclusive routing invariance
  T5.2 (Finding #437): user adapter validation, max|cos|=0.2528
"""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
BASE = Path(__file__).parent
REPO_ROOT = BASE.parent.parent.parent.parent

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

USER_A_ADAPTER = BASE.parent / "exp_p1_t5_user_local_training" / "personal_adapter"
USER_A_PREFERENCE = "Hope that helps, friend!"

USER_B_PREFERENCE = "Best regards, colleague."
USER_B_ADAPTER_DIR = BASE / "user_b_adapter"

# Training config for User B
N_TRAIN_B = 5 if SMOKE else 20
TRAIN_ITERS_B = 10 if SMOKE else 100

# Evaluation sizes
N_B_QUERIES = 2 if SMOKE else 5      # K1110: User B's queries
N_MEMBER = 3 if SMOKE else 10        # K1111: training-set questions
N_NON_MEMBER = 3 if SMOKE else 10    # K1111: held-out questions
MAX_TOKENS = 15 if SMOKE else 150

# Memory limits
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)


# ─── User B training data (science questions not in T5.1 dataset) ─────────────

USER_B_TRAIN_QUESTIONS = [
    "What is the speed of sound?",
    "How do black holes form?",
    "What is a neutron star?",
    "How does sonar work?",
    "What is a prism?",
    "How do computers process graphics?",
    "What is osmosis?",
    "How does an MRI scanner work?",
    "What is plate tectonics?",
    "How do birds navigate when migrating?",
    "What causes thunder and lightning?",
    "How does a laser work?",
    "What is the difference between AC and DC current?",
    "How do muscles contract?",
    "What is nuclear waste?",
    "What causes seasons on Earth?",
    "How does a microscope work?",
    "What is the Doppler effect?",
    "How do plants reproduce?",
    "What is continental drift?",
]

USER_B_ANSWERS = [
    "The speed of sound in air is approximately 343 meters per second at room temperature.",
    "Black holes form when massive stars exhaust their fuel and collapse under their own gravity.",
    "A neutron star is an extremely dense stellar remnant composed almost entirely of neutrons.",
    "Sonar works by emitting sound pulses and measuring the time for echoes to return from objects.",
    "A prism refracts light at different angles by wavelength, separating white light into its spectrum.",
    "Computers process graphics using specialized GPUs with thousands of parallel cores for pixel calculations.",
    "Osmosis is the movement of water molecules across a semi-permeable membrane from lower to higher solute concentration.",
    "An MRI scanner uses strong magnetic fields and radio waves to create detailed images of soft tissue.",
    "Plate tectonics describes how Earth's crust is divided into large moving plates that cause earthquakes and volcanoes.",
    "Birds navigate using a combination of the Earth's magnetic field, star positions, landmarks, and the sun's position.",
    "Thunder is caused by the rapid expansion of air heated by a lightning bolt's electrical discharge.",
    "A laser works by stimulating atoms to emit coherent photons all at the same wavelength and phase.",
    "AC current alternates direction periodically while DC current flows in one direction continuously.",
    "Muscles contract when motor neurons release acetylcholine, triggering calcium release and actin-myosin cross-bridge cycling.",
    "Nuclear waste consists of radioactive byproducts from nuclear reactors requiring careful long-term isolation.",
    "Earth's seasons are caused by the tilt of Earth's axis relative to its orbital plane around the Sun.",
    "A microscope uses lenses to magnify objects too small to see with the naked eye by bending light.",
    "The Doppler effect is the change in frequency of a wave as its source moves relative to an observer.",
    "Plants reproduce through pollination (sexual) or vegetative propagation (asexual) depending on the species.",
    "Continental drift is the gradual movement of Earth's continents over geological time due to mantle convection.",
]

# User B test queries for K1110 (what User B typically asks)
USER_B_TEST_QUERIES = [
    "What is the boiling point of water?",
    "How does a rainbow form?",
    "What is the largest planet in the solar system?",
    "How does friction work?",
    "What is a semiconductor?",
]

# K1111: Member questions = first 10 from T5.1 TRAIN_QUESTIONS (questions User A was trained on)
MEMBER_QUESTIONS = [
    "What is photosynthesis?",
    "How do computers store data?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is machine learning?",
    "Why do leaves change color in autumn?",
    "How does the internet work?",
    "What is DNA?",
    "Why do we dream?",
]

# K1111: Non-member questions = held-out questions never in T5.1 training
NON_MEMBER_QUESTIONS = [
    "What is the boiling point of water?",
    "How does a rainbow form?",
    "What is the largest planet in the solar system?",
    "How does friction work?",
    "What is a semiconductor?",
    "What is the capital of France?",
    "How do hurricanes form?",
    "What is a superconductor?",
    "How does the human digestive system work?",
    "What is nuclear fission?",
]


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_lora_a_matrices(adapter_path: Path) -> dict:
    """Load lora_a (q_proj) matrices from adapter safetensors. Returns {layer: np.array}."""
    from safetensors import safe_open
    safetensors_path = adapter_path / "adapters.safetensors"
    matrices = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for key in f.keys():
            if "lora_a" in key and "q_proj" in key:
                layer_idx = int(key.split(".layers.")[1].split(".")[0])
                matrices[layer_idx] = f.get_tensor(key)
    return matrices


def column_orthonormal(A: np.ndarray) -> np.ndarray:
    """QR decomposition to get orthonormal column basis."""
    A64 = A.astype(np.float64)
    col_norms = np.linalg.norm(A64, axis=0)
    valid = col_norms >= 1e-10
    A64 = A64[:, valid]
    if A64.shape[1] == 0:
        return np.zeros((A.shape[0], 1), dtype=np.float64)
    Q, _ = np.linalg.qr(A64, mode="reduced")
    return Q


def max_principal_angle_cosine(A1: np.ndarray, A2: np.ndarray) -> float:
    """Max cosine similarity between two subspaces via principal angles (largest singular value)."""
    Q1 = column_orthonormal(A1)
    Q2 = column_orthonormal(A2)
    cross = Q1.T @ Q2
    svd_vals = np.linalg.svd(cross, compute_uv=False)
    return float(np.clip(svd_vals[0], 0.0, 1.0))


def generate_responses(model, tokenizer, prompts: list) -> list:
    """Run generation for a list of prompts. Returns response strings."""
    from mlx_lm import generate
    responses = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out = generate(model, tokenizer, prompt=formatted, max_tokens=MAX_TOKENS, verbose=False)
        responses.append(out)
    return responses


def compliance_rate(responses: list, marker: str) -> float:
    return sum(1 for r in responses if marker in r) / max(len(responses), 1)


# ─── Phase 0: Train User B adapter ────────────────────────────────────────────

def phase0_train_user_b():
    """Train User B's personal style adapter via mlx_lm.lora subprocess."""
    import tempfile

    print("\n=== Phase 0: Train User B adapter ===")
    USER_B_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        config_path = Path(tmpdir) / "config.yaml"

        n = N_TRAIN_B
        questions = USER_B_TRAIN_QUESTIONS[:n]
        answers = USER_B_ANSWERS[:n]

        # Write train.jsonl with User B's style
        train_lines = []
        for q, a in zip(questions, answers):
            example = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"{a}\n\n{USER_B_PREFERENCE}"},
                ]
            }
            train_lines.append(json.dumps(example))

        (data_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n")
        # Use last example as validation
        (data_dir / "valid.jsonl").write_text(train_lines[-1] + "\n")

        config_lines = [
            f"model: {MODEL_ID}",
            f"data: {data_dir}",
            f"adapter_path: {USER_B_ADAPTER_DIR}",
            "train: true",
            "fine_tune_type: lora",
            f"iters: {TRAIN_ITERS_B}",
            "batch_size: 1",
            "num_layers: 16",
            "learning_rate: 1e-4",
            "lora_parameters:",
            "  rank: 4",
            "  scale: 4.0",
            "  dropout: 0.0",
            "  keys:",
            "    - self_attn.q_proj",
            "max_seq_length: 256",
            "mask_prompt: true",
            "grad_checkpoint: true",
            f"save_every: {TRAIN_ITERS_B}",
            "steps_per_report: 50",
            "seed: 99",
        ]
        config_path.write_text("\n".join(config_lines) + "\n")

        proc = subprocess.run(
            ["uv", "run", "python", "-m", "mlx_lm.lora", "--config", str(config_path)],
            cwd=REPO_ROOT,
        )

    elapsed = time.perf_counter() - t0
    ok = (proc.returncode == 0) and (USER_B_ADAPTER_DIR / "adapters.safetensors").exists()
    print(f"  User B training: {elapsed:.1f}s, returncode={proc.returncode}, ok={ok}")

    if not ok:
        print("  ERROR: User B training failed!")
        return {"ok": False, "elapsed_s": elapsed}

    return {"ok": True, "elapsed_s": elapsed, "adapter_path": str(USER_B_ADAPTER_DIR)}


# ─── Phase 1: K1112 Grassmannian isolation ────────────────────────────────────

def phase1_grassmannian():
    """K1112: max|cos(Y_A, Y_B)| < 0.50 across all shared layers."""
    print("\n=== Phase 1: Grassmannian Isolation (K1112) ===")
    t0 = time.perf_counter()

    a_matrices = load_lora_a_matrices(USER_A_ADAPTER)
    b_matrices = load_lora_a_matrices(USER_B_ADAPTER_DIR)

    shared_layers = sorted(set(a_matrices.keys()) & set(b_matrices.keys()))
    print(f"  User A layers: {len(a_matrices)}, User B layers: {len(b_matrices)}, shared: {len(shared_layers)}")

    cosines = []
    for layer in shared_layers:
        cos = max_principal_angle_cosine(a_matrices[layer], b_matrices[layer])
        cosines.append((layer, cos))

    max_cos = float(np.max([c for _, c in cosines]))
    mean_cos = float(np.mean([c for _, c in cosines]))
    worst_layer, worst_cos = max(cosines, key=lambda x: x[1])

    k1112_pass = max_cos < 0.50
    print(f"  max|cos|={max_cos:.4f} (layer {worst_layer}), mean={mean_cos:.4f}")
    print(f"  JL bound for r=4, d=2560: {np.sqrt(4/2560):.4f}")
    print(f"  K1112: max|cos|={max_cos:.4f} < 0.50 → {'PASS' if k1112_pass else 'FAIL'}")

    elapsed = time.perf_counter() - t0
    return {
        "max_cos": max_cos,
        "mean_cos": mean_cos,
        "worst_layer": worst_layer,
        "worst_cos": worst_cos,
        "n_layers": len(cosines),
        "jl_bound": float(np.sqrt(4/2560)),
        "k1112_pass": k1112_pass,
        "elapsed_s": elapsed,
    }


# ─── Phase 2: K1110 Behavioral isolation ──────────────────────────────────────

def phase2_behavioral_isolation(model, tokenizer):
    """K1110: User A's adapter applied to User B's queries → 0/N produce User B's sign-off."""
    print("\n=== Phase 2: Behavioral Isolation (K1110) ===")
    t0 = time.perf_counter()

    queries = USER_B_TEST_QUERIES[:N_B_QUERIES]
    print(f"  Running {len(queries)} User B queries with User A's adapter...")
    responses_a_on_b = generate_responses(model, tokenizer, queries)

    # Check User B's sign-off compliance (should be ~0%)
    b_compliance = compliance_rate(responses_a_on_b, USER_B_PREFERENCE)
    # Check User A's sign-off compliance (should be high — A's adapter is working)
    a_compliance = compliance_rate(responses_a_on_b, USER_A_PREFERENCE)

    n_b_signoff = sum(1 for r in responses_a_on_b if USER_B_PREFERENCE in r)
    k1110_pass = n_b_signoff == 0

    print(f"  User A's sign-off ('Hope...'): {a_compliance:.0%}")
    print(f"  User B's sign-off ('Best...'): {b_compliance:.0%} ({n_b_signoff}/{len(queries)})")
    print(f"  K1110: 0/{len(queries)} produce User B's sign-off → {'PASS' if k1110_pass else 'FAIL'}")

    if responses_a_on_b:
        print(f"  Sample response (truncated): {responses_a_on_b[0][:120]!r}")

    elapsed = time.perf_counter() - t0
    return {
        "n_queries": len(queries),
        "n_b_signoff": n_b_signoff,
        "a_compliance": a_compliance,
        "b_compliance_with_a_adapter": b_compliance,
        "responses_sample": [r[:100] for r in responses_a_on_b[:3]],
        "k1110_pass": k1110_pass,
        "elapsed_s": elapsed,
    }


# ─── Phase 3: K1111 MIA test ──────────────────────────────────────────────────

def phase3_mia(model, tokenizer):
    """K1111: |member_compliance - non_member_compliance| < 20pp.

    Tests whether User A's style adapter distinguishes training samples from held-out samples.
    If the adapter generalizes uniformly (style injection, not content memorization), the
    compliance rate should be similar for both member and non-member prompts.
    """
    print("\n=== Phase 3: MIA Test (K1111) ===")
    t0 = time.perf_counter()

    member_qs = MEMBER_QUESTIONS[:N_MEMBER]
    non_member_qs = NON_MEMBER_QUESTIONS[:N_NON_MEMBER]

    print(f"  Running {len(member_qs)} member + {len(non_member_qs)} non-member queries...")
    member_responses = generate_responses(model, tokenizer, member_qs)
    non_member_responses = generate_responses(model, tokenizer, non_member_qs)

    member_compliance = compliance_rate(member_responses, USER_A_PREFERENCE)
    non_member_compliance = compliance_rate(non_member_responses, USER_A_PREFERENCE)
    delta_pp = abs(member_compliance - non_member_compliance) * 100

    k1111_pass = delta_pp < 20.0

    print(f"  Member compliance:     {member_compliance:.1%} ({sum(1 for r in member_responses if USER_A_PREFERENCE in r)}/{len(member_qs)})")
    print(f"  Non-member compliance: {non_member_compliance:.1%} ({sum(1 for r in non_member_responses if USER_A_PREFERENCE in r)}/{len(non_member_qs)})")
    print(f"  |delta| = {delta_pp:.1f}pp < 20pp → {'PASS' if k1111_pass else 'FAIL'}")

    elapsed = time.perf_counter() - t0
    return {
        "n_member": len(member_qs),
        "n_non_member": len(non_member_qs),
        "member_compliance": member_compliance,
        "non_member_compliance": non_member_compliance,
        "delta_pp": delta_pp,
        "k1111_pass": k1111_pass,
        "elapsed_s": elapsed,
        "member_samples": [r[:80] for r in member_responses[:3]],
        "non_member_samples": [r[:80] for r in non_member_responses[:3]],
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    if SMOKE:
        print("=== SMOKE TEST MODE ===")
    print("=" * 60)
    print("T5.4: User Adapter Privacy (Isolation Proof)")
    print("=" * 60)

    t_start = time.perf_counter()

    # Phase 0: Train User B adapter
    p0 = phase0_train_user_b()
    if not p0["ok"]:
        print("Phase 0 failed — cannot proceed.")
        results = {"smoke": SMOKE, "phase0": p0, "all_pass": False}
        (BASE / "results.json").write_text(json.dumps(results, indent=2))
        return 1

    # Phase 1: CPU-only Grassmannian check
    p1 = phase1_grassmannian()

    # Load User A's adapter model ONCE for phases 2 + 3
    print("\nLoading User A adapter model...")
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    t_load = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    model = load_adapters(model, str(USER_A_ADAPTER))
    mx.eval(model.parameters())
    print(f"  Model loaded in {time.perf_counter() - t_load:.1f}s")

    # Phase 2: K1110 behavioral isolation
    p2 = phase2_behavioral_isolation(model, tokenizer)

    # Phase 3: K1111 MIA test (reuse same loaded model)
    p3 = phase3_mia(model, tokenizer)

    del model
    gc.collect()
    mx.clear_cache()

    # Summary
    total_s = time.perf_counter() - t_start
    all_pass = p1["k1112_pass"] and p2["k1110_pass"] and p3["k1111_pass"]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"K1110 (behavioral isolation): {p2['n_b_signoff']}/5 User-B sign-offs → {'PASS' if p2['k1110_pass'] else 'FAIL'}")
    print(f"K1111 (MIA delta):            |{p3['member_compliance']:.0%} - {p3['non_member_compliance']:.0%}| = {p3['delta_pp']:.1f}pp < 20pp → {'PASS' if p3['k1111_pass'] else 'FAIL'}")
    print(f"K1112 (Grassmannian):         max|cos|={p1['max_cos']:.4f} < 0.50 → {'PASS' if p1['k1112_pass'] else 'FAIL'}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'} (total={total_s:.1f}s)")

    results = {
        "smoke": SMOKE,
        "model_id": MODEL_ID,
        "user_a_adapter": str(USER_A_ADAPTER),
        "user_b_adapter": str(USER_B_ADAPTER_DIR),
        "user_a_preference": USER_A_PREFERENCE,
        "user_b_preference": USER_B_PREFERENCE,
        "phase0_train_b": p0,
        "k1110": p2,
        "k1111": p3,
        "k1112": p1,
        "total_s": total_s,
        "all_pass": all_pass,
    }

    (BASE / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {BASE / 'results.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
