#!/usr/bin/env python3
"""End-to-end cost accounting for the SOLE expert pipeline.

Decomposes true $/expert from actual pilot-50 data into:
  1. Data generation (Groq API)
  2. Data preparation & transfer
  3. Training (GPU compute)
  4. Model loading overhead (amortized)
  5. Orthogonality check
  6. Gram-Schmidt projection
  7. Merge into composed model
  8. Benchmark evaluation
  9. Quality gate decision

Uses actual logs, pricing, and timing from the pilot-50 run.
"""

import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
GEN_LOG = RESULTS_DIR / "pilot50_generate.log"

# ─── Pricing constants (actual RunPod/Groq rates) ───────────────────
RUNPOD_4090_PER_HR = 0.34    # $/hr for RTX 4090 on RunPod
RUNPOD_A5000_PER_HR = 0.16   # $/hr for A5000 on RunPod (training alternative)
GROQ_70B_PER_EXPERT = 0.355  # $/expert from actual log (most domains)
N_EXPERTS = 50

# ─── 1. Data Generation Costs (from actual Groq log) ────────────────
def parse_generation_log():
    """Parse pilot50_generate.log for per-domain costs and times."""
    if not GEN_LOG.exists():
        print(f"WARNING: {GEN_LOG} not found, using estimates")
        return None

    text = GEN_LOG.read_text()
    # Pattern: "domain: 1000/1000 complete (Xs, Y/s, ~$Z)"
    pattern = r'(\S+): \d+/\d+ complete \((\d+)s, [\d.]+/s, ~\$([\d.]+)\)'
    matches = re.findall(pattern, text)

    domains = {}
    for domain, time_s, cost in matches:
        domains[domain] = {
            'gen_time_s': int(time_s),
            'gen_cost': float(cost),
        }

    # Extract total from log
    total_match = re.search(r'Estimated total cost: \$([\d.]+)', text)
    total_cost = float(total_match.group(1)) if total_match else sum(d['gen_cost'] for d in domains.values())

    return domains, total_cost

# ─── 2. Training Costs (from actual timing) ──────────────────────────
def estimate_training_costs():
    """Estimate training costs from known parameters.

    From pilot-50 PAPER.md: ~15 min/expert on 4090, 300 steps, rank-16.
    Training script spawns subprocess per domain for memory cleanup.
    """
    # Known: 50 experts, ~15 min each on 4090
    train_time_per_expert_min = 15.0

    # But this includes model loading per expert (~2-3 min for Qwen2.5-7B 4-bit)
    # The subprocess approach reloads the model each time for memory safety
    model_load_time_min = 2.5  # estimated: 7B model in 4-bit from cache
    pure_train_time_min = train_time_per_expert_min - model_load_time_min

    train_cost_per_expert = train_time_per_expert_min / 60 * RUNPOD_4090_PER_HR
    pure_train_cost = pure_train_time_min / 60 * RUNPOD_4090_PER_HR
    load_overhead_cost = model_load_time_min / 60 * RUNPOD_4090_PER_HR

    return {
        'total_time_min': train_time_per_expert_min,
        'pure_train_min': pure_train_time_min,
        'model_load_min': model_load_time_min,
        'total_cost': train_cost_per_expert,
        'pure_train_cost': pure_train_cost,
        'load_overhead_cost': load_overhead_cost,
    }

# ─── 3. Data Transfer Costs ─────────────────────────────────────────
def estimate_transfer_costs():
    """Estimate rsync/scp time and GPU idle cost during transfer.

    50 domains * ~1MB each = ~50MB total data.
    RunPod has fast network, ~100 MB/s typical rsync.
    Transfer is one-time batch, not per-expert.
    """
    data_size_mb = 50  # ~1MB per domain (1000 JSONL examples)
    transfer_speed_mbps = 50  # conservative rsync over SSH
    transfer_time_s = data_size_mb / transfer_speed_mbps

    # GPU is idle during transfer (if renting during transfer)
    # But transfer happens before training starts, so it's overhead
    idle_cost = transfer_time_s / 3600 * RUNPOD_4090_PER_HR
    per_expert = idle_cost / N_EXPERTS

    return {
        'data_size_mb': data_size_mb,
        'transfer_time_s': transfer_time_s,
        'total_idle_cost': idle_cost,
        'per_expert': per_expert,
    }

# ─── 4. Orthogonality Check Costs ───────────────────────────────────
def estimate_orthogonality_costs():
    """CPU-only computation: load adapter A matrices, compute pairwise cosines.

    At N=50, r=16, d=4096: each A is (r, d) = (16, 4096) = 262K params.
    Pairwise: C(50,2) = 1225 pairs, each dot product is O(r*d).
    Total: ~1225 * 16 * 4096 = ~80M FLOPs. Trivial.
    """
    n_pairs = N_EXPERTS * (N_EXPERTS - 1) // 2
    flops_per_pair = 16 * 4096  # dot product of flattened A matrices
    total_flops = n_pairs * flops_per_pair

    # CPU can do ~10 GFLOPS, so this is <0.01s
    time_s = total_flops / 10e9

    return {
        'n_pairs': n_pairs,
        'total_flops': total_flops,
        'time_s': time_s,
        'cost': 0.0,  # negligible, CPU only
    }

# ─── 5. Gram-Schmidt Projection Costs ───────────────────────────────
def estimate_gs_projection_costs():
    """GS projection after each new expert: O(N * r * d) per expert.

    From micro experiments: 1-15s at d=64-256, estimated 5-10 min at d=4096.
    But at SOLE production cosines (cos=0.0002), GS is essentially a no-op
    (projection magnitude ~ cos * ||B|| is negligible).

    In practice: can skip GS entirely at SOLE cosines, or run on CPU.
    """
    # Conservative: 5 min total for N=50 on CPU at d=4096
    time_s = 300
    # No GPU needed
    cost = 0.0  # CPU only, done locally
    per_expert = time_s / N_EXPERTS

    return {
        'total_time_s': time_s,
        'per_expert_time_s': per_expert,
        'cost': cost,
    }

# ─── 6. Merge Costs ─────────────────────────────────────────────────
def estimate_merge_costs():
    """Pre-merge: W_composed = W_base + sum(B_i @ A_i).

    Each B_i @ A_i is (d, r) @ (r, d) = d^2 multiply. At d=4096, r=16:
    each matmul is 4096*16*4096 = 268M FLOPs. For 50 experts over 7 module
    types and 28 layers: 50 * 7 * 28 = 9800 matmuls.
    Total: ~2.6T FLOPs. GPU at 80 TFLOPS: ~33s. CPU: ~260s.

    But pre-merge is done ONCE for serving, not per-expert.
    Per-expert marginal cost: one set of matmuls = 7 * 28 = 196 matmuls.
    """
    n_modules = 7  # q,k,v,o,gate,up,down
    n_layers = 28  # Qwen2.5-7B
    d = 4096
    r = 16

    flops_per_matmul = d * r * d
    matmuls_per_expert = n_modules * n_layers
    total_flops = N_EXPERTS * matmuls_per_expert * flops_per_matmul

    # CPU at 100 GFLOPS (Apple Silicon)
    time_s_cpu = total_flops / 100e9
    # GPU at 80 TFLOPS
    time_s_gpu = total_flops / 80e12

    return {
        'matmuls_per_expert': matmuls_per_expert,
        'total_flops': total_flops,
        'cpu_time_s': time_s_cpu,
        'gpu_time_s': time_s_gpu,
        'cost': 0.0,  # done on serving machine, not rented GPU
    }

# ─── 7. Benchmark/Eval Costs ────────────────────────────────────────
def estimate_benchmark_costs():
    """From pilot50_orchestrate.sh: ~2 hours for benchmarking 50 experts.

    Each expert: load base + adapter, compute PPL on 100 examples.
    Bottleneck: model loading per adapter (need to swap LoRA weights).
    With vLLM: adapter hot-swap is ~1s. Without: full reload ~2.5 min.

    Current implementation: sequential, full reload per domain.
    """
    # From orchestrate.sh comment: "~2 hours" for benchmark step
    total_bench_time_hr = 2.0
    per_expert_min = total_bench_time_hr * 60 / N_EXPERTS  # 2.4 min/expert

    # Most of that is model loading, not inference
    model_load_min = 2.0
    inference_min = per_expert_min - model_load_min  # ~0.4 min = 24s for 100 examples

    # GPU cost during benchmarking
    total_cost = total_bench_time_hr * RUNPOD_4090_PER_HR
    per_expert = total_cost / N_EXPERTS

    return {
        'total_time_hr': total_bench_time_hr,
        'per_expert_min': per_expert_min,
        'model_load_min': model_load_min,
        'inference_min': inference_min,
        'total_cost': total_cost,
        'per_expert_cost': per_expert,
    }

# ─── 8. Quality Gate Costs ───────────────────────────────────────────
def estimate_quality_gate_costs():
    """Automated decision: compare expert_ppl < base_ppl.
    Single comparison per domain. Negligible compute."""
    return {
        'time_s': 0.001,
        'cost': 0.0,
    }

# ─── 9. GPU Idle Time / Waste ────────────────────────────────────────
def estimate_idle_costs():
    """From MEMORY.md: ~$22 spent including idle time waste.

    Sources of GPU idle time:
    - Pod startup/setup (~5 min): install deps, download model
    - Between-expert subprocess overhead (~30s each): Python restart, CUDA init
    - Debugging/monitoring SSH sessions while GPU rented
    - Pod not terminated promptly after job completion

    From the pilot: $22 total spent. If training was 50 * 15min = 12.5 hr
    at $0.34/hr = $4.25, and data gen was $15.91, that leaves $1.84 for
    overhead/idle.
    """
    total_spent = 22.0  # from PAPER.md
    data_gen_cost = 15.91  # from generate log
    pure_training_cost = 50 * 15 / 60 * RUNPOD_4090_PER_HR  # 50 * 0.25hr * 0.34
    bench_cost = 2.0 * RUNPOD_4090_PER_HR  # ~2 hr benchmark

    accounted = data_gen_cost + pure_training_cost + bench_cost
    idle_waste = max(0, total_spent - accounted)
    per_expert = idle_waste / N_EXPERTS

    return {
        'total_spent': total_spent,
        'data_gen': data_gen_cost,
        'training': pure_training_cost,
        'benchmark': bench_cost,
        'accounted': accounted,
        'idle_waste': idle_waste,
        'per_expert': per_expert,
    }

# ─── 10. First-Expert Setup Costs (amortized) ───────────────────────
def estimate_setup_costs():
    """One-time costs amortized across N experts:
    - Domain taxonomy generation (Claude/GPT call): ~$0.02
    - Pipeline code development: human time, not $/expert
    - Base model download to RunPod: ~10 min at first use, cached after
    - Dependency installation: ~2 min
    """
    taxonomy_cost = 0.02  # single API call to generate 50 domains
    model_download_time_min = 10  # 7B model, fast RunPod network
    dep_install_time_min = 2
    setup_gpu_idle_min = model_download_time_min + dep_install_time_min
    setup_gpu_cost = setup_gpu_idle_min / 60 * RUNPOD_4090_PER_HR

    per_expert = (taxonomy_cost + setup_gpu_cost) / N_EXPERTS

    return {
        'taxonomy_cost': taxonomy_cost,
        'setup_gpu_cost': setup_gpu_cost,
        'total': taxonomy_cost + setup_gpu_cost,
        'per_expert': per_expert,
    }

# ─── Main Analysis ───────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SOLE Expert Pipeline: End-to-End Cost Accounting")
    print("=" * 70)
    print(f"Based on: 50-expert pilot (Qwen2.5-7B, rank-16, all-modules)")
    print(f"Hardware: RunPod RTX 4090 (${RUNPOD_4090_PER_HR}/hr)")
    print(f"Teacher: Groq Llama 3.3 70B batch API")
    print()

    # Parse actual data
    gen_data = parse_generation_log()
    if gen_data:
        domains, total_gen_cost = gen_data
        n_domains = len(domains)
        gen_times = [d['gen_time_s'] for d in domains.values()]
        gen_costs = [d['gen_cost'] for d in domains.values()]

        print(f"1. DATA GENERATION (Groq API)")
        print(f"   Domains parsed: {n_domains}")
        print(f"   Total cost: ${total_gen_cost:.2f}")
        print(f"   Per expert: ${total_gen_cost/n_domains:.3f}")
        print(f"   Time range: {min(gen_times)}-{max(gen_times)}s (median {statistics.median(gen_times):.0f}s)")
        print(f"   Cost range: ${min(gen_costs):.3f}-${max(gen_costs):.3f}")
        print(f"   Wall-clock (sequential): {sum(gen_times)/3600:.1f} hr")
        # But generation runs with 8 workers, so effective wall clock is less
        # However, domains are sequential (one at a time in the log)
        print(f"   Note: runs 8 concurrent API workers per domain")
        data_gen_per_expert = total_gen_cost / n_domains
    else:
        data_gen_per_expert = GROQ_70B_PER_EXPERT
        print(f"1. DATA GENERATION: ${data_gen_per_expert:.3f}/expert (estimated)")

    print()

    # Training
    train = estimate_training_costs()
    print(f"2. TRAINING (QLoRA on 4090)")
    print(f"   Total time per expert: {train['total_time_min']:.1f} min")
    print(f"     - Pure training (300 steps): {train['pure_train_min']:.1f} min")
    print(f"     - Model loading overhead: {train['model_load_min']:.1f} min")
    print(f"   Cost per expert: ${train['total_cost']:.3f}")
    print(f"     - Pure training: ${train['pure_train_cost']:.3f}")
    print(f"     - Model loading: ${train['load_overhead_cost']:.3f}")
    print()

    # Transfer
    transfer = estimate_transfer_costs()
    print(f"3. DATA TRANSFER (rsync to RunPod)")
    print(f"   Data size: ~{transfer['data_size_mb']} MB")
    print(f"   Transfer time: ~{transfer['transfer_time_s']:.0f}s")
    print(f"   GPU idle cost: ${transfer['total_idle_cost']:.4f} total, ${transfer['per_expert']:.5f}/expert")
    print()

    # Orthogonality
    ortho = estimate_orthogonality_costs()
    print(f"4. ORTHOGONALITY CHECK (CPU)")
    print(f"   Pairwise comparisons: {ortho['n_pairs']}")
    print(f"   Compute time: <{ortho['time_s']*1000:.1f}ms")
    print(f"   Cost: $0.00 (CPU only)")
    print()

    # GS projection
    gs = estimate_gs_projection_costs()
    print(f"5. GRAM-SCHMIDT PROJECTION (CPU)")
    print(f"   Total time: ~{gs['total_time_s']:.0f}s")
    print(f"   Per expert: ~{gs['per_expert_time_s']:.0f}s")
    print(f"   Cost: $0.00 (CPU only, negligible at SOLE cosines)")
    print()

    # Merge
    merge = estimate_merge_costs()
    print(f"6. PRE-MERGE COMPOSITION")
    print(f"   Matmuls per expert: {merge['matmuls_per_expert']}")
    print(f"   CPU time (all 50): ~{merge['cpu_time_s']:.1f}s")
    print(f"   GPU time (all 50): ~{merge['gpu_time_s']*1000:.1f}ms")
    print(f"   Cost: $0.00 (done on serving machine)")
    print()

    # Benchmark
    bench = estimate_benchmark_costs()
    print(f"7. BENCHMARK EVALUATION (GPU)")
    print(f"   Total time: ~{bench['total_time_hr']:.1f} hr")
    print(f"   Per expert: ~{bench['per_expert_min']:.1f} min")
    print(f"     - Model loading: ~{bench['model_load_min']:.1f} min")
    print(f"     - Inference (100 examples): ~{bench['inference_min']:.1f} min")
    print(f"   Total cost: ${bench['total_cost']:.2f}")
    print(f"   Per expert: ${bench['per_expert_cost']:.3f}")
    print()

    # Quality gate
    gate = estimate_quality_gate_costs()
    print(f"8. QUALITY GATE (CPU)")
    print(f"   Cost: $0.00 (single comparison)")
    print()

    # Setup (amortized)
    setup = estimate_setup_costs()
    print(f"9. SETUP (amortized over {N_EXPERTS} experts)")
    print(f"   Taxonomy generation: ${setup['taxonomy_cost']:.2f}")
    print(f"   Model download + deps: ${setup['setup_gpu_cost']:.2f}")
    print(f"   Per expert: ${setup['per_expert']:.4f}")
    print()

    # Idle/waste
    idle = estimate_idle_costs()
    print(f"10. GPU IDLE TIME / WASTE")
    print(f"    Total spent (from logs): ${idle['total_spent']:.2f}")
    print(f"    Accounted costs: ${idle['accounted']:.2f}")
    print(f"    Unaccounted (idle/waste): ${idle['idle_waste']:.2f}")
    print(f"    Per expert: ${idle['per_expert']:.3f}")
    print()

    # ─── TOTALS ──────────────────────────────────────────────────────
    print("=" * 70)
    print("COST DECOMPOSITION PER EXPERT")
    print("=" * 70)

    components = {
        'Data generation (Groq API)': data_gen_per_expert,
        'Training (pure, 300 steps)': train['pure_train_cost'],
        'Training (model loading)': train['load_overhead_cost'],
        'Data transfer': transfer['per_expert'],
        'Orthogonality check': 0.0,
        'GS projection': 0.0,
        'Pre-merge composition': 0.0,
        'Benchmark evaluation': bench['per_expert_cost'],
        'Quality gate': 0.0,
        'Setup (amortized)': setup['per_expert'],
        'GPU idle/waste': idle['per_expert'],
    }

    total_per_expert = sum(components.values())
    training_cost = train['total_cost']  # pure training + model load
    non_training_cost = total_per_expert - training_cost

    for name, cost in components.items():
        pct = cost / total_per_expert * 100 if total_per_expert > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {name:35s} ${cost:.4f}  ({pct:5.1f}%) {bar}")

    print(f"  {'─' * 55}")
    print(f"  {'TOTAL PER EXPERT':35s} ${total_per_expert:.4f}")
    print()

    # Categorize
    print("COST CATEGORIES:")
    print(f"  Training (GPU compute):        ${training_cost:.4f}  ({training_cost/total_per_expert*100:.1f}%)")
    print(f"  Non-training overhead:         ${non_training_cost:.4f}  ({non_training_cost/total_per_expert*100:.1f}%)")
    print(f"  Overhead / Training ratio:     {non_training_cost/training_cost:.2f}x")
    print()

    # Kill criteria
    print("=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)
    k1_pass = total_per_expert <= 1.00
    k2_ratio = non_training_cost / training_cost
    k2_pass = k2_ratio <= 3.0

    print(f"  K1: True cost/expert <= $1.00")
    print(f"      Actual: ${total_per_expert:.4f}")
    print(f"      Verdict: {'PASS' if k1_pass else 'KILL'} (margin: {(1.00 - total_per_expert)/1.00*100:.0f}%)")
    print()
    print(f"  K2: Overhead <= 3x training cost")
    print(f"      Training: ${training_cost:.4f}")
    print(f"      Overhead: ${non_training_cost:.4f}")
    print(f"      Ratio: {k2_ratio:.2f}x")
    print(f"      Verdict: {'PASS' if k2_pass else 'KILL'}")
    print()

    # Scaling projections
    print("=" * 70)
    print("SCALING PROJECTIONS")
    print("=" * 70)

    # At scale (N=500), amortized setup is negligible
    # But: 8B teacher instead of 70B would change data gen cost
    print("Scenario A: Current pipeline at N=500")
    scaled_setup = setup['total'] / 500
    scaled_idle = idle['idle_waste'] / 500  # fixed idle amortized more
    scaled_total = data_gen_per_expert + train['total_cost'] + bench['per_expert_cost'] + scaled_setup + scaled_idle
    print(f"  Per expert: ${scaled_total:.3f}")
    print()

    print("Scenario B: 8B teacher (Groq batch $0.02/expert)")
    cheap_gen = 0.02
    b_total = cheap_gen + train['total_cost'] + bench['per_expert_cost'] + scaled_setup
    print(f"  Per expert: ${b_total:.3f}")
    print()

    print("Scenario C: A5000 ($0.16/hr) instead of 4090 ($0.34/hr)")
    a5000_train = train['total_time_min'] / 60 * RUNPOD_A5000_PER_HR
    a5000_bench = bench['per_expert_min'] / 60 * RUNPOD_A5000_PER_HR
    c_total = data_gen_per_expert + a5000_train + a5000_bench + scaled_setup
    print(f"  Per expert: ${c_total:.3f}")
    print()

    print("Scenario D: Optimal (8B teacher + A5000 + vLLM eval)")
    # vLLM eval: hot-swap adapters, no model reload, ~10s per expert instead of 2.4 min
    vllm_bench = (10 / 3600) * RUNPOD_A5000_PER_HR
    d_total = cheap_gen + a5000_train + vllm_bench + scaled_setup
    print(f"  Per expert: ${d_total:.3f}")
    print()

    # Biggest cost driver
    sorted_components = sorted(components.items(), key=lambda x: -x[1])
    print("COST RANKING (biggest to smallest):")
    for i, (name, cost) in enumerate(sorted_components, 1):
        if cost > 0:
            print(f"  {i}. {name}: ${cost:.4f} ({cost/total_per_expert*100:.1f}%)")

    # Save results
    results = {
        'n_experts': N_EXPERTS,
        'total_per_expert': round(total_per_expert, 4),
        'training_cost': round(training_cost, 4),
        'non_training_overhead': round(non_training_cost, 4),
        'overhead_ratio': round(k2_ratio, 2),
        'k1_pass': k1_pass,
        'k1_threshold': 1.00,
        'k2_pass': k2_pass,
        'k2_threshold': 3.0,
        'components': {k: round(v, 5) for k, v in components.items()},
        'scaling': {
            'current_at_500': round(scaled_total, 3),
            '8b_teacher': round(b_total, 3),
            'a5000_gpu': round(c_total, 3),
            'optimal': round(d_total, 3),
        },
        'dominant_cost': sorted_components[0][0],
        'dominant_pct': round(sorted_components[0][1] / total_per_expert * 100, 1),
    }

    results_path = Path(__file__).parent / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
