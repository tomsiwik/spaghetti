"""
exp_g4_adapter_class_composition_full_impl — Phase A executable slice.

Per MATH.md §1: this iteration loads the Gemma 4 E4B 4-bit base, enumerates
v_proj + o_proj target presence (F#627), and confirms mlx-lm DoRA availability.
No training, no eval. Writes results.json with verdict=PROVISIONAL, all KCs
(K1-K4) untested, and a phase_a_readout block populated at runtime.

Mirrors the marginal Phase A slice precedent of:
  - exp_memento_gemma4_replication_impl (F#799, iter ~104)
  - exp_jepa_adapter_residual_stream_impl (F#772)
"""

import gc
import json
import sys
import time
from pathlib import Path


KILL_CRITERIA = [
    {
        "id": "K1",
        "text": (
            "Structural + adapter-class health: >=13/15 class-domain trainings "
            "converge (final_loss < 1.1 * min_train_loss AND < 0.7 * initial_loss)"
        ),
        "result": "untested",
        "note": "Phase B deferred (same-dir follow-up at P=3, ~6-10h).",
    },
    {
        "id": "K2",
        "text": (
            "Target behavioral: acc_A - max(acc_{B.1}, acc_{B.2}) >= 0.03 "
            "with 95% CI lower bound > 0 on MMLU-Pro n=1000 at N=5 composition"
        ),
        "result": "untested",
        "note": "Phase C deferred (after Phase B trainings complete).",
    },
    {
        "id": "K3",
        "text": "Proxy confirmation: median(dev_D on trained DoRA) > 1e-3",
        "result": "untested",
        "note": "Phase C deferred (requires trained DoRA artifacts).",
    },
    {
        "id": "K4",
        "text": "Rank ablation: sign of K2 stable across r=6 and r=8",
        "result": "untested",
        "note": "Phase B+C deferred at second rank.",
    },
]


def phase_a_inspect():
    """Load Gemma 4 E4B 4-bit + topology readout. Returns A1/A2/A3 dict."""
    readout = {
        "A1_base_loads": {"status": "untested", "note": ""},
        "A2_v_proj_o_proj_present": {"status": "untested", "note": ""},
        "A3_dora_available": {"status": "untested", "note": ""},
        "wall_clock_s": None,
    }

    t0 = time.time()
    try:
        import mlx_lm
        readout["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception as e:
        readout["A1_base_loads"]["status"] = "fail"
        readout["A1_base_loads"]["note"] = f"mlx_lm import failed: {e!r}"
        readout["wall_clock_s"] = time.time() - t0
        return readout

    # A3: DoRA support detection (no model load needed for this).
    try:
        from mlx_lm.tuner import lora as lora_mod
        dora_symbols = [s for s in dir(lora_mod) if "dora" in s.lower()]
        readout["A3_dora_available"]["status"] = "pass" if dora_symbols else "fail"
        readout["A3_dora_available"]["note"] = (
            f"dora-related symbols in mlx_lm.tuner.lora: {dora_symbols}"
        )
    except Exception as e:
        readout["A3_dora_available"]["status"] = "fail"
        readout["A3_dora_available"]["note"] = f"mlx_lm.tuner.lora import failed: {e!r}"

    # A1 + A2: load model and inspect v_proj/o_proj presence per F#627.
    repo_id = "mlx-community/gemma-4-e4b-it-4bit"
    try:
        from mlx_lm import load
        model, tokenizer = load(repo_id)
        readout["A1_base_loads"]["status"] = "pass"
        readout["A1_base_loads"]["note"] = f"loaded {repo_id}"

        # Enumerate v_proj + o_proj across layers.
        v_count, o_count, layer_count = 0, 0, 0
        first_layer_names = []
        seen_layers = set()
        for name, _module in model.named_modules() if hasattr(model, "named_modules") else []:
            if ".layers." in name:
                # extract layer index
                try:
                    idx = name.split(".layers.")[1].split(".")[0]
                    seen_layers.add(idx)
                except Exception:
                    pass
            if name.endswith(".v_proj"):
                v_count += 1
                if len(first_layer_names) < 5:
                    first_layer_names.append(name)
            if name.endswith(".o_proj"):
                o_count += 1
        layer_count = len(seen_layers)

        # Fallback: walk the tree manually if named_modules absent.
        if v_count == 0:
            for attr_path in _walk_module_paths(model, max_depth=8):
                if attr_path.endswith(".v_proj"):
                    v_count += 1
                if attr_path.endswith(".o_proj"):
                    o_count += 1
                if len(first_layer_names) < 5 and attr_path.endswith(".v_proj"):
                    first_layer_names.append(attr_path)

        readout["A2_v_proj_o_proj_present"]["status"] = (
            "pass" if (v_count >= 1 and o_count >= 1) else "fail"
        )
        readout["A2_v_proj_o_proj_present"]["note"] = (
            f"v_proj count={v_count}, o_proj count={o_count}, "
            f"distinct .layers.* indices={layer_count}, "
            f"first_v_proj_paths={first_layer_names[:5]}"
        )

        # Cleanup.
        del model
        del tokenizer
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()

    except Exception as e:
        readout["A1_base_loads"]["status"] = "fail"
        readout["A1_base_loads"]["note"] = f"mlx_lm.load({repo_id!r}) failed: {e!r}"

    readout["wall_clock_s"] = round(time.time() - t0, 2)
    return readout


def _walk_module_paths(obj, prefix="", max_depth=8, depth=0):
    """Yield dotted attribute paths for nn.Module-like objects (fallback enumeration)."""
    if depth > max_depth:
        return
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        # Heuristic: child module is a callable with parameters
        if hasattr(val, "parameters") and not callable(val.parameters):
            continue
        path = f"{prefix}.{attr}" if prefix else attr
        if hasattr(val, "__class__") and val.__class__.__module__.startswith("mlx"):
            yield path
            yield from _walk_module_paths(val, path, max_depth, depth + 1)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if hasattr(item, "__class__") and item.__class__.__module__.startswith("mlx"):
                    yield from _walk_module_paths(item, f"{path}.{i}", max_depth, depth + 1)


def main():
    out_dir = Path(__file__).parent
    results_path = out_dir / "results.json"

    phase_a = phase_a_inspect()

    blockers = {
        "B1_molora_module": {
            "description": "MoLoRA has no turn-key mlx_lm mode.",
            "fix": "Custom micro/utils/molora.py with N experts + softmax gate.",
        },
        "B2_15_trainings": {
            "description": "3 classes x 5 domains = 15 LoRA/DoRA/MoLoRA trainings, "
                           "~30-60 min each at Gemma 4 E4B 4-bit = 8-15h wall-clock.",
            "fix": "Phase B follow-up at P=3 with dedicated compute budget; "
                   "mx.clear_cache() between trainings (F#673).",
        },
        "B3_domain_corpora": {
            "description": "5 non-overlapping domain corpora not yet curated.",
            "fix": "HumanEval/GSM8K/PubMedQA/CaseHOLD/LegalBench with non-overlap audit.",
        },
        "B4_composition_eval": {
            "description": "N=5 composition-eval harness does not exist for DoRA / MoLoRA.",
            "fix": "Phase C harness: load adapters, apply class formula, "
                   "MMLU-Pro n=1000 with enable_thinking=False (F#793/F#795 mitigation).",
        },
        "B5_bootstrap_statistics": {
            "description": "Paired bootstrap 95% CI not yet implemented.",
            "fix": "numpy bootstrap, 10000 resamples, 3 pairwise comparisons.",
        },
    }

    results = {
        "experiment_id": "exp_g4_adapter_class_composition_full_impl",
        "verdict": "PROVISIONAL",
        "is_smoke": False,
        "is_design_only": False,
        "is_phase_a_executable_slice": True,
        "all_pass": None,
        "kill_criteria": KILL_CRITERIA,
        "blockers": blockers,
        "phase_a_readout": phase_a,
        "iteration_scope": {
            "researcher_iteration_budget_min": 30,
            "phase_a_executable_slice_runtime_s": phase_a.get("wall_clock_s"),
            "estimated_full_pipeline_h": "8-15",
            "verdict_reason": (
                "Phase A topology readout completed within single-iteration budget. "
                "Phase B-E (15 trainings + N=5 harness + bootstrap) realistically "
                "8-15h, deferred to same-dir follow-up at P=3. Per "
                "mem-antipattern-novel-mechanism-single-iteration-scope option (ii) "
                "and reviewer.md §5 'Phase A executable slice' precedent "
                "(memento_replication_impl iter ~104, jepa_adapter_residual_stream_impl)."
            ),
        },
        "reference_parent": {
            "experiment_id": "exp_g4_adapter_class_composition_full",
            "finding_id": 686,
            "status": "provisional",
            "note": (
                "Parent is design-only PROVISIONAL. This _impl follow-up adds "
                "Phase A executable slice (model load + F#627 target inspect + "
                "DoRA availability check) within single-iter budget. K1-K4 "
                "remain untested in this iter; future Phase B-E execution "
                "on this dir at P=3 will close them."
            ),
        },
        "antipattern_compliance": {
            "mem_novel_mechanism_single_iteration_scope": "applied (option ii: Phase A executable slice)",
            "mem_proxy_kc_mislabeled_target": "K2 behavioral, K3 proxy — F#666 target-gating preserved",
            "mem_preempt_child_parent_target_unverified": "N/A (parent design-only is intentional deferral; this is designated _impl)",
            "mem_thinking_mode_truncates_judge_budget": "N/A this iter (no eval); Phase B-E must apply enable_thinking=False per F#793/795/797/798",
            "mem_researcher_prefiles_finding_before_review": "HONORED — no finding pre-fill (4th consecutive observance post-mitigation)",
            "mem_finding_add_scratchpad_drift": "verify via finding-list before any future citation",
            "mem_schema_incomplete": "all 4 KCs reference trained-object properties (verbatim from parent)",
            "scope_preservation_F1_F5": "all 5 forbid-list items binding for future Phase B execution",
            "F666_target_gating": "matrix unchanged from parent §3",
            "F669_cascade_cluster": "N/A (Phase A is pure inspection, no schema-broken deps)",
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[PROVISIONAL] Wrote {results_path}")
    print(f"[PHASE_A] runtime={phase_a.get('wall_clock_s')}s")
    print(f"[PHASE_A] A1_base_loads={phase_a['A1_base_loads']['status']}")
    print(f"[PHASE_A] A2_v_proj_o_proj={phase_a['A2_v_proj_o_proj_present']['status']}")
    print(f"[PHASE_A] A3_dora_available={phase_a['A3_dora_available']['status']}")
    print(f"[PROVISIONAL] 5 structural blockers (B1-B5); Phase B-E deferred to P=3 follow-up.")
    print(f"[PROVISIONAL] 4 KCs untested; see results.json.kill_criteria")
    return 0


if __name__ == "__main__":
    sys.exit(main())
