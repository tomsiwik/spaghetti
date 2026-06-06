# experiments/ — the research log

The honest, append-only record of every experiment. ~887 experiments, each self-contained in its own folder. **Frozen history** — read it, don't rewrite it. Live status is in the root `STATUS.md`.

> **How to run an experiment → [`GUIDE.md`](GUIDE.md)** (the canonical process: loop, proof-first discipline, kill/verdict rules, CLI, platform rules).

## Layout

```
experiments/
├── __init__.py, data.py, train.py, arena.py, metrics.py
│       shared MLX helpers imported by experiments as `from experiments.X import …`
├── models/        ← micro-scale experiments (local, MLX, Apple Silicon)   [the bulk: ~735]
│   └── <exp>/      one folder per experiment:
│         MATH.md            theorem + predictions + pre-registered kill criteria
│         run_experiment.py  the experiment (run via `experiment run <id>`)
│         results.json       measured numbers (gitignored — local only)
│         PAPER.md           prediction vs measurement
│         REVIEW-adversarial.md   adversarial check
├── macro/         ← macro-scale experiments (larger models, GPU/RunPod, torch)
│       Off the MLX-first path; mostly frozen RunPod records (paths were /workspace/llm).
└── _runs/         ← jobs/ + logs/ : raw run + job logs
```

## micro vs macro (the two scales)

| | **micro** (`models/`) | **macro** (`macro/`) |
|---|---|---|
| Runs on | Apple Silicon, MLX-native (the target platform) | rented GPUs (RunPod), torch/CUDA |
| Status | active — this is where current work happens | frozen records; off-strategy under MLX-first |
| In the DB | `scale: micro` | `scale: macro` |

## How paths resolve

The experiment DB (Turso) stores each `experiment_dir` as e.g. `experiments/models/<name>/`. `experiment run <id>` looks it up and runs `<experiment_dir>/run_experiment.py` from repo root. Experiment scripts import shared code as `from experiments.<module>` and reach repo-root resources via a `REPO_ROOT` computed from `__file__`. Trained adapter weights live in `../data/adapters/` (referenced as `REPO_ROOT / "data" / "adapters"`).

> Reorganized 2026-06-04 (was `micro/` + `macro/` at repo root). See `STATUS.md` §7.
