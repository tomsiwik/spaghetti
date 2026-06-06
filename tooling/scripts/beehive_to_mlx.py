#!/usr/bin/env python3
"""
beehive_to_mlx — convert beehive trajectories to mlx_lm.lora-compatible JSONL.

DATA SOURCE: live Turso DB via `bee training export` (NOT local SQLite snapshot).
The local `beehive.db` is stale; truth lives on Turso.

Usage (as library):
    from tooling.scripts.beehive_to_mlx import export_split

    export_split(
        out_dir=Path("data/corpora/beehive_approved"),
        quality="approved",
        traj_type=None,           # or "prepare" | "act" | "integrate" | "full"
        skill_id=None,            # int | None
        min_score=None,           # int | None
        val_frac=0.2,
        seed=42,
        balance_stratify=True,    # stratified split by trajectory type
    )

Produces:
    <out_dir>/train.jsonl
    <out_dir>/valid.jsonl
    <out_dir>/manifest.json   (counts, stratification, per-row provenance)

Each row: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
— directly consumable by `mlx_lm.lora` with `mask_prompt: true`.

Implementation: shells out to `bee training export -f messages` to get full
remote-DB state, then handles split/stratification locally.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BEEHIVE_DIR = REPO_ROOT.parent / "beehive"
SNAPSHOT_DIR = REPO_ROOT / "data" / "corpora" / "beehive_snapshot"


@dataclass
class TrajRow:
    id: int
    skill_id: int
    skill_name: str
    type: str
    principle: str
    trajectory: str         # the assistant message content (token-format trajectory)
    quality: str
    score: Optional[int]
    user_prompt: str        # the user message content (skill+type+principle prompt)


def _read_snapshot(quality: str) -> list[dict]:
    """Read the canonical local snapshot for a given quality bucket.

    Snapshots are saved at data/beehive_snapshot/{approved,rejected}.jsonl. This is
    point-in-time data; refresh by running:
        cd ../beehive && bee training export --quality {q} --format messages \\
                       --skip-validation --output ../llm/data/beehive_snapshot/{q}.jsonl
    """
    path = SNAPSHOT_DIR / f"{quality}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Beehive snapshot missing: {path}\n"
            "Run from beehive dir: bee training export --quality {q} --format messages "
            "--skip-validation --output ../llm/data/beehive_snapshot/{q}.jsonl"
        )
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bee_export(quality: str, traj_type: Optional[str], skill_id: Optional[int],
                min_score: Optional[int]) -> list[dict]:
    """Read from local snapshot (data/beehive_snapshot/) and apply filters.

    Snapshot is point-in-time; data freshness depends on when it was last updated.
    See _read_snapshot docstring for refresh command. This snapshot-first approach
    avoids Turso 401 / network instability that broke earlier runs.
    """
    rows = _read_snapshot(quality)

    if traj_type:
        rows = [r for r in rows if r.get("type") == traj_type]
    if skill_id is not None:
        rows = [r for r in rows if r.get("skill_id") == skill_id]
    if min_score is not None:
        rows = [r for r in rows if r.get("score") is not None and r["score"] >= min_score]

    return rows


def fetch_rows(
    quality: Optional[str] = "approved",
    traj_type: Optional[str] = None,
    skill_id: Optional[int] = None,
    min_score: Optional[int] = None,
) -> list[TrajRow]:
    """Live read from Turso via bee CLI. Returns TrajRow objects.

    bee export schema per row:
        {"id", "skill_id", "skill_name", "skill_source", "type", "principle",
         "quality", "score", "synthesis", "messages": [system, user, assistant]}
    """
    if quality is None or quality == "all":
        # bee CLI doesn't have an "all" mode; union approved + rejected
        approved = _bee_export("approved", traj_type, skill_id, min_score)
        rejected = _bee_export("rejected", traj_type, skill_id, min_score)
        raw_rows = approved + rejected
    else:
        raw_rows = _bee_export(quality, traj_type, skill_id, min_score)

    out: list[TrajRow] = []
    for r in raw_rows:
        msgs = r.get("messages", [])
        # bee schema: [system, user, assistant]. We only use user+assistant.
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        out.append(TrajRow(
            id=r["id"],
            skill_id=r["skill_id"],
            skill_name=r["skill_name"],
            type=r["type"],
            principle=r["principle"],
            trajectory=assistant_msg,
            quality=r["quality"],
            score=r.get("score"),
            user_prompt=user_msg,
        ))
    return out


def _build_user_prompt(row: TrajRow) -> str:
    """Use bee's canonical user prompt verbatim (same one the synthesis worker saw)."""
    return row.user_prompt


def _to_messages_record(row: TrajRow) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row.user_prompt},
            {"role": "assistant", "content": row.trajectory},
        ],
        "_meta": {
            "training_id": row.id,
            "skill_id": row.skill_id,
            "skill_name": row.skill_name,
            "type": row.type,
            "quality": row.quality,
            "score": row.score,
        },
    }


def _stratified_split(rows: list[TrajRow], val_frac: float, seed: int) -> tuple[list, list]:
    """Stratified by `type` only (full/prepare/act/integrate). Skill-level strata are too
    granular at the current dataset size (35 skills × 4 types ≈ 140 cells / 104 rows).

    Each type-stratum yields max(1, round(n*val_frac)) val rows when n ≥ 3, else 1 val
    when n == 2, else all-train when n == 1.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[TrajRow]] = defaultdict(list)
    for r in rows:
        by_type[r.type].append(r)

    train, val = [], []
    for t, items in by_type.items():
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            train.append(items[0])
        elif n == 2:
            train.append(items[0])
            val.append(items[1])
        else:
            n_val = max(1, int(round(n * val_frac)))
            val.extend(items[:n_val])
            train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def export_split(
    out_dir: Path,
    quality: Optional[str] = "approved",
    traj_type: Optional[str] = None,
    skill_id: Optional[int] = None,
    min_score: Optional[int] = None,
    val_frac: float = 0.2,
    seed: int = 42,
    balance_stratify: bool = True,
    keep_meta_in_manifest_only: bool = True,
) -> dict:
    """Export a train/valid split to JSONL files. Returns manifest dict."""
    rows = fetch_rows(quality=quality, traj_type=traj_type, skill_id=skill_id, min_score=min_score)
    if not rows:
        raise ValueError(f"No rows matched filters quality={quality} type={traj_type} skill_id={skill_id} min_score={min_score}")

    if balance_stratify:
        train_rows, val_rows = _stratified_split(rows, val_frac, seed)
    else:
        rng = random.Random(seed)
        rng.shuffle(rows)
        cut = int(len(rows) * (1 - val_frac))
        train_rows, val_rows = rows[:cut], rows[cut:]

    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(jsonl_path: Path, items: list[TrajRow]):
        with open(jsonl_path, "w") as f:
            for r in items:
                rec = _to_messages_record(r)
                if keep_meta_in_manifest_only:
                    rec_out = {"messages": rec["messages"]}
                else:
                    rec_out = rec
                f.write(json.dumps(rec_out) + "\n")

    _write(out_dir / "train.jsonl", train_rows)
    _write(out_dir / "valid.jsonl", val_rows)

    manifest = {
        "filters": {
            "quality": quality,
            "type": traj_type,
            "skill_id": skill_id,
            "min_score": min_score,
        },
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_valid": len(val_rows),
        "val_frac": val_frac,
        "seed": seed,
        "balance_stratify": balance_stratify,
        "stratification": _stratify_summary(train_rows, val_rows),
        "score_stats": _score_stats(rows),
        "train_ids": [r.id for r in train_rows],
        "valid_ids": [r.id for r in val_rows],
        "valid_meta": [
            {"id": r.id, "skill_name": r.skill_name, "type": r.type, "score": r.score, "principle": r.principle}
            for r in val_rows
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _stratify_summary(train: list[TrajRow], val: list[TrajRow]) -> dict:
    def _by(items, key):
        d: dict = defaultdict(int)
        for r in items:
            d[key(r)] += 1
        return dict(d)

    return {
        "train_by_type": _by(train, lambda r: r.type),
        "valid_by_type": _by(val, lambda r: r.type),
        "train_unique_skills": len({r.skill_id for r in train}),
        "valid_unique_skills": len({r.skill_id for r in val}),
    }


def _score_stats(rows: list[TrajRow]) -> dict:
    scores = [r.score for r in rows if r.score is not None]
    if not scores:
        return {"n_with_score": 0}
    return {
        "n_with_score": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "median": sorted(scores)[len(scores) // 2],
    }


def fetch_eval_pairs(val_ids: list[int]) -> list[dict]:
    """Return rows for principle-following eval — used by experiment eval harnesses.

    Each pair: {"prompt": <user>, "expected": <trajectory>, "principle": ..., "skill_name": ..., "type": ..., "score": ...}

    Pulls from live Turso (via fetch_rows). No fast indexed lookup; we filter client-side.
    """
    if not val_ids:
        return []
    val_set = set(val_ids)
    # We don't know the quality of held-out rows, so check both buckets
    all_rows = _bee_export("approved", None, None, None) + _bee_export("rejected", None, None, None)
    out = []
    for r in all_rows:
        if r["id"] not in val_set:
            continue
        msgs = r.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        out.append({
            "id": r["id"],
            "prompt": user_msg,
            "expected": assistant_msg,
            "principle": r["principle"],
            "skill_name": r["skill_name"],
            "type": r["type"],
            "score": r.get("score"),
        })
    return out


# ─────────────────────────────────────────────
# Principle-following eval
# ─────────────────────────────────────────────

def principle_following_score(generated: str, expected: str, principle: str) -> dict:
    """Lightweight intrinsic eval: does generated trajectory match the GOLD trajectory.

    Three signals (no model judge needed; deterministic):

    1. token-format compliance: contains expected channel marker(s) like "<|channel>prepare"
    2. trajectory-keyword recall: content words from EXPECTED trajectory present in generated.
       NOT principle keywords — the prompt contains the principle, so base model scored 100%
       by echoing. Measuring against the held-out gold trajectory is the non-tautological version.
    3. structural overlap: section labels (goal:, decomposition:, ...) match between gold and generated.

    Returns dict with each subscore + a weighted aggregate in [0,1].
    """
    import re

    stop = {"that", "this", "with", "from", "into", "when", "than", "their", "your", "have",
            "been", "must", "will", "what", "which", "where", "should", "would", "could",
            "about", "every", "after", "before", "while", "until", "because", "those", "these",
            "they", "them", "some", "more", "less", "also", "such", "even", "both", "each",
            "very", "well", "back", "much", "many", "most", "make", "made", "does", "doing"}

    # 1. Token format compliance — channel markers in generated match expected's
    expected_channels = set(re.findall(r"<\|channel>(\w+)", expected))
    generated_channels = set(re.findall(r"<\|channel>(\w+)", generated))
    fmt = (
        len(expected_channels & generated_channels) / len(expected_channels)
        if expected_channels else (1.0 if not generated_channels else 0.5)
    )

    # 2. Keyword recall vs GOLD trajectory (not principle text → prevents prompt-echo tautology)
    gold_kws = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", expected) if w.lower() not in stop}
    if gold_kws:
        gen_low = generated.lower()
        hits = sum(1 for k in gold_kws if k in gen_low)
        kw_recall = hits / len(gold_kws)
    else:
        kw_recall = 0.0

    # 3. Section labels match (goal:, decomposition:, etc.)
    label_re = re.compile(r"(?m)^([a-z_]{3,30}):", re.IGNORECASE)
    expected_labels = [m.lower() for m in label_re.findall(expected)]
    generated_labels = [m.lower() for m in label_re.findall(generated)]
    if expected_labels:
        exp_set = set(expected_labels)
        gen_set = set(generated_labels)
        struct = len(exp_set & gen_set) / len(exp_set)
    else:
        struct = 1.0 if not generated_labels else 0.5

    aggregate = 0.4 * fmt + 0.3 * kw_recall + 0.3 * struct
    return {
        "format": round(fmt, 3),
        "keyword_recall": round(kw_recall, 3),  # NOTE: now vs GOLD trajectory, not principle
        "structure": round(struct, 3),
        "aggregate": round(aggregate, 3),
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export beehive trajectories to mlx_lm JSONL")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quality", default="approved", choices=["approved", "rejected", "all"])
    ap.add_argument("--type", dest="traj_type", default=None,
                    choices=["full", "prepare", "act", "integrate"])
    ap.add_argument("--skill-id", type=int, default=None)
    ap.add_argument("--min-score", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = export_split(
        out_dir=args.out_dir,
        quality=args.quality,
        traj_type=args.traj_type,
        skill_id=args.skill_id,
        min_score=args.min_score,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
