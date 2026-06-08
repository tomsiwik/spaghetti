#!/usr/bin/env python
"""Rank spark prompts from a promptfoo eval JSON.

Metrics per prompt:
  reframe       mean of the llm-rubric 'reframe' assertion (judge: reframe vs reconstruct)
  novelty_j     mean of the llm-rubric 'novelty' assertion (judge)
  diversity     mean pairwise cosine DISTANCE among the `repeat` samples of the SAME
                problem, averaged over problems (how widely the prompt explores) — the
                cross-output metric vanilla promptfoo can't compute per-output
  novelty_e     mean cosine DISTANCE of each output from the problem's obvious/textbook
                approach (embedding-based novelty), if problems.json is provided

Usage: python score.py eval_v3.json [problems.json]
"""
import json, sys, collections, itertools
import numpy as np
from sentence_transformers import SentenceTransformer

eval_path = sys.argv[1]
problems_path = sys.argv[2] if len(sys.argv) > 2 else None
obvious = {}
if problems_path:
    for p in json.load(open(problems_path)):
        obvious[p["problem"].strip()] = p.get("obvious_approach", "")

d = json.load(open(eval_path))
rows = d["results"]["results"]

def pname(r):
    l = (r.get("prompt") or {}).get("label", "")
    return l.split("prompts/")[-1].split(".md")[0] if "prompts/" in l else l

# group: prompt -> problem -> list of (output, scores{})
G = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    out = ((r.get("response") or {}).get("output") or "").strip()
    sc = {}
    for c in (r.get("gradingResult") or {}).get("componentResults") or []:
        m = (c.get("assertion") or {}).get("metric") or "?"
        if c.get("score") is not None: sc[m] = c["score"]
    G[pname(r)][(r.get("vars") or {}).get("problem", "").strip()].append((out, sc))

print("embedding outputs...", file=sys.stderr)
emb = SentenceTransformer("all-MiniLM-L6-v2")
def vecs(texts):
    texts = [t if t else " " for t in texts]
    return emb.encode(texts, normalize_embeddings=True)

def cos_dist(a, b): return 1.0 - float(np.dot(a, b))

rows_out = []
for prompt, byprob in G.items():
    rf, nv, divs, nve, n = [], [], [], [], 0
    for problem, samples in byprob.items():
        outs = [o for o, _ in samples]
        for _, s in samples:
            if "reframe" in s: rf.append(s["reframe"])
            if "novelty" in s: nv.append(s["novelty"])
        valid = [o for o in outs if o]
        if len(valid) >= 2:
            V = vecs(valid)
            pd = [cos_dist(V[i], V[j]) for i, j in itertools.combinations(range(len(V)), 2)]
            divs.append(float(np.mean(pd)))
        if obvious.get(problem) and valid:
            ob = vecs([obvious[problem]])[0]
            VV = vecs(valid)
            nve.extend(cos_dist(v, ob) for v in VV)
        n += len(outs)
    def m(x): return float(np.mean(x)) if x else float("nan")
    rows_out.append({
        "prompt": prompt, "n": n,
        "reframe": m(rf), "novelty_j": m(nv),
        "diversity": m(divs), "novelty_e": m(nve),
    })

base = next((r for r in rows_out if r["prompt"] == "baseline"), None)
def composite(r):  # rank signal: reframe + judge-novelty + exploration(diversity) + embedding-novelty
    parts = [r["reframe"], r["novelty_j"], r["diversity"], r["novelty_e"]]
    parts = [p for p in parts if p == p]
    return float(np.mean(parts)) if parts else float("nan")
for r in rows_out: r["composite"] = composite(r)

w = max(len(r["prompt"]) for r in rows_out) + 2
print(f"\n{'prompt':<{w}}{'reframe':>9}{'novelty_j':>11}{'diversity':>11}{'novelty_e':>11}{'composite':>11}{'n':>5}")
print("-" * (w + 58))
for r in sorted(rows_out, key=lambda r: -(r["composite"] if r["composite"] == r["composite"] else -9)):
    tag = " (baseline)" if r["prompt"] == "baseline" else ""
    def f(x): return f"{x:.2f}" if x == x else "  - "
    print(f"{r['prompt']:<{w}}{f(r['reframe']):>9}{f(r['novelty_j']):>11}{f(r['diversity']):>11}{f(r['novelty_e']):>11}{f(r['composite']):>11}{r['n']:>5}{tag}")
if base:
    print(f"\nbaseline composite = {base['composite']:.2f}  | lift = (prompt - baseline):")
    for r in sorted(rows_out, key=lambda r: -(r["composite"] if r["composite"] == r["composite"] else -9)):
        if r["prompt"] != "baseline" and r["composite"] == r["composite"]:
            print(f"  {r['prompt']:<{w}} {r['composite']-base['composite']:+.2f}")
