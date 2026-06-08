#!/usr/bin/env bash
# verify-experiment.sh <id-or-dir>
# Exit 0 if the experiment has a REAL result; exit 1 (with a reason on stdout) if it's a
# mock / smoke / incomplete. Defense-in-depth behind the Reviewer's adversarial checklist —
# its single job is to make the exact failure that burned us (a smoke/numpy placeholder
# completed as "supported") IMPOSSIBLE. Subtler fabrication is the Reviewer's call.
set -uo pipefail
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
ARG="${1:-}"
[ -z "$ARG" ] && { echo "verify: no experiment id/dir given"; exit 1; }

if   [ -d "$ARG" ];            then DIR="$ARG"
elif [ -d "$ROOT/$ARG" ];      then DIR="$ROOT/$ARG"
else DIR="$ROOT/experiments/models/$ARG"; fi
RES="$DIR/results.json"
CODE="$DIR/run_experiment.py"

[ -f "$RES" ] || { echo "verify: no results.json in ${DIR#$ROOT/} — experiment did not produce a real result"; exit 1; }

python3 - "$RES" "$CODE" <<'PY'
import json, os, re, sys
res_path, code_path = sys.argv[1], sys.argv[2]
try:
    res = json.load(open(res_path))
except Exception as e:
    print(f"verify: results.json is not valid JSON ({e})"); sys.exit(1)

hard = []   # block supported/killed
warn = []   # surface to reviewer, do not block

if res.get("is_smoke") is True:
    hard.append("is_smoke=true (smoke/placeholder result — only 'provisional' is allowed, never supported/killed)")
if not str(res.get("verdict", "")).strip():
    hard.append("results.json has no non-empty 'verdict' field")

code = ""
if os.path.exists(code_path):
    code = open(code_path, encoding="utf-8", errors="ignore").read()
else:
    hard.append("no run_experiment.py next to results.json")

low = code.lower()
loads_model = any(k in code for k in
    ["mlx_lm", "mlx.core", "import mlx", "from mlx", ".load(", "from_pretrained", "load_model", "gemma", "AutoModel"])
numpy_only = ("import numpy" in low or "numpy as np" in low) and not loads_model
if numpy_only:
    warn.append("run_experiment.py appears numpy-only with no real model load — confirm this is a genuine analytical experiment, not a placeholder")
if "shutil.copy" in code and "adapter" in low:
    warn.append("run_experiment.py copies a sibling adapter ('shutil.copy' + adapter) — confirm it is not faking a new domain")
if re.search(r'["\']pass["\']\s*:\s*True', code):
    warn.append("run_experiment.py contains a literal '\"pass\": True' — confirm results are computed, not hardcoded")

if hard:
    print("verify: MOCK/INCOMPLETE — " + "; ".join(hard) + ((" | warnings: " + "; ".join(warn)) if warn else ""))
    sys.exit(1)
msg = "verify: REAL result ok (is_smoke!=true, verdict present" + (", model-backed" if loads_model else "") + ")"
if warn:
    msg += " | reviewer-check: " + "; ".join(warn)
print(msg); sys.exit(0)
PY
