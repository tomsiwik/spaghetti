#!/usr/bin/env bash
# Stop hook — keep the conductor going until the in-flight experiment reaches a terminal verdict.
# Blocks the lead from stopping while any experiment is still ACTIVE (claimed, no verdict yet);
# allows the stop once nothing is active. A counter cap prevents trapping the user forever.
set -uo pipefail
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export PATH="$HOME/.local/bin:$HOME/.vite-plus/bin:$HOME/.bun/bin:/opt/homebrew/bin:$PATH"
CNT="${TMPDIR:-/tmp}/conductor_stop_count"

ACTIVE=$(cd "$ROOT" && experiment list -s active 2>/dev/null | grep -oE 'exp_[A-Za-z0-9_]+' | head -1)
if [ -z "$ACTIVE" ]; then
  rm -f "$CNT"
  exit 0   # nothing in flight -> allow the stop
fi

N=$(cat "$CNT" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CNT"
if [ "$N" -gt 60 ]; then
  rm -f "$CNT"
  ACTIVE="$ACTIVE" python3 -c 'import json,os
print(json.dumps({"hookSpecificOutput":{"hookEventName":"Stop","additionalContext":
"Conductor stop-gate gave up after 60 continuations; "+os.environ["ACTIVE"]+" is still active. Manual attention needed."}}))'
  exit 0
fi

ACTIVE="$ACTIVE" python3 -c 'import json,os
exp=os.environ["ACTIVE"]
print(json.dumps({"decision":"block","reason":
f"Experiment {exp} is still ACTIVE with no final verdict. Do NOT stop. Continue the loop: researcher (real run, NO mocks) -> reviewer (adversarial no-mock gate, calls experiment complete) -> analyst (finding-add), until {exp} reaches a terminal status (supported/killed/provisional) AND a finding is recorded. Then you may stop."}))'
exit 0
