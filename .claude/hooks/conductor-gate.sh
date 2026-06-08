#!/usr/bin/env bash
# PreToolUse(Bash) gate — DENY `experiment complete … --status supported|killed` when the
# experiment's result is a mock/smoke. This makes "fake green" structurally impossible.
# Reads the hook JSON on stdin; emits a deny decision as JSON when it blocks.
set -uo pipefail
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
INPUT=$(cat)
CMD=$(printf '%s' "$INPUT" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)

# Only gate strong terminal completions; everything else flows normally.
case "$CMD" in
  *"experiment complete"*"--status supported"*|*"experiment complete"*"--status killed"*) : ;;
  *) exit 0 ;;
esac

# Prefer the explicit --dir; fall back to the id right after "experiment complete".
DIR=$(printf '%s' "$CMD" | sed -nE 's/.*--dir[= ]+([^ ]+).*/\1/p')
ID=$(printf '%s'  "$CMD" | sed -nE 's/.*experiment complete[[:space:]]+([^ -][^ ]*).*/\1/p')
TARGET="${DIR:-$ID}"

REASON=$(bash "$ROOT/.claude/hooks/verify-experiment.sh" "$TARGET" 2>&1)
if [ $? -ne 0 ]; then
  REASON="$REASON" python3 -c 'import json,os
print(json.dumps({"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny",
"permissionDecisionReason":"NO-MOCK GATE blocked this completion. "+os.environ.get("REASON","")+
" — Either run REAL code that produces a non-smoke results.json with a measured target-metric verdict, or mark the experiment provisional / BLOCKED. Do not complete a placeholder as supported/killed."}}))'
fi
exit 0
