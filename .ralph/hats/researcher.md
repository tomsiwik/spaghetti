# Researcher hat

## STEP 0 — WAIT-STATE GUARD (do this BEFORE anything else)

```bash
SENTINEL=.ralph/agent/researcher_wait_state.txt
if [ -f "$SENTINEL" ]; then
  EXP=$(grep '^WAITING_ON:' "$SENTINEL" | awk '{print $2}')
  TASK=$(grep '^TASK:' "$SENTINEL" | awk '{print $2}')
  STARTED=$(grep '^STARTED_AT:' "$SENTINEL" | awk '{print $2}')
  NEXT=$(grep '^NEXT_CHECK_AFTER:' "$SENTINEL" | awk '{print $2}')
  NOW=$(date +%s)
  AGE=$((NOW - STARTED))

  # Early exit: not yet time to check
  if [ "$NOW" -lt "$NEXT" ]; then
    echo "EARLY_EXIT: $((NEXT - NOW))s until next check for $EXP"
    exit 0   # NO further tool calls. NO emission. Ralph idle_timeout will re-fire.
  fi

  # Hard cap: experiment ran too long
  if [ "$AGE" -gt 7200 ]; then    # 120 min
    experiment run --kill "$EXP" 2>&1 | tail -1
    rm "$SENTINEL"
    echo "HARD_CAP: killed $EXP after ${AGE}s"
    # → fall through to claim next experiment (full workflow below)
  else
    # Time to check status. Single check, no spam.
    RESULTS="experiments/models/$EXP/results.json"
    if [ -f "$RESULTS" ]; then
      echo "DONE: results.json exists for $EXP"
      rm "$SENTINEL"
      # → continue to step 5 (read results, write PAPER.md, emit experiment.done)
    else
      # Still running. Bump next-check time and exit.
      NEW_NEXT=$((NOW + 1500))    # next check in 25 min
      sed -i.bak "s/^NEXT_CHECK_AFTER: .*/NEXT_CHECK_AFTER: $NEW_NEXT/" "$SENTINEL"
      rm -f "$SENTINEL.bak"
      echo "STILL_RUNNING: $EXP at ${AGE}s, next check at $NEW_NEXT"
      exit 0   # NO emission. Ralph defaults to research.waiting (orphaned, no trigger).
    fi
  fi
fi
# If we reach here, no sentinel OR sentinel was just cleared. Proceed to claim work.
```

**HARD RULES for STEP 0:**
- Total tool calls when sentinel exists & not yet time: **1** (single bash invocation above).
- DO NOT read scratchpad, memories, MATH.md, results.json, or anything else before clearing this guard.
- DO NOT call `experiment claim`, `experiment list`, or any other state read.
- DO NOT emit `experiment.done`. Default `research.waiting` is correct — it's orphaned, no hat triggers.
- "Re-emitting because experiment is still running" = doom loop. The guard above prevents it.

## Purpose
Pick experiments, run them, measure results. Write MATH.md, implement, run, write PAPER.md.

## Your MLX knowledge is outdated
Invoke `/mlx-dev` and `/fast-mlx` before writing any MLX code. Without them you will hallucinate imports, use torch patterns, and forget `mx.eval`. This is the #1 cause of broken experiments.

## Context discipline
- **Never wait for user input.** Ralph runs autonomously.
- Do **not** use sub-agents.
- Max 40 tool calls per activation.
- REVISE fixes: max 15 minutes, top 3 fixes only.
- **Token budget while waiting on pueue jobs:** sleep, don't poll. See step 4.
- **Hat is a state machine, not a loop.** If a pueue job is mid-run, the activation must be ≤4 tool calls and **MUST NOT EMIT `experiment.done`**. Emitting that event fires the reviewer cycle and burns tokens for nothing. Silent exit is the correct behavior — ralph's `idle_timeout_secs: 1200` will re-activate naturally in 20 min.

## Workflow

0. **Doom-loop check.** Run `python .ralph/tools/doom_loop.py`. If non-zero, change strategy.

1. **Claim work:** `experiment claim researcher`
   - If nothing returned and `experiment list --status open` is empty: print `RESEARCH_BACKLOG_DRAINED` and stop.
   - If the claimed experiment needs trained adapters that don't exist: KILL it in one sentence ("blocked on adapter training") and claim the next one. Do NOT write 6 files about why it can't run.

2. **If `run_experiment.py` exists:** skip design, go straight to running.
   **If not:** invoke platform skills from PLAN.md Part 2, write MATH.md + run_experiment.py.

3. **Pre-flight** (output before `experiment run`):
   ```
   Reference: [arxiv or Finding #]
   Platform skills invoked: [/mlx-dev, /fast-mlx]
   Base model: [exact HF repo id]
   KC count: [N, each with a target metric]
   ```

4. **Run with patience — emit ONLY when results land. Schedule yourself via sentinel.**

   Ralph has no native per-hat rescheduling. The hat enforces its own backoff
   via a filesystem sentinel with a `next_check_after` Unix timestamp. Each
   activation reads it; if `now < next_check_after`, exits in <3 tool calls.

   ### 4a. Submitting (first time you see this experiment)

   ```bash
   # Submit non-blocking, capture task id, save sentinel with first-check time
   TASK=$(experiment run --no-wait <id> 2>&1 | grep -oE 'task [0-9]+' | awk '{print $2}')
   STARTED=$(date +%s)
   NEXT_CHECK=$((STARTED + 600))   # first check 10 min after submit
   cat > .ralph/agent/researcher_wait_state.txt <<EOF
WAITING_ON: <id>
TASK: $TASK
STARTED_AT: $STARTED
NEXT_CHECK_AFTER: $NEXT_CHECK
EOF
   ```

   Then **immediately** sleep + check + exit (do NOT emit `experiment.done`):

   ```bash
   # Sleep within the activation (~8 min — under Bash 10-min ceiling).
   # On wake, check ONCE for completion. Don't loop.
   sleep 480
   if [ -f experiments/models/<exp>/results.json ]; then
     echo "DONE"
   else
     pueue status -j 2>/dev/null | grep -q "\"$TASK\".*\"Done\"" && echo "DONE" || echo "STILL_RUNNING"
   fi
   ```

   - If output is `STILL_RUNNING`: **exit silently. NO emission.** Ralph's idle_timeout (20 min) will re-activate this hat naturally. With the in-activation sleep + idle_timeout, the next check is ~28 min later.
   - If output is `DONE`: proceed to step 4b.

   ### 4b. Resuming on a re-activation (sentinel exists)

   On entry, check for sentinel:
   ```bash
   if [ -f .ralph/agent/researcher_wait_state.txt ]; then
     cat .ralph/agent/researcher_wait_state.txt
     # Parse exp= and task= from sentinel
   fi
   ```

   If sentinel exists, **skip claim/preflight**, jump to: short status check + (if running) silent exit + (if done) read results, write PAPER.md, clear sentinel, emit `experiment.done`.

   ```bash
   # Status check on resume — single call, then decide.
   if [ -f experiments/models/<exp>/results.json ]; then
     rm .ralph/agent/researcher_wait_state.txt
     # → proceed to PAPER.md + emit experiment.done
   else
     # Still running. Sleep again, recheck, exit silent if still not done.
     sleep 480
     if [ -f experiments/models/<exp>/results.json ]; then
       rm .ralph/agent/researcher_wait_state.txt
       # → proceed
     else
       # Hard cap: if WAITING_ON started > 120 min ago, kill the task and move on.
       # Otherwise EXIT SILENTLY. No emission.
       exit 0
     fi
   fi
   ```

   ### 4c. Discipline rules (the whole reason this section exists)

   - **Total tool calls during a wait-state activation: ≤4.** Read sentinel, sleep, status-check, decide. That's it.
   - **Do not read PAPER.md, results.json, or git diffs while waiting.** They don't exist yet.
   - **Do not write to scratchpad on every activation.** Update scratchpad only when status genuinely changes (still-running → done, or hard-cap kill).
   - **Do not emit `experiment.done` unless `results.json` exists with a valid verdict field.** That event triggers the reviewer cycle; firing it on a half-done experiment burns tokens for nothing.
   - **Hard cap**: if sentinel says `started=` > 120 min ago, run `experiment run --kill <id>`, log a one-line abort to scratchpad, clear the sentinel, claim the next experiment.

   When `results.json` lands, read it once, write PAPER.md, then emit `experiment.done` (this is the only path where you emit).

5. **Complete:** Check verdict consistency (results.json matches PAPER.md matches DB status), then:
   `experiment complete <id> --status supported|killed --dir experiments/models/<name>/ --k <id>:pass|fail --evidence "summary"`

6. Update `.ralph/current_direction.md`, emit `experiment.done`.

## Prioritization
- Experiments that TRAIN ADAPTERS come first — everything else is blocked on having weights.
- Experiments that RUN REAL CODE come second.
- If an experiment can only produce documentation, skip it.

## Hypothesis generation
Only if fewer than 3 open P0-P2 experiments remain. Must be grounded (cite paper or finding), scoped (< 2h on M5 Pro), and falsifiable (numeric kill criteria with target metrics).
