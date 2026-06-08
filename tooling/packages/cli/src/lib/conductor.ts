/**
 * Conductor — the multi-model experiment orchestration loop (config-driven by conductor.yml).
 *
 * Orchestration is JS; the EXPERIMENT code it runs stays Python (numpy/MLX). Models are decoupled CLI
 * workers assigned to roles in conductor.yml (explore->Gemini, systematize->GPT-5.5, synthesize->Opus).
 * Per job: file -> run -> analyze(deterministic) -> complete. Every job is error-wrapped; on failure it
 * falls back / records the reason and CONTINUES, so it drains a queue indefinitely.
 *
 * Imported in-process by the `orchestrate` command (`runQueue`); also runnable standalone via the
 * import.meta.main guard at the bottom:  bun tooling/packages/cli/src/lib/conductor.ts [queue.json]
 */
import YAML from "yaml";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// src/lib -> repo root (5 up). Derived, not hardcoded, so it works from any cwd / machine.
const ROOT = resolve(import.meta.dir, "../../../../..");
const HERE = `${ROOT}/tooling/orchestrator`;
const PY = `${ROOT}/.venv/bin/python`;
const EXP = `${ROOT}/experiments/models`;
const HOME = process.env.HOME ?? "";
const ENVPATH = `${HOME}/.local/bin:${HOME}/.vite-plus/bin:${HOME}/.bun/bin:/opt/homebrew/bin:${process.env.PATH ?? ""}`;

const DEFAULTS: any = {
  models: { gemini: { cmd: "gemini", model: "gemini-pro", timeout: 200 }, gpt: { cmd: "codex", model: "gpt-5.5", timeout: 240 }, opus: { cmd: "claude", model: "opus", timeout: 300 } },
  roles: { explore: "gemini", systematize: "gpt", synthesize: "opus" },
  spark_source: "tooling/spark/pipeline_gap/3_systematize.txt", run_timeout: 300, codegen: true,
  experiment: { scale: "micro", tags: ["conductor", "novel"], kill: "result does not change under the intervention" },
  queue: [{ finding: "exp_gap_as_signal", exp_name: "exp_gap_collapse_demo", hypothesis: 1 }],
};

const log = (m: string) => console.log(`[conductor] ${m}`);

function loadConf(): any {
  const p = `${HERE}/conductor.yml`;
  if (!existsSync(p)) return DEFAULTS;
  let c: any;
  try { c = YAML.parse(readFileSync(p, "utf8")) || {}; }
  catch (e) { throw new Error(`conductor.yml is invalid YAML: ${e}`); } // surfaces to oclif, not a raw crash
  return {
    ...DEFAULTS, ...c,
    models: { ...DEFAULTS.models, ...(c.models || {}) },
    roles: { ...DEFAULTS.roles, ...(c.roles || {}) },
    experiment: { ...DEFAULTS.experiment, ...(c.experiment || {}) },
  };
}

/** run a CLI; return [stdout|null, err|null]. Some CLIs exit non-zero yet produce usable output. */
function run(cmd: string[], opts: { input?: string; timeoutMs?: number; cwd?: string } = {}): [string | null, string | null] {
  try {
    const out = execFileSync(cmd[0], cmd.slice(1), {
      input: opts.input, cwd: opts.cwd, encoding: "utf8",
      timeout: opts.timeoutMs ?? 240000, maxBuffer: 32 * 1024 * 1024,
      env: { ...process.env, PATH: ENVPATH }, stdio: ["pipe", "pipe", "pipe"],
    });
    const t = (out || "").trim();
    return t ? [t, null] : [null, "empty output"];
  } catch (e: any) {
    const t = e?.stdout ? String(e.stdout).trim() : "";
    if (t) return [t, null];
    return [null, String(e?.stderr || e?.message || e).slice(0, 400)];
  }
}

/** dispatch to the CLI configured for `role`. */
function modelCall(role: string, prompt: string, conf: any): [string | null, string | null] {
  const spec = conf.models[conf.roles[role]] || {};
  const { cmd, model } = spec; const to = (spec.timeout ?? 240) * 1000;
  if (cmd === "gemini") {
    const base = ["gemini", "-p", prompt, "-o", "text", "--skip-trust"];
    let [out, err] = run(model ? [...base, "-m", model] : base, { timeoutMs: to });
    if (out === null && model) [out, err] = run(base, { timeoutMs: to }); // fallback to default model id
    if (out === null) return [null, err];
    const clean = out.split("\n").filter((l) => !/^(Ripgrep|Falling back|Loaded cached)/.test(l)).join("\n").trim();
    return clean ? [clean, null] : [null, "empty after noise filter"];
  }
  if (cmd === "codex") {
    const f = join(tmpdir(), `cx_${process.pid}_${Date.now()}.txt`);
    run(["codex", "exec", "-m", model, "--skip-git-repo-check", "-s", "read-only", "--output-last-message", f, prompt], { timeoutMs: to });
    if (existsSync(f)) { const t = readFileSync(f, "utf8").trim(); try { unlinkSync(f); } catch {} if (t) return [t, null]; }
    return [null, "codex no output"];
  }
  if (cmd === "claude") return run(["claude", "-p", "--model", model || "opus"], { input: prompt, timeoutMs: to });
  return [null, `unknown cmd for role ${role}`];
}

const sh = (args: string[]) => run(args, { timeoutMs: 120000 });
const stripFence = (s: string) => { const m = s.match(/```(?:python)?\s*([\s\S]*?)```/); return (m ? m[1] : s).trim(); };

/** guaranteed-runnable fallback EXPERIMENT (python numpy; the experiment stays python). */
const FALLBACK_RUN = `
"""Micro mechanism check (numpy, <5s): does collapsing the interference gap g=f_A-f_B destroy A-vs-B
separability of the COMPOSED output? alpha 1->0 = routed->merged. Predict: separability -> chance.
Writes results.json in CWD. is_smoke micro."""
import json, numpy as np
np.seterr(all="ignore")
rng = np.random.RandomState(0)
d, r, n = 256, 8, 400
A = (rng.randn(d, r) @ rng.randn(r, d)) / np.sqrt(d)
B = (rng.randn(d, r) @ rng.randn(r, d)) / np.sqrt(d)
X = rng.randn(n, d); fA, fB = X @ A, X @ B
lab = np.arange(n) % 2
target = np.where(lab[:, None] == 0, fA, fB); direction = fA - fB
def sep(a):
    out = a * target + (1 - a) * 0.5 * (fA + fB)
    s = (out * direction).sum(1); p = (s < np.median(s)).astype(int)
    return float(max((p == lab).mean(), (p != lab).mean()))
full, collapsed = sep(1.0), sep(0.0); drop = full - collapsed
k1 = bool(drop >= 0.15 and collapsed <= 0.60)
json.dump({"is_smoke": True, "scale": "micro", "separability_full_gap": round(full, 3),
           "separability_collapsed": round(collapsed, 3), "separability_drop": round(drop, 3),
           "verdict": "SUPPORTED" if k1 else "KILLED", "all_pass": k1,
           "kc": {"K1_gap_collapse_reduces_separability": "pass" if k1 else "fail"},
           "note": "Micro numpy mechanism sim; not the full frozen-Gemma-4 run."},
          open("results.json", "w"), indent=2)
print("RESULT done")
`;

function extractHypothesis(file: string, idx = 1): [string, string] {
  const txt = readFileSync(file, "utf8");
  const parts = txt.split(/\n##\s+\d+\./);
  const body = parts[idx] ?? txt;
  const t = txt.match(/##\s+\d+\.\s*(.+)/);
  return [t ? t[1].trim() : "Conductor hypothesis", ("##" + body).trim().slice(0, 6000)];
}

function fileExperiment(job: any, conf: any): string {
  const name = job.exp_name, mathmd = job.math, d = `${EXP}/${name}`;
  mkdirSync(d, { recursive: true });
  const used: any = { gemini: false, gpt: false };
  const [angle, gerr] = modelCall("explore", "In ONE or TWO sentences, give the single sharpest way to FALSIFY this hypothesis — the test most likely to break it. Be concrete.\n\n" + mathmd.slice(0, 1500), conf);
  used.gemini = !!angle; log("  explore(angle): " + (angle ? "ok" : `skipped (${(gerr || "").slice(0, 40)})`));
  writeFileSync(`${d}/MATH.md`, `# ${job.title}\n\n${mathmd}\n` + (angle ? `\n## Sharpest falsification angle (${conf.roles.explore})\n${angle}\n` : ""));
  let code: string | null = null;
  if (conf.codegen) {
    const [gen] = modelCall("systematize", "Write a SELF-CONTAINED python experiment (numpy only; NO network/model load; <30s) that tests this prediction with a TARGET behavioral metric and writes results.json IN THE CWD with keys verdict('SUPPORTED'|'KILLED'), all_pass(bool), kc(dict), is_smoke(true), measured values. Output ONLY python.\n\nPREDICTION:\n" + mathmd.slice(0, 2500), conf);
    if (gen) {
      const cand = stripFence(gen);
      const tf = join(tmpdir(), `cg_${Date.now()}.py`); writeFileSync(tf, cand);
      const [ok] = run([PY, "-m", "py_compile", tf], { timeoutMs: 30000 }); try { unlinkSync(tf); } catch {}
      if (ok !== null && cand.includes("results.json")) { code = cand; used.gpt = true; }
    }
    log("  systematize(codegen): " + (used.gpt ? "USED" : "rejected -> fallback"));
  }
  writeFileSync(`${d}/run_experiment.py`, code ?? FALLBACK_RUN);
  const ex = conf.experiment;
  const add = ["experiment", "add", name, "--title", job.title.slice(0, 120), "--scale", ex.scale ?? "micro",
    "--dir", `experiments/models/${name}/`, "--kill", ex.kill ?? "result unchanged",
    "--notes", `Conductor-orchestrated from ${job.finding ?? "spark"}. ` + job.title.slice(0, 200)];
  for (const t of ex.tags ?? []) add.push("--tag", t);
  sh(add);
  job._used = used; return d;
}

function runExperiment(d: string, conf: any): [any, string | null] {
  const rp = `${d}/results.json`;
  if (existsSync(rp)) { try { unlinkSync(rp); } catch {} }
  let [, err] = run([PY, "run_experiment.py"], { timeoutMs: (conf.run_timeout ?? 300) * 1000, cwd: d });
  if (existsSync(rp)) return [JSON.parse(readFileSync(rp, "utf8")), null];
  log(`  no results.json (${(err || "").slice(0, 60)}) -> guaranteed fallback`);
  writeFileSync(`${d}/run_experiment.py`, FALLBACK_RUN);
  [, err] = run([PY, "run_experiment.py"], { timeoutMs: 120000, cwd: d });
  return existsSync(rp) ? [JSON.parse(readFileSync(rp, "utf8")), null] : [null, err || "no results"];
}

function analyze(res: any): [string, string] {
  if (!res) return ["provisional", "blocked: no results.json"];
  const v = String(res.verdict || "").toUpperCase();
  let status = "provisional";
  if (res.is_smoke) status = "provisional";
  else if (v === "KILLED" || res.all_pass === false) status = "killed";
  else if (v === "SUPPORTED") status = "supported";
  const ev = ["verdict", "all_pass", "separability_drop", "kc"].filter((k) => k in res).map((k) => `${k}=${JSON.stringify(res[k])}`).join("; ");
  return [status, ev || JSON.stringify(res).slice(0, 300)];
}

function complete(name: string, d: string, status: string, evidence: string, res: any, job: any) {
  writeFileSync(`${d}/PAPER.md`, `# ${job.title} — PAPER\n\nStatus: ${status}\n\n## Measured\n\`\`\`json\n${JSON.stringify(res, null, 2).slice(0, 1500)}\n\`\`\`\n\n## Verdict\n${evidence}\n`);
  writeFileSync(`${d}/REVIEW-adversarial.md`, `# Review\nConductor auto-review. Status ${status}. Smoke/micro mechanism check, not the full claim. Blocking: ${res ? "none" : "no results -> provisional/blocked"}.\n`);
  const dirf = `experiments/models/${name}/`;
  sh(["experiment", "claim", "conductor", "--id", name]);
  if (["supported", "killed", "proven"].includes(status)) {
    sh(["experiment", "complete", name, "--status", status, "--dir", dirf, "--evidence", evidence.slice(0, 400)]);
  } else { // provisional/blocked: `complete` rejects these -> `update` + evidence
    sh(["experiment", "update", name, "--status", "provisional", "--dir", dirf]);
    sh(["experiment", "evidence", name, "--claim", evidence.slice(0, 300), "--source", `${dirf}results.json`]);
  }
  const fstatus = status === "provisional" ? "provisional" : status === "supported" ? "supported" : "killed";
  sh(["experiment", "finding-add", "--title", job.title.slice(0, 110), "--status", fstatus, "--result", evidence.slice(0, 400), "--experiment", name, "--scale", "micro"]);
}

function processOne(job: any, conf: any): any {
  const name = job.exp_name;
  log(`=== JOB ${name} : ${job.title.slice(0, 70)} ===`);
  try {
    const d = fileExperiment(job, conf); log("  filed (MATH.md + run_experiment.py + DB)");
    const [res, err] = runExperiment(d, conf); log(`  ran -> ${res ? "results.json" : "NO RESULTS: " + String(err).slice(0, 50)}`);
    const [status, evidence] = analyze(res); log(`  analyze (deterministic) -> ${status}`);
    complete(name, d, status, evidence, res ?? { verdict: "BLOCKED", error: err }, job);
    log(`  COMPLETED status=${status}`);
    return { job: name, status, ok: true, used: job._used ?? {} };
  } catch (e: any) {
    log(`  JOB ERROR (handled, continuing): ${e}`);
    try { sh(["experiment", "update", name, "--status", "provisional"]); } catch {}
    return { job: name, status: "error", ok: false, used: job._used ?? {} };
  }
}

function enrich(queue: any[], conf: any): any[] {
  const src = `${ROOT}/${conf.spark_source}`;
  for (const job of queue) {
    if (!("math" in job)) {
      let t = "Conductor hypothesis", m = "CLAIM: the intervention changes the target behavioral metric.";
      if (existsSync(src)) [t, m] = extractHypothesis(src, job.hypothesis ?? 1);
      job.title = job.title ?? t; job.math = m;
    }
  }
  return queue;
}

export interface QueueSummary { results: any[]; okCount: number; total: number; modelsUsed: string[]; }

/** Drain a queue of experiment hunches. In-process entry point used by the `orchestrate` command.
 *  Throws only on un-recoverable setup errors (bad conductor.yml / queue file); per-job failures are
 *  handled internally and the loop continues. Does NOT call process.exit — the caller owns the exit. */
export function runQueue(queuePath?: string): QueueSummary {
  const conf = loadConf();
  let queue: any[];
  if (queuePath) {
    if (!existsSync(queuePath)) throw new Error(`queue file not found: ${queuePath}`);
    try { queue = JSON.parse(readFileSync(queuePath, "utf8")); }
    catch (e) { throw new Error(`queue file is invalid JSON (${queuePath}): ${e}`); }
  } else queue = conf.queue;
  if (!Array.isArray(queue) || queue.length === 0) throw new Error("queue is empty — nothing to orchestrate");

  queue = enrich(queue, conf);
  log(`draining queue of ${queue.length} job(s)... (roles: ${JSON.stringify(conf.roles)})`);
  const results = queue.map((j: any) => processOne(j, conf));
  log("=== QUEUE DRAINED ===");
  const modelsUsed = [...new Set(results.flatMap((r) => Object.entries(r.used || {}).filter(([, v]) => v).map(([k]) => k)))];
  for (const r of results) {
    const used = ["gemini", "gpt"].filter((m) => r.used?.[m]).join(",") || "fallback-only";
    log(`  ${r.job}: ${r.status}  (models used: ${used})`);
  }
  const okCount = results.filter((r: any) => r.ok).length;
  log(`DONE: ${okCount}/${results.length} jobs completed cleanly.`);
  return { results, okCount, total: results.length, modelsUsed };
}

// Standalone:  bun tooling/packages/cli/src/lib/conductor.ts [queue.json]
if (import.meta.main) {
  try { runQueue(process.argv[2]); process.exit(0); }
  catch (e: any) { console.error(`[conductor] FATAL: ${e?.message ?? e}`); process.exit(1); }
}
