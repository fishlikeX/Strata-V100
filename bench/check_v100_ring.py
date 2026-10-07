#!/usr/bin/env python3
"""Teacher-forced expert-ring check for the batched-prefill path (two V100, IQ3_S, layer split 16).

GOAL
Compare one engine binary, one config and one prompt across three serve arms: the engine's own
ring (default), the same arm again (default-control, the numerical control), and
STRATA_SPLIT_RING=<N> (ring<N>, --ring default 192).

MECHANISM
The prompt is [target realistic tokens] + [<|im_start|>] + [600 realistic tokens]. The engine
splits the read at the last <|im_start|>. The head is longer than --short-read, so it takes the
batched prefill (streamed ring and expert-cache loan). The 600-token tail is read through the
verify windows, and STRATA_LOGPOS writes one row per tail token, teacher-forced: the row target
is the prompt's own next token.

FIXED CONTROLS
--prompt-cache 1 with every reuse tier off (0 reused tokens required), --short-read 640,
--adapt-every 100000, --prefill auto, and both ring env vars cleared from both arms.

METRICS AND LIMITS
Exact, over the 600 scored positions: argmax agreement and the disagreeing positions, top-10
overlap, and the true-token perplexity per arm. No KL: a top-k approximation is not a valid KL.
A nonzero same-arm control difference is a warning (same_arm_control.warning), not a gate.
Equal numbers are not equivalence of the arms.

COMMANDS
  .venv/bin/python bench/check_v100_ring.py --target 4096 --out build-perf/ring-check/4k
  .venv/bin/python bench/check_v100_ring.py --target 32768 --out build-perf/ring-check/32k

One --out directory holds one target: plan.json, result.json, the arm dumps and engine logs. The
prompt builders need the repo venv's `regex`; the script re-executes itself under that python.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- interpreter bootstrap: tools/strata_tokenizer.py needs the repo venv's regex
if __name__ == "__main__":
    try:
        import regex  # noqa: F401
    except ModuleNotFoundError:
        _venv = REPO / ".venv" / "bin" / "python"
        if _venv.exists():
            os.execv(str(_venv), [str(_venv), str(Path(__file__).resolve()), *sys.argv[1:]])

# The engine defaults this mechanism relies on (src/program/generate.cpp, src/prefill/prefill.cpp).
TURN_TOKEN = 248045          # <|im_start|>, the turn boundary the read splits on
LOGPOS_EXTRA = 248046        # <|im_end|>, the extra column the hook records
TAIL_TOKENS = 600            # <= SHORT_READ - 1, so the tail segment goes to the windows
SHORT_READ = 640
LOGPOS_TOPK = 10             # only the top-10 ids are needed: argmax and top-10 overlap
MAX_NEW = 1                  # the scored positions are teacher-forced; one decode token is enough
REPEAT = 1                   # harness prompt repeat (part of the prompt hash)
ADAPT_EVERY = 100000         # freeze the adaptive expert tier
STARTUP_TIMEOUT_S = 2400.0
REQUEST_TIMEOUT_S = 3600.0

# The split ring. STRATA_SPLIT_RING sets it; STRATA_PREFILL_RING overrides every ring in the
# planner and would win over it, so both are cleared from both arms and only the candidate then
# sets the first. A native pack caps the ring at 512 slots, and ring 0 is the pinned-share rule.
RING_ENV = "STRATA_SPLIT_RING"
PREFILL_RING_ENV = "STRATA_PREFILL_RING"
RING_MIN = 16
RING_MAX = 512
# Never passed to the child and never recorded: the engine reads no api key in --serve mode.
SECRET_ENV = ("STRATA_API_KEY", "STRATA_API_KEY_FILE")

# The only env names recorded, by name: this script's fixed request vars and ring override, plus
# the harness's own shared knobs. A wildcard over STRATA_* is never used, because the shell can
# hold secrets such as STRATA_API_KEY.
RECORDED_ENV = ("STRATA_LOGPOS", "STRATA_LOGPOS_TOPK", "STRATA_LOGPOS_EXTRA", RING_ENV)

# The engine's own startup lines for the resolved prompt chunk and ring.
CHUNK_AUTO_RE = re.compile(r"prompt chunk auto: (\d+) tokens, a (\d+)-slot ring")
SPLIT_RING_RE = re.compile(r"layer split: (\d+)% of the experts resident, "
                           r"the prompt path's streamed ring (\d+) slots")


def load_harness():
    """Import bench/run_v100_prefill.py for its prompt builders and engine I/O."""
    path = REPO / "bench" / "run_v100_prefill.py"
    spec = importlib.util.spec_from_file_location("v100_prefill", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- prompt
def build_matched_prompt(H, tokenizer, pool, seed: int, target: int):
    """The seeded head prompt, the turn token, and a deterministic tail slice (one prompt for
    every arm; the head is too long for the window path)."""
    if target <= SHORT_READ:
        raise SystemExit(f"--target must exceed --short-read ({SHORT_READ}) so the head is batched")
    head = H.build_prompt(tokenizer, pool, seed, target, REPEAT)
    h = int.from_bytes(hashlib.sha256(f"tail:{seed}:{target}:{REPEAT}".encode()).digest()[:8], "big")
    span = len(pool) - TAIL_TOKENS - 1
    if span <= 0:
        raise SystemExit("token pool is too short for the tail slice")
    start = h % span
    return list(head) + [TURN_TOKEN] + pool[start:start + TAIL_TOKENS]


# --------------------------------------------------------------------------- argv/env
def check_argv(H, cfg: dict, exe: str) -> list[str]:
    """The harness's serve argv, but with the turn split enabled.

    bench/run_v100_prefill.py forces --prompt-cache 0 for timing arms; the STRATA_LOGPOS hook
    needs --prompt-cache > 0, so this keeps it at 1 and still disables every reuse tier and the
    periodic/root checkpoints."""
    args = list(cfg["args"])
    gpus = cfg.get("gpu") or [0]
    if len(gpus) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    args = H.drop_flag(args, "--conversation-cache-disk", "--conversation-cache-disk-gib",
                       "--conversation-cache-disk-slots", "--conversation-cache-disk-min-free-mib")
    args = H.drop_flag(args, "--prompt-cache", "--prompt-cache-every", "--prompt-cache-root",
                       "--short-read", "--conversation-cache-mib", "--adapt-every", "--pcie-frac")
    args += [
        # the turn split (and therefore the tail scoring) needs prompt-cache > 0; every reuse
        # tier and extra checkpoint stays off, so one fresh engine + one request cannot reuse
        "--prompt-cache", "1", "--prompt-cache-every", "0", "--prompt-cache-root", "0",
        "--conversation-cache-mib", "0",
        "--short-read", str(SHORT_READ),
        "--adapt-every", str(ADAPT_EVERY),
    ]
    if any(a.split("=", 1)[0] in ("--api-key", "--api-key-file") for a in args):
        raise SystemExit("refusing to run: the config argv carries an api-key flag")
    return [exe, "--serve", *args]


def clean_env(env: dict) -> list[str]:
    """Drop the ring overrides and any secret from one arm's child env.

    STRATA_PREFILL_RING overrides every ring in the planner, so it would beat the split ring and
    make the two arms equal. The engine reads no api key in --serve mode. Returns the removed
    names only: a removed value is never recorded."""
    removed = []
    for k in (RING_ENV, PREFILL_RING_ENV, *SECRET_ENV):
        if k in env:
            env.pop(k)
            removed.append(k)
    return removed


# --------------------------------------------------------------------------- run
def run_arm(H, label: str, cfg: dict, argv: list[str], env: dict, overrides: list[str],
            ids: list[int], out_dir: Path) -> dict:
    logpos = (out_dir / f"logpos-{label}.tsv").resolve()
    if logpos.exists():
        logpos.unlink()
    emb = (out_dir / "geni-empty.sve").resolve()
    emb.write_bytes(b"")

    arm_env = dict(env)
    arm_env["STRATA_LOGPOS"] = str(logpos)
    arm_env["STRATA_LOGPOS_TOPK"] = str(LOGPOS_TOPK)
    arm_env["STRATA_LOGPOS_EXTRA"] = str(LOGPOS_EXTRA)

    log_path = out_dir / f"engine-{label}.log"
    proto_path = out_dir / f"protocol-{label}.jsonl"
    proto_f = proto_path.open("w", encoding="utf-8")

    def line_cb(line: str):
        proto_f.write(json.dumps({"t": round(time.time(), 3), "line": line}) + "\n")
        proto_f.flush()

    row = {"label": label, "argv": argv, "env_overrides": list(overrides),
           "env_recorded": {k: arm_env[k] for k in (*RECORDED_ENV, *H.SERVICE_ENV, *H.REQUIRED_ENV)
                            if k in arm_env},
           "engine_log": str(log_path)}
    logf = log_path.open("w", encoding="utf-8")
    proc = None
    io = None
    try:
        proc = subprocess.Popen(argv, cwd=cfg.get("cwd") or str(REPO), env=arm_env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logf,
                                text=True, encoding="utf-8", bufsize=1)
        io = H.EngineIO(proc)
        row["engine_info"] = H.wait_ready(io, STARTUP_TIMEOUT_S, line_cb)
        request = f"GENI {MAX_NEW} temperature=0 {emb} {','.join(map(str, ids))}"
        ev = H.run_request(io, request, REQUEST_TIMEOUT_S, line_cb)
        row["resume"] = ev["resume"]
        row["reused"] = ev["reused"]
        row["pp"] = ev["pp"]
        row["generated_ids"] = ev["tokens"]
        row["done"] = ev["done"]
        row["err"] = ev["err"]
    finally:
        if io is not None and proc is not None and proc.poll() is None:
            H.quit_engine(proc, io, logf)
        else:
            logf.close()
        proto_f.close()
    row["logpos"] = str(logpos)
    row["logpos_bytes"] = logpos.stat().st_size if logpos.exists() else 0
    return row


# --------------------------------------------------------------------------- compare
def load_logpos(path: Path):
    """{pos: (target, lp_target, argmax_id, [top ids by descending logprob])}."""
    rows = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            topk = []
            for p in parts[8:]:
                i, sep, lp = p.partition(":")
                if sep:
                    topk.append((float(lp), int(i)))
            topk.sort(key=lambda t: (-t[0], t[1]))
            rows[int(parts[0])] = (int(parts[1]), float(parts[2]), int(parts[3]),
                                   [i for _, i in topk])
    return rows


def compare(a_path: Path, b_path: Path, label_a: str, label_b: str) -> dict:
    A = load_logpos(a_path)
    B = load_logpos(b_path)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit(f"no common positions between {a_path} and {b_path} "
                         f"(A={len(A)} rows, B={len(B)} rows)")
    same, overlap, target_mismatch = 0, 0.0, 0
    disagree: list[int] = []
    lp_a: list[float] = []
    lp_b: list[float] = []
    lp_identical = True
    for pos in common:
        ta, lpa, topa, ka = A[pos]
        tb, lpb, topb, kb = B[pos]
        if topa == topb:
            same += 1
        else:
            disagree.append(pos)
        overlap += len(set(ka[:LOGPOS_TOPK]) & set(kb[:LOGPOS_TOPK])) / float(LOGPOS_TOPK)
        if ta == tb:
            lp_a.append(lpa)
            lp_b.append(lpb)
        else:
            target_mismatch += 1
        if lpa != lpb:
            lp_identical = False
    n = len(common)
    out = {
        "label_a": label_a, "label_b": label_b,
        "positions": n,
        "positions_only_in_a": len(set(A) - set(B)),
        "positions_only_in_b": len(set(B) - set(A)),
        "target_mismatches": target_mismatch,
        "argmax_agreements": same,
        "argmax_agreement_pct": round(100.0 * same / n, 4),
        "argmax_disagreement_positions": disagree,
        "top10_overlap_pct": round(100.0 * overlap / n, 4),
        "mean_lp_true_a": round(sum(lp_a) / len(lp_a), 6) if lp_a else None,
        "mean_lp_true_b": round(sum(lp_b) / len(lp_b), 6) if lp_b else None,
        "ppl_a": round(math.exp(-sum(lp_a) / len(lp_a)), 6) if lp_a else None,
        "ppl_b": round(math.exp(-sum(lp_b) / len(lp_b)), 6) if lp_b else None,
        "lp_identical": lp_identical,
    }
    # exact_match is exact on the reported fields: positions, targets, argmax, the top-k id sets
    # and the true-token log-probabilities. It is not a claim about the untested vocabulary tail.
    out["exact_match"] = (out["positions_only_in_a"] == 0 and out["positions_only_in_b"] == 0
                          and target_mismatch == 0 and same == n and overlap == n and lp_identical)
    return out


# --------------------------------------------------------------------------- gates
def _int_at(items: list, i: int):
    try:
        return int(items[i])
    except (IndexError, TypeError, ValueError):
        return None


def parse_engine_log(path: Path) -> dict:
    """The engine's own resolved prompt chunk and ring, from its startup lines."""
    out = {"chunk": None, "ring": None, "split_ring": None, "resident_pct": None}
    if path is not None and path.exists():
        with path.open(errors="replace") as f:
            for line in f:
                m = CHUNK_AUTO_RE.search(line)
                if m:
                    out["chunk"], out["ring"] = int(m.group(1)), int(m.group(2))
                m = SPLIT_RING_RE.search(line)
                if m:
                    out["resident_pct"], out["split_ring"] = int(m.group(1)), int(m.group(2))
    return out


def validate_arm(row: dict, ids: list[int], target: int) -> dict:
    """Per-row hard gates. A row that fails any gate is a failure, never a comparison."""
    n = len(ids)
    done = row.get("done") or []
    pp = row.get("pp") or []
    reached = [int(p[0]) for p in pp if len(p) >= 2 and p[0].lstrip("-").isdigit()]
    totals = {int(p[1]) for p in pp if len(p) >= 2 and p[1].isdigit()}
    row["resolved"] = parse_engine_log(Path(row["engine_log"]) if row.get("engine_log") else None)
    gates = {
        "no_engine_error": row.get("err") in (None, "") and _int_at(done, 1) == n,
        "reused_zero": row.get("reused") == 0 and _int_at(done, 7) == 0,
        "prompt_read_full": _int_at(done, 13) == n,
        "resolved_ring_logged": row["resolved"]["ring"] is not None,
        # --target > --short-read means windows_ok(0, target) is false, so the engine must read
        # [0, target) through the batched path; its progress line lands exactly on the head end
        # (the window-read tail that follows reports n-1).
        "batched_head_pp": target > SHORT_READ and target in reached,
        "pp_full_prompt": bool(reached) and reached == sorted(reached)
                          and reached[-1] == n - 1 and totals == {n},
    }
    logpos = Path(row["logpos"]) if row.get("logpos") else None
    rows = load_logpos(logpos) if logpos is not None and logpos.exists() else {}
    expected = list(range(target, n - 1))
    gates["scored_positions_600"] = sorted(rows) == expected
    gates["matched_targets"] = bool(rows) and all(p + 1 < n and rows[p][0] == ids[p + 1]
                                                  for p in rows)
    gates["topk_complete"] = bool(rows) and all(len(rows[p][3]) == LOGPOS_TOPK for p in rows)
    row["gates"] = gates
    row["logpos_rows"] = len(rows)
    row["pp_reached"] = reached
    row["scored_positions"] = {"count": len(rows), "first": min(rows) if rows else None,
                               "last": max(rows) if rows else None, "expected": [target, n - 2]}
    row["matched_targets_count"] = sum(1 for p in rows
                                       if p + 1 < n and rows[p][0] == ids[p + 1])
    return row


def finalize_row(row: dict) -> dict:
    row["failures"] = sorted(k for k, v in row["gates"].items() if not v)
    row["status"] = "ok" if not row["failures"] else "failed"
    return row


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "strata-iq3_s-arena.json")
    ap.add_argument("--exe", default=None, help="engine binary (default: the config's exe)")
    ap.add_argument("--target", type=int, default=4096,
                    help=f"batched head length (must exceed --short-read {SHORT_READ})")
    ap.add_argument("--seed", type=int, default=617, help="prompt seed (617 = the runtime benchmark's)")
    ap.add_argument("--ring", type=int, default=192,
                    help=f"{RING_ENV} slots for the ring arm ({RING_MIN}..{RING_MAX})")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "results" / "ring-check",
                    help="evidence directory for this target (plan.json, result.json, dumps)")
    args = ap.parse_args()
    if args.target <= SHORT_READ:
        raise SystemExit(f"--target must exceed --short-read ({SHORT_READ}) so the head is batched")
    if not RING_MIN <= args.ring <= RING_MAX:
        raise SystemExit(f"--ring must be {RING_MIN}..{RING_MAX} (a native pack caps the ring at "
                         f"{RING_MAX} slots, and 0 means the pinned-share rule)")

    H = load_harness()
    cfg = H.load_config(args.config)
    exe = str(Path(args.exe or cfg["exe"]).resolve())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    argv = check_argv(H, cfg, exe)
    gpus = [int(g) for g in (cfg.get("gpu") or [0])]
    if len(gpus) < 2:
        raise SystemExit("this check needs a layer split: the config lists one GPU, so no split "
                         f"ring ({RING_ENV}) is used")
    if "--layer-split" not in argv:
        raise SystemExit("this check needs an explicit --layer-split in the config argv")
    prefill = H.flag_value(argv, "--prefill")
    if prefill is None or not (prefill == "auto" or prefill.startswith("auto:")):
        raise SystemExit("this check needs --prefill auto: the engine logs the resolved ring only "
                         "for an auto chunk")
    tokenizer_dir = Path(cfg.get("tokenizer")
                         or (Path(cfg["args"][cfg["args"].index("--pack") + 1]) / "tokenizer"))
    tokenizer = H.load_tokenizer(tokenizer_dir)
    need = args.target * 2 + TAIL_TOKENS + 8192
    pool = H.build_pool(tokenizer, args.seed, need, None)
    ids = build_matched_prompt(H, tokenizer, pool, args.seed, args.target)
    n = len(ids)

    label_a, control_label, label_b = "default", "default-control", f"ring{args.ring}"
    plan = {
        "config": H.SCRUB(str(args.config.resolve())),
        "exe": H.SCRUB(exe),
        "exe_sha256": H.file_sha256(Path(exe)),
        "argv": H.sanitize(argv),
        "prompt": {"seed": args.seed, "target": args.target, "repeat": REPEAT,
                   "total_tokens": n, "head_batched_tokens": args.target,
                   "turn_token": TURN_TOKEN, "tail_window_tokens": n - 1 - args.target,
                   "tail_ok": 0 < n - args.target <= SHORT_READ,
                   "sha256": H.ids_sha256(ids), "first_ids": ids[:12], "last_ids": ids[-4:]},
        "arms": {"a": label_a, "a_control": control_label, "b": label_b,
                 "requested_ring": args.ring, "ring_env": RING_ENV},
        "mechanism": [
            f"head [0,{args.target}) runs the batched prefill (ring + expert-cache loan/refill); "
            f"the {n - 1 - args.target}-token tail is scored by STRATA_LOGPOS",
            "the turn split needs --prompt-cache > 0 (set to 1); every reuse tier is off",
            f"--adapt-every {ADAPT_EVERY} freezes the adaptive expert tier",
            f"the tail is scored only if (tail_window_tokens + 1) <= --short-read ({SHORT_READ})",
            f"STRATA_LOGPOS_TOPK={LOGPOS_TOPK}: the exact top-{LOGPOS_TOPK} ids per position",
            f"{RING_ENV} and {PREFILL_RING_ENV} are removed from both arms' env "
            f"({PREFILL_RING_ENV} would override the split ring); the ring arm then sets "
            f"{RING_ENV}={args.ring}",
        ],
    }
    print(json.dumps(H.sanitize(plan), indent=2))
    if not plan["prompt"]["tail_ok"]:
        raise SystemExit(f"the tail segment ({n - args.target} tokens) exceeds --short-read {SHORT_READ}")
    (out / "plan.json").write_text(json.dumps(H.sanitize(plan), indent=2) + "\n")

    env_a = H.engine_env(cfg, gpus, [], drop_diagnostics=True)
    removed_a = clean_env(env_a)   # arm A is the engine default, whatever the shell exported
    env_b = H.engine_env(cfg, gpus, [], drop_diagnostics=True)
    removed_b = clean_env(env_b)
    env_b[RING_ENV] = str(args.ring)   # set after the clean, so it is the only ring in the env

    results = {"plan": plan, "arms": []}
    arms = [(label_a, env_a, removed_a, []), (control_label, env_a, removed_a, []),
            (label_b, env_b, removed_b, [f"{RING_ENV}={args.ring}"])]
    for label, env, removed, overrides in arms:
        print(f"[ring-check] running {label} ...", file=sys.stderr)
        try:
            row = run_arm(H, label, cfg, argv, env, overrides, ids, out)
        except (TimeoutError, EOFError, RuntimeError, OSError) as e:
            row = {"label": label, "error": f"{type(e).__name__}: {e}",
                   "engine_log": str((out / f"engine-{label}.log").resolve()),
                   "logpos": str((out / f"logpos-{label}.tsv").resolve())}
        row["env_removed"] = removed
        validate_arm(row, ids, args.target)
        finalize_row(row)
        results["arms"].append(row)
        print(f"[ring-check] {label}: status={row['status']} rows={row.get('logpos_rows')} "
              f"reused={row.get('reused')} ring={row['resolved']['ring']} "
              f"failures={row['failures']}", file=sys.stderr)

    by_label = {r["label"]: r for r in results["arms"]}
    rings = {lbl: by_label[lbl]["resolved"]["ring"] for lbl in (label_a, control_label, label_b)}

    # Cross-arm gates: the candidate must resolve to the requested ring, the two default runs to
    # the same ring, and the two arms to different rings (an ignored override proves nothing).
    def add_gate(label: str, name: str, ok: bool) -> None:
        by_label[label]["gates"][name] = ok

    add_gate(label_b, "candidate_ring_requested", rings[label_b] == args.ring)
    add_gate(label_b, "candidate_split_ring",
             by_label[label_b]["resolved"]["split_ring"] == args.ring)
    for lbl in (label_a, control_label):
        add_gate(lbl, "control_same_ring",
                 rings[label_a] is not None and rings[label_a] == rings[control_label])
    for lbl in (label_a, control_label, label_b):
        add_gate(lbl, "rings_differ",
                 rings[label_a] is not None and rings[label_b] is not None
                 and rings[label_a] != rings[label_b])
    for r in results["arms"]:
        finalize_row(r)

    results["ring_resolution"] = {
        "requested": args.ring,
        "arms": {lbl: by_label[lbl]["resolved"] for lbl in (label_a, control_label, label_b)},
        "rings_differ": rings[label_a] != rings[label_b],
    }

    def pair(label_x: str, label_y: str) -> dict:
        a, b = by_label[label_x], by_label[label_y]
        if a["status"] != "ok" or b["status"] != "ok":
            return {"skipped": "arm gates failed",
                    "arm_a": {"label": a["label"], "failures": a["failures"]},
                    "arm_b": {"label": b["label"], "failures": b["failures"]}}
        return compare(Path(a["logpos"]), Path(b["logpos"]), a["label"], b["label"])

    comparisons = {
        f"{label_a}_vs_{control_label}": pair(label_a, control_label),
        f"{label_a}_vs_{label_b}": pair(label_a, label_b),
    }
    ctrl = comparisons[f"{label_a}_vs_{control_label}"]
    if "skipped" in ctrl:
        control_warning = f"the same-arm control is not a valid comparison ({ctrl['skipped']})"
    elif ctrl["exact_match"]:
        control_warning = None
    else:
        control_warning = (
            "numerical control warning: the same-arm pair (default run 1 vs run 2) is NOT "
            f"bit-exact on the scored tail -- argmax {ctrl['argmax_agreement_pct']}%, "
            f"top-10 {ctrl['top10_overlap_pct']}%, PPL {ctrl['ppl_a']} vs {ctrl['ppl_b']}, "
            f"{ctrl['argmax_agreements']}/{ctrl['positions']} agreements. Run-to-run "
            "nondeterminism of this size is present in this run, so the arm-to-arm differences "
            "cannot be attributed to the ring. This is a warning, not a pass/fail gate.")
    results["comparisons"] = comparisons
    results["same_arm_control"] = {"exact_match": ctrl.get("exact_match"),
                                   "warning": control_warning}
    results["caveats"] = [
        "Exact numbers only, over the 600 teacher-forced tail positions of one prompt. Equal "
        "agreement or PPL is not equivalence of the two arms: the head's batched prefill is "
        "measured only through the state it leaves, and one prompt is not a distribution.",
        "No KL is reported: a top-k approximation is not a valid KL.",
        "Timings are not measured here; use bench/run_v100_prefill.py.",
    ]
    failed = any(r["status"] != "ok" for r in results["arms"])
    results["status"] = "failed" if failed else "ok"
    (out / "result.json").write_text(json.dumps(H.sanitize(results), indent=2) + "\n")

    print(json.dumps(H.sanitize({"status": results["status"], "comparisons": comparisons,
                                 "same_arm_control": results["same_arm_control"],
                                 "ring_resolution": results["ring_resolution"]}), indent=2))
    if control_warning:
        print(f"[ring-check] {control_warning}", file=sys.stderr)
    print(f"[ring-check] wrote {H.SCRUB(str(out))}/result.json", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
