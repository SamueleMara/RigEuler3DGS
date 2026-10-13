#!/usr/bin/env python
"""A machine-produced suite-by-suite ledger. Nothing here is hand-totalled.

Section 0 of CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md.

An earlier report quoted "15 suites, 394 checks" for a list of counts that
actually sums to 437. The number was typed, not computed, and that is the
whole reason this exists: every figure below comes from running the suite and
parsing its own last line, and the log of each run is hashed so the claim can
be checked against the bytes that produced it.

A passing count is NOT an acceptance gate. The ledger records what executed;
the gates live in docs/unified_completion_report_after_f80e3e1.md.

    python tools/run_test_ledger.py --out docs/test_ledger.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The suites this pipeline owns. Listed explicitly rather than globbed, so a
# suite cannot silently leave the ledger by being renamed.
SUITES = [
    "test_unified_source_index",
    "test_unified_process_jacobians",
    "test_unified_full_window_map",
    "test_unified_cuda_renderer_jvp_vjp",
    "test_unified_nonlinear_ieks",
    "test_unified_ieks_convergence",
    "test_unified_ieks_gradient_components",
    "test_unified_ieks_observability",
    "test_unified_direction_invariance",
    "test_unified_coherence_t10",
    "test_unified_streaming_window",
    "test_unified_curvature_and_observability",
    "test_unified_gt_isolation_and_gauge",
    "test_unified_full_state_export",
    "test_unified_representation_depth_character",
    "test_unified_coherence_semantics",
    "test_bidirectional_sequence_direction",
    "test_bidirectional_time_and_source",
    "test_motion_coherence_support",
]

COUNT = re.compile(r"^(\d+) checks passed")
FAILED = re.compile(r"^(\d+) of (\d+) checks FAILED")
NOT_RUN = re.compile(r"NOT_RUN")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="docs/test_ledger.json")
    ap.add_argument("--logs", default=None,
                    help="where to keep each run's stdout; default a temp dir "
                         "beside the ledger")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    logs = Path(args.logs) if args.logs else out.parent / "test_ledger_logs"
    logs.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    suites = [s for s in SUITES if args.only is None or s in args.only]
    rows, started = [], time.time()
    for name in suites:
        path = REPO / "tests" / f"{name}.py"
        row = {"suite": name, "path": f"tests/{name}.py"}
        if not path.is_file():
            row.update({"status": "MISSING", "exit_code": None, "checks": None})
            rows.append(row)
            print(f"  MISSING   {name}")
            continue
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([args.python, str(path)], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=args.timeout)
            stdout, code = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf8", "replace") if isinstance(
                exc.stdout, bytes) else (exc.stdout or "")
            code = None
        wall = time.perf_counter() - t0

        log_path = logs / f"{name}.log"
        log_path.write_text(stdout)
        digest = hashlib.sha256(stdout.encode()).hexdigest()
        last = [l for l in stdout.strip().splitlines() if l.strip()]
        last = last[-1] if last else ""

        checks = failures = None
        if (m := COUNT.match(last)):
            checks, failures, status = int(m.group(1)), 0, "PASSED"
        elif (m := FAILED.match(last)):
            failures, checks = int(m.group(1)), int(m.group(2))
            status = "FAILED"
        elif NOT_RUN.search(stdout):
            status = "NOT_RUN"
        elif code is None:
            status = "TIMEOUT"
        else:
            status = "UNPARSED"
        if code not in (0, None) and status == "PASSED":
            status = "EXIT_CODE_MISMATCH"

        row.update({"status": status, "exit_code": code, "checks": checks,
                    "failures": failures, "wall_time_s": round(wall, 2),
                    "last_line": last, "log": str(log_path),
                    "log_sha256": digest})
        rows.append(row)
        print(f"  {status:<18} {name:<44} "
              f"{'' if checks is None else str(checks) + ' checks':<12} "
              f"exit {code}  {wall:6.1f} s")

    executed = [r for r in rows if r["status"] in ("PASSED", "FAILED")]
    payload = {
        "generated_by": "tools/run_test_ledger.py",
        "python": args.python,
        "total_wall_time_s": round(time.time() - started, 2),
        "num_suites_listed": len(suites),
        "num_suites_executed": len(executed),
        "num_suites_passed": sum(1 for r in rows if r["status"] == "PASSED"),
        # Computed here, from the parsed rows. Never typed.
        "total_checks_executed": sum(r["checks"] or 0 for r in executed),
        "total_failures": sum(r["failures"] or 0 for r in executed),
        "all_passed": all(r["status"] == "PASSED" for r in rows),
        "suites": rows,
        "note": ("Counts record what EXECUTED. They are not an acceptance "
                 "gate and must not be quoted as one; the gates and their "
                 "evidence are in the completion report."),
    }
    out.write_text(json.dumps(payload, indent=1))

    table = ["| suite | status | checks | exit | wall (s) | log sha256 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        table.append(f"| `{r['suite']}` | {r['status']} | "
                     f"{'' if r.get('checks') is None else r['checks']} | "
                     f"{r.get('exit_code')} | {r.get('wall_time_s', '')} | "
                     f"`{(r.get('log_sha256') or '')[:16]}` |")
    table.append(f"| **{payload['num_suites_executed']} suites executed** | "
                 f"{'all passed' if payload['all_passed'] else 'SEE ABOVE'} | "
                 f"**{payload['total_checks_executed']}** | | "
                 f"{payload['total_wall_time_s']} | |")
    out.with_suffix(".md").write_text("\n".join(table) + "\n")

    print(f"\n  {payload['num_suites_executed']} suites executed, "
          f"{payload['total_checks_executed']} checks, "
          f"{payload['total_failures']} failures "
          f"(computed, not typed)")
    print(f"  {out}")
    print(f"  {out.with_suffix('.md')}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
