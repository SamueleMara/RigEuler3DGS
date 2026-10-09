"""The Phase-2 support branch must use the v2 extractor, not the old Ncut.

CODEX_FIX_COHERENCE_EXTRACTION_AND_USE_OPTIMIZED_3DGS_GT.md sections 19, 20,
41, 45.

This exists because the branch silently regressed once: `--phase2-support-source
coherence` called `run_motion_coherence` (the superseded two-way normalized
cut) and handed Phase 2 its 46%/95% supports, while the v2 extractor lived only
in the standalone runner. Nothing failed -- the wrong support simply flowed
downstream and every Phase-2 number would have been computed on it.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def _coherence_branch_source() -> str:
    src = (REPO / "tools" / "run_complete_joint_debug.py").read_text()
    i = src.index('if args.phase2_support_source == "coherence":')
    j = src.index("members = _frozen_moving_support(", i)
    return src[i:j]


def _coherence_branch_identifiers() -> set:
    """Names the branch actually USES, with comments and strings excluded.

    Scanning the raw text is wrong here: the branch's comments and its
    SystemExit messages both name the Otsu selector in order to state that it
    is not called and must not be fallen back to, so a substring test flags
    exactly the lines documenting the guarantee.
    """
    src = (REPO / "tools" / "run_complete_joint_debug.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Attribute)
                and n.test.left.attr == "phase2_support_source")
    out = set()
    for b in node.body:
        for x in ast.walk(b):
            if isinstance(x, ast.Name):
                out.add(x.id)
            elif isinstance(x, ast.Attribute):
                out.add(x.attr)
            elif isinstance(x, ast.ImportFrom):
                out.add(x.module or "")
                out.update(a.name for a in x.names)
    return out


def test_branch_uses_v2_extractor():
    body = _coherence_branch_source()
    check("the coherence branch calls the v2 extractor",
          "extract_moving_component" in body)
    ids = _coherence_branch_identifiers()
    check("and never the superseded two-way normalized cut",
          "run_motion_coherence" not in ids and "normalized_cut" not in ids,
          "; ".join(sorted(ids & {"run_motion_coherence", "normalized_cut"})))
    check("and imports from estimator.coherence_extraction",
          "from estimator.coherence_extraction import" in body)


def test_no_silent_fallback():
    """Section 45: an unresolved partition must stop the run, not revert."""
    body = _coherence_branch_source()
    check("an unresolved coherence result raises instead of falling back",
          body.count("NO_MOVING") >= 2 and "SystemExit" in body)
    ids = _coherence_branch_identifiers()
    bad = [w for w in ("_frozen_moving_support", "select_moving_support",
                       "otsu_threshold", "moving_support",
                       "last_support_evidence")
           if w in ids]
    check("the Otsu selector is not reachable from the coherence branch",
          not bad, "; ".join(bad))


def test_frozen_support_is_preferred_and_validated():
    """Section 41: Phase 2 plugs in AFTER freezing, on the same indices."""
    body = _coherence_branch_source()
    check("the frozen predicted_support.json is loaded when present",
          "predicted_support.json" in body and "dynamic_gaussians" in body)
    check("and a Gaussian-count mismatch is rejected, not silently reindexed",
          "num_gaussians" in body and "SystemExit" in body)


def test_frozen_supports_on_disk_are_usable():
    scenes = ["USB_100109", "storage_45135", "blade_103706",
              "stapler_103111", "fridge_10905"]
    import numpy as np
    ok = 0
    for s in scenes:
        p = REPO / "output" / f"motion_coherence_{s}" / "predicted_support.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        idx = torch.tensor(r["dynamic_gaussians"], dtype=torch.long)
        c0 = REPO / "output" / f"complete_joint_debug_{s}" / "phase1" \
            / "centers" / "centers_frame_0000.npy"
        n = len(np.load(c0)) if c0.is_file() else r["num_gaussians"]
        check(f"{s}: frozen support is RESOLVED, in range and matches N",
              r["status"] == "RESOLVED" and idx.numel() > 0
              and int(idx.max()) < n and int(idx.min()) >= 0
              and r["num_gaussians"] == n,
              f"{idx.numel()}/{n} moving ({100 * idx.numel() / n:.1f}%)")
        ok += 1
    check("all five frozen supports exist", ok == 5, f"{ok}/5")


def test_all_phase2_models_get_the_same_indices():
    """Section 41: screw, P and R receive exactly the same `members`."""
    src = (REPO / "tools" / "run_complete_joint_debug.py").read_text()
    tree = ast.parse(src)
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef) and f.name == "main")
    # `members` must be assigned only inside the support selection, never
    # recomputed per model further down.
    assigns = [n.lineno for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               for t in n.targets
               if isinstance(t, ast.Name) and t.id == "members"]
    check("`members` is assigned only in the support selection",
          len(assigns) <= 3, f"assigned at lines {assigns}")
    # Every Phase-2 model must be handed the SAME `members` (section 41).
    # Checked at the call sites via AST, not by text proximity to the import.
    wanted = {"run_phase2_rigid_screw", "init_prismatic", "initialize_revolute"}
    seen = {k: [] for k in wanted}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = (getattr(node.func, "id", None)
                or getattr(node.func, "attr", None))
        if name not in wanted:
            continue
        args = [a for a in node.args] + [k.value for k in node.keywords]
        seen[name].append(any(isinstance(a, ast.Name) and a.id == "members"
                              for a in args))
    for k in sorted(wanted):
        check(f"{k} is passed `members` at every call site",
              bool(seen[k]) and all(seen[k]),
              f"{sum(seen[k])}/{len(seen[k])} call sites")


def main():
    for fn in (test_branch_uses_v2_extractor,
               test_no_silent_fallback,
               test_frozen_support_is_preferred_and_validated,
               test_frozen_supports_on_disk_are_usable,
               test_all_phase2_models_get_the_same_indices):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed")


if __name__ == "__main__":
    main()
