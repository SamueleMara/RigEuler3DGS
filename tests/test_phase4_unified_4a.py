"""Unified spec 2026-11-11, section 3.6 (4A) and the 4A side of section 4.

  artifact      stage_4a writes 4a_original_membership/ with every listed file; complete.json only when the gate passes;
                every original UID has a status (no UNTESTED); N_final = N0; canonical bytes and joint unchanged;
                S / M / OMIT evidence recorded per group; re-invocation does not re-run; tampering is detected
  direction     the same withheld scene processed forward and backward (each with its own anchor and q mapping) gives
                the same decided labels
  core          a core contaminated with static back-wall Gaussians fails the integrity gate as
                UPSTREAM_JOINT_OR_SUPPORT_CONFLICT (never silently relabelled; no complete.json)
(withheld-moving R / P recovery, the hinge group, resume and static controls: tests/test_phase4a_exact_labels.py;
 ROI exactness: tests/test_phase4_exact_roi_parity.py; decode-once: tests/test_phase4_view_cache.py)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import phase4_artifacts as AR  # noqa: E402
from recovery import phase4a_exact as X  # noqa: E402
from recovery.exact_objective import ExactLossConfig  # noqa: E402
from recovery.phase4_stages import load_membership, stage_4a  # noqa: E402
from phase4_ctx import QUIET, ctx_from_scene  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

POL = X.Policy4A(sens_groups=4)
FILES = ("manifest.json", "original_gaussians_snapshot.npz", "membership.npz", "counterfactual_evidence.jsonl",
         "joint_sensitivity.json", "all_view_model_audit.json", "resource_report.json", "complete.json")


def test_4a_artifact_contract():
    s = make_scene("revolute", "forward", withhold_frac=0.4, seed=0)
    ctx = ctx_from_scene(s)
    h0 = AR.canonical_hashes(ctx.scene.g)
    with tempfile.TemporaryDirectory() as tmp:
        r = stage_4a(ctx, tmp, POL, ExactLossConfig(), log=QUIET)
        assert r["complete"], r.get("reasons")
        d = Path(tmp) / AR.STAGE_A
        for f in FILES:
            assert (d / f).is_file(), f
        c = AR.require_complete(d, AR.STAGE_A)
        assert c["N0"] == c["N_final"] == len(s.model) and c["untested"] == 0 and c["all_view_verification"]
        assert AR.canonical_hashes(ctx.scene.g) == h0 and c["canonical_sha256"] == h0["combined"]
        status, render_label, z = load_membership(d)
        assert len(status) == len(s.model) and not (status == X.UNTESTED).any()
        assert np.array_equal(z["uid"], s.model.lineage["gaussian_uid"])
        snap = np.load(d / "original_gaussians_snapshot.npz")
        assert bool(snap["equal"]) and int(snap["n_before"]) == int(snap["n_after"])
        ev = [json.loads(l) for l in (d / "counterfactual_evidence.jsonl").read_text().splitlines()]
        leaves = [e for e in ev if e["kind"] == "leaf"]
        assert leaves and all(e["E_S"] is not None and e["E_M"] is not None and e["E_OMIT"] is not None for e in leaves)
        assert all("E_S_roi" in e for e in leaves)
        withheld = np.setdiff1d(np.nonzero(s.gt_moving_model)[0], s.core)
        assert (status[withheld] == X.MOVING_S).mean() == 1.0
        assert not (status[~s.gt_moving_model] == X.MOVING_S).any()
        # immutable: a second invocation verifies and skips; tampering is detected
        assert stage_4a(ctx, tmp, POL, ExactLossConfig(), log=QUIET).get("skipped")
        with open(d / "membership.npz", "ab") as fh:
            fh.write(b"x")
        try:
            AR.require_complete(d, AR.STAGE_A)
            raise AssertionError("a modified 4A artifact must be refused")
        except SystemExit:
            pass
        cnt = c["counts"]
    print(f"  4A artifact: {len(FILES)} files, N0 = N_final = {len(s.model)}, canonical bytes identical, statuses {cnt}; "
          f"re-invocation skipped; tampering refused")


def test_4a_forward_backward_same_outcome():
    out = {}
    for direction in ("forward", "backward"):
        s = make_scene("prismatic", direction, withhold_frac=0.4, seed=11)
        ctx = ctx_from_scene(s)
        with tempfile.TemporaryDirectory() as tmp:
            r = stage_4a(ctx, tmp, X.Policy4A(sensitivity=False), ExactLossConfig(), log=QUIET)
            assert r["complete"], r.get("reasons")
            out[direction] = (load_membership(Path(tmp) / AR.STAGE_A)[0], s)
    sf, s = out["forward"]
    sb, _ = out["backward"]
    agree = (sf == sb).mean()
    assert agree == 1.0, f"forward / backward statuses agree on only {agree:.3f} of the originals"
    nc = np.setdiff1d(np.arange(len(sf)), s.core)
    dec = np.isin(sf[nc], [X.MOVING_S, X.STATIC_S]).mean()
    assert dec > 0.9, f"only {dec:.2f} of the non-core originals decided"
    print(f"  direction: forward (anchor frame {out['forward'][1].joint.frames[0]}) and backward (anchor frame "
          f"{out['backward'][1].joint.frames[0]}) give identical statuses for all {len(sf)} originals; "
          f"{dec:.2f} of the non-core decided")


def _contaminated_core():
    s = make_scene("revolute", "forward", withhold_frac=0.4, seed=12)
    back = np.nonzero(s.surface_model == "back")[0]
    rng = np.random.default_rng(0)
    bad = rng.choice(back, len(back) // 2, replace=False)
    return s, bad, np.union1d(s.core, bad)               # the "core" now contains static back-wall Gaussians


def test_4a_contaminated_support_is_revised():
    """Default core_policy 'revisable': Phase 4 corrects the Phase-3 support instead of stopping."""
    s, bad, core = _contaminated_core()
    ctx = ctx_from_scene(s, core=core)
    with tempfile.TemporaryDirectory() as tmp:
        r = stage_4a(ctx, tmp, X.Policy4A(sensitivity=False), ExactLossConfig(), log=QUIET)
        assert r["complete"], r.get("reasons")
        status, render_label, z = load_membership(Path(tmp) / AR.STAGE_A)
        rev = AR.require_complete(Path(tmp) / AR.STAGE_A, AR.STAGE_A)["core_relabelled_static"]
        assert rev and rev["gaussians"] > 0
        bad_static = (status[bad] == X.STATIC_S).mean()
        true_core_kept = (status[s.core] == X.CORE).mean()
        assert bad_static > 0.9, f"only {bad_static:.2f} of the planted static support rows relabelled STATIC"
        assert true_core_kept > 0.95, f"only {true_core_kept:.2f} of the true support kept"
        assert not render_label[status == X.STATIC_S].any()
        assert z["core_member"][bad].all()                # Phase-3 membership is still recorded
    print(f"  revisable support: {rev['groups']} groups / {rev['gaussians']} Gaussians relabelled STATIC; "
          f"{bad_static:.2f} of the planted static rows STATIC, {true_core_kept:.2f} of the true support kept CORE")


def test_4a_core_conflict_fails_gate():
    """core_policy 'gate' (opt-in): the old integrity gate."""
    s, bad, core = _contaminated_core()
    ctx = ctx_from_scene(s, core=core)
    with tempfile.TemporaryDirectory() as tmp:
        r = stage_4a(ctx, tmp, X.Policy4A(sensitivity=False, core_policy="gate"), ExactLossConfig(), log=QUIET)
        d = Path(tmp) / AR.STAGE_A
        assert not r["complete"] and not (d / "complete.json").exists() and (d / "incomplete.json").exists()
        reasons = json.loads((d / "incomplete.json").read_text())["reasons"]
        assert any("UPSTREAM_JOINT_OR_SUPPORT_CONFLICT" in x for x in reasons), reasons
        status = r["res"]["status"]
        flagged = status == X.JOINT_INVALID
        assert flagged.any() and np.isin(np.nonzero(flagged)[0], core).all()
        assert (r["res"]["render_label"][core] == 1).all()          # the core is never relabelled
    print(f"  core conflict: {int(flagged.sum())} contaminated core Gaussians flagged UPSTREAM_JOINT_OR_SUPPORT_CONFLICT "
          f"({int(np.isin(np.nonzero(flagged)[0], bad).mean() * 100)} % of them are the planted static rows); "
          f"no complete.json; core labels untouched")


if __name__ == "__main__":
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed.append(name)
    raise SystemExit(1 if failed else 0)
