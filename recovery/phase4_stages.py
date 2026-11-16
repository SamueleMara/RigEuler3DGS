"""Phase 4 stage control (unified spec 2026-11-11, sections 3.5, 4, 5.7, 6.6, 8). Never calls Phase 5.

    stage_4a(ctx, out)               -> <out>/4a_original_membership/      (complete.json only when the 4A gate passes)
    stage_4b(ctx, out, a_dir)        -> <out>/4b_new_surface_proposals/    (refuses without a valid 4A complete.json)
    stage_4c(ctx, out, a_dir, b_dir) -> <out>/4c_accepted_moving/          (refuses without valid 4A and 4B)

`ctx` (Context) holds the scene (canonical originals, fixed joint, views), the decoded views and the Phase-2 core;
the CLI builds it from the upstream directories, the procedural tests build it in memory. Every stage re-verifies
the upstream artifacts' hashes before reading them and again after writing its own (a stage cannot mutate an
earlier artifact without being detected), and proves that the canonical original Gaussians and the joint are
byte-identical. `run_all` = 4A -> 4B -> 4C with those checks between stages; any failure stops it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import phase4_artifacts as AR
from .runlog import atomic_json


@dataclass
class Context:
    scene: object
    pv: object
    core: np.ndarray
    setup: dict                         # json-able record of how the inputs were built (CLI args or "procedural")
    input_proof: dict
    sensor_model: str | None = None
    not_for_results: bool = False
    code: dict = field(default_factory=dict)
    recovery_stride: int = 1            # 4B / 4C only: every k-th Phase-3 frame (4A always uses its full view set)

    @property
    def joint(self):
        return self.scene.joint


def _resources(log):
    r = {"cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None}
    try:
        r.update(log.resources())
    except Exception:
        pass
    return r


def _check_ctx(ctx: Context, prior: dict, name: str, a_dir=None):
    ch = AR.canonical_hashes(ctx.scene.g)
    if ch["combined"] != prior["canonical_sha256"]:
        raise SystemExit(f"{name}: the canonical original Gaussians differ from those of the 4A artifact")
    if AR.joint_fingerprint(ctx.joint) != prior["joint_fingerprint"]:
        raise SystemExit(f"{name}: the joint differs from the one recorded by 4A")
    mine = json.loads(json.dumps(ctx.input_proof, default=str))
    if ctx.recovery_stride <= 1:
        if mine != prior["input_proof"]:
            raise SystemExit(f"{name}: inputs (upstream hashes / decoded views) differ from the 4A artifact's")
        return
    # recovery frame stride: same upstream files and joint, and every decoded view byte-identical to 4A's copy
    for k in ("upstream_sha256", "joint"):
        if mine[k] != prior["input_proof"][k]:
            raise SystemExit(f"{name}: {k} differs from the 4A artifact's")
    vh = Path(a_dir) / "view_hashes.json" if a_dir else None
    if vh is None or not vh.is_file():
        raise SystemExit(f"{name}: --recovery-frame-stride needs a 4A artifact with view_hashes.json (re-run 4A)")
    ref = {(v["camera"], int(v["frame"])): v["sha256"] for v in json.loads(vh.read_text())["per_view"]}
    bad = [v for v in ctx.pv.manifest()["per_view"] if ref.get((v["camera"], int(v["frame"]))) != v["sha256"]]
    if bad:
        raise SystemExit(f"{name}: {len(bad)} recovery views are not byte-identical views of the 4A run, e.g. {bad[0]}")


def recovery_views(ctx: Context) -> dict:
    return {"frame_stride": ctx.recovery_stride, "full_sequence": ctx.recovery_stride <= 1, "views": len(ctx.scene.views),
            "frames": sorted({int(vr.frame) for vr in ctx.scene.views}),
            "cameras": sorted({vr.cam.name for vr in ctx.scene.views})}


def load_membership(a_dir):
    from .phase4a_exact import STATUS_CODES
    z = np.load(Path(a_dir) / "membership.npz")
    inv = {v: k for k, v in STATUS_CODES.items()}
    status = np.array([inv[int(c)] for c in z["status_code"]], dtype=object)
    return status, z["render_label"].astype(bool), z


# ====================================================================================== 4A
def stage_4a(ctx: Context, out, pol, loss_cfg, resume=False, log=print) -> dict:
    from . import phase4a_exact as X
    from .phase4c_validate import full_audit
    d = Path(out) / AR.STAGE_A
    if (d / "complete.json").is_file():
        log(f"4A: {d} is complete (immutable); not re-run")
        return {"complete": True, "skipped": True, "dir": d, "payload": AR.require_complete(d, AR.STAGE_A)}
    d.mkdir(parents=True, exist_ok=True)
    sc = ctx.scene
    N0 = len(sc.g)
    canon0 = AR.canonical_hashes(sc.g)
    jf0 = AR.joint_fingerprint(ctx.joint)
    t0 = time.time()
    ph = X.Phase4A(sc, ctx.pv, ctx.core, pol, loss_cfg, log=log, checkpoint_dir=d / "checkpoint")
    res = ph.run(resume=resume)
    canon1 = AR.canonical_hashes(sc.g)
    jf1 = AR.joint_fingerprint(ctx.joint)
    labels = torch.as_tensor(res["render_label"].astype(bool), device=sc.device)
    up_labels = torch.as_tensor(res["upstream_provisional"].astype(bool), device=sc.device)
    audit_before = full_audit(sc, up_labels, ctx.pv, loss_cfg)
    audit_after = full_audit(sc, labels, ctx.pv, loss_cfg)
    uid = sc.g.lineage["gaussian_uid"]
    status = res["status"]
    core_member = np.zeros(N0, bool)
    core_member[ctx.core] = True
    atomic_json(d / "manifest.json", {
        "stage": "4a", "setup": ctx.setup, "input_proof": ctx.input_proof, "joint_fingerprint": jf0,
        "direction": getattr(ctx.joint, "direction", None), "joint": ctx.joint.to_dict(),
        "views": [{"camera": vr.cam.name, "frame": int(vr.frame)} for vr in sc.views],
        "policy": pol.to_dict(), "loss": loss_cfg.to_dict(), "code": ctx.code, "NOT_FOR_RESULTS": ctx.not_for_results,
        "statuses": list(X.STATUS_CODES), "render_label_note": "UNRESOLVED / SUSPECT / UNTESTED rows are rendered with a "
        "provisional pose; that is not a decision", "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    atomic_json(d / "view_hashes.json", {"per_view": ctx.pv.manifest()["per_view"]})
    AR.save_npz(d / "original_gaussians_snapshot.npz", uid=uid, row=np.arange(N0), n_before=N0, n_after=len(sc.g),
                hashes_before=json.dumps(canon0), hashes_after=json.dumps(canon1),
                equal=canon0["combined"] == canon1["combined"],
                note="sha256 of the raw stored canonical tensors (xyz, rotation, scaling, opacity, dc, rest) in row order")
    AR.save_npz(d / "membership.npz", uid=uid, row=np.arange(N0),
                status_code=np.array([X.STATUS_CODES[s] for s in status], np.int8), legend=json.dumps(X.STATUS_CODES),
                render_label=res["render_label"], upstream_provisional=res["upstream_provisional"],
                group_of=res["group_of"], core_member=core_member)
    E_cur = float(np.asarray(res["E_after"]).sum()) if res["E_after"] is not None else None
    rows = []
    for i, g in enumerate(ph.groups):
        r = {"gid": g.gid, "kind": g.kind, "origin": g.origin, "parent": g.parent, "depth": g.depth,
             "size": int(len(g.members)),
             "status": g.status, "reason": g.reason, "member_uids": uid[g.members].tolist(),
             "evidence_unit": "group (labels from a jointly tested group are not single-Gaussian likelihoods)"}
        if g.kind in ("leaf", "core_audit") and E_cur is not None and i < len(ph.delta):
            dsum = float(ph.delta[i].sum())
            lab = bool(ph.labels[int(g.members[0])])
            E_S, E_M = (E_cur + dsum, E_cur) if lab else (E_cur, E_cur - dsum)
            om = ph.omit[i]
            br = ph.base_roi.get(i)
            r.update(E_S=E_S, E_M=E_M, E_OMIT=E_cur + float(np.nansum(om)) if np.isfinite(om).any() else None,
                     render_label=int(lab), Delta_channels=dict(zip(("rgb_l1", "rgb_ssim", "mask", "depth"),
                                                                    ph.delta[i].sum(0).tolist())),
                     informative_views=int((ph.changed[i] > 0).sum()), changed_pixels=int(ph.changed[i].sum()),
                     stats=g.stats)
            if br is not None:
                e_roi = float(br.sum())
                r.update(E_current_roi=e_roi, E_S_roi=(e_roi + dsum) if lab else e_roi,
                         E_M_roi=e_roi if lab else (e_roi - dsum),
                         E_OMIT_roi=(e_roi + float(np.nansum(om))) if np.isfinite(om).any() else None)
        rows.append(r)
    AR.write_jsonl(d / "counterfactual_evidence.jsonl", rows)
    atomic_json(d / "joint_sensitivity.json", res["sensitivity"] or {"joint_confidence": "JOINT_CONFIDENCE_UNCALIBRATED",
                                                                     "note": "sensitivity disabled by the policy"})
    atomic_json(d / "all_view_model_audit.json", {
        "before_4A_upstream_provisional_labels": audit_before, "after_4A_render_labels": audit_after,
        "E_before": float(sum(r["E"] for r in audit_before)), "E_after": float(sum(r["E"] for r in audit_after)),
        "contradiction_tolerance_m": 0.01, "tolerance_note": "ASSUMED 1 cm for the audit counts only",
        "core_check": res["core_check"], "core_audit": res["core_audit"], "history": res["history"]})
    atomic_json(d / "resource_report.json", {"seconds": time.time() - t0, "counters": res["counters"],
                                             "views_decode": ctx.pv.counters.to_dict(), **_resources(log)})
    counts = {s: int((status == s).sum()) for s in X.STATUS_CODES}
    reasons = []
    if res["aborted"]:
        reasons.append(f"core check failed ({res['core_check'].get('decision')}): UPSTREAM_JOINT_OR_SUPPORT_CONFLICT")
    if not res["complete"]:
        reasons.append(f"resource stop: {res['stop_reason']}")
    if counts[X.UNTESTED]:
        reasons.append(f"{counts[X.UNTESTED]} original Gaussians UNTESTED_RESOURCE_LIMIT")
    if res.get("core_conflict"):
        reasons.append("core audit: a core group decisively prefers STATIC (UPSTREAM_JOINT_OR_SUPPORT_CONFLICT)")
    if canon0["combined"] != canon1["combined"] or len(sc.g) != N0:
        reasons.append("canonical original Gaussians changed during 4A")
    if jf0 != jf1:
        reasons.append("joint changed during 4A")
    if len(audit_after) != len(sc.views):
        reasons.append("all-view audit incomplete")
    if reasons:
        AR.write_incomplete(d, reasons, {"counts": counts})
        log(f"4A INCOMPLETE: {reasons}")
        return {"complete": False, "dir": d, "reasons": reasons, "res": res, "phase": ph}
    payload = AR.write_complete(d, {
        "stage": "4a", "N0": N0, "N_final": len(sc.g), "canonical_sha256": canon0["combined"],
        "joint_fingerprint": jf0, "input_proof": json.loads(json.dumps(ctx.input_proof, default=str)),
        "counts": counts, "untested": 0, "core_conflict": False, "all_view_verification": True,
        "core_policy": pol.core_policy, "core_relabelled_static": res.get("core_revised"),
        "views": len(sc.views), "NOT_FOR_RESULTS": ctx.not_for_results, "setup": ctx.setup})
    log(f"4A COMPLETE: {counts}")
    return {"complete": True, "dir": d, "payload": payload, "res": res, "phase": ph}


# ====================================================================================== 4B
def stage_4b(ctx: Context, out, a_dir, cfg4b, log=print) -> dict:
    from . import phase4b_surface as B
    from .data_roles import Roles
    from .phase4a_exact import CORE, MOVING_S, UNDECIDED
    ca = AR.require_complete(a_dir, AR.STAGE_A)
    _check_ctx(ctx, ca, "4B", a_dir)
    d = Path(out) / AR.STAGE_B
    if (d / "complete.json").is_file():
        cb = AR.require_complete(d, AR.STAGE_B)
        if cb["phase4a_artifact_sha256"] != ca["artifact_sha256"]:
            raise SystemExit(f"{d} was built from a different 4A artifact")
        log(f"4B: {d} is complete (immutable); not re-run")
        return {"complete": True, "skipped": True, "dir": d, "payload": cb}
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sc = ctx.scene
    status, render_label, _ = load_membership(a_dir)
    dev = sc.device
    labels = torch.as_tensor(render_label, device=dev)
    trusted = torch.as_tensor(np.isin(status, [CORE, MOVING_S]), device=dev)
    und = torch.as_tensor(np.isin(status, UNDECIDED), device=dev)
    noise = B.SensorNoise(ctx.sensor_model, cfg4b)
    roles = Roles(sc.views, cfg4b.roles())
    canon0 = AR.canonical_hashes(sc.g)
    prop = B.run_proposals(sc, ctx.pv, labels, trusted, und, cfg4b, noise, roles, log=log, cycle=0)
    write_4b(d, prop, cfg4b, noise, roles, sc, trusted)
    atomic_json(d / "resource_report.json", {"seconds": time.time() - t0, "views_decode": ctx.pv.counters.to_dict(),
                                             **_resources(log)})
    AR.require_complete(a_dir, AR.STAGE_A)                    # the 4A artifact is untouched
    if AR.canonical_hashes(sc.g)["combined"] != canon0["combined"]:
        raise RuntimeError("4B changed the canonical original Gaussians")
    payload = AR.write_complete(d, {
        "stage": "4b", "phase4a_dir": str(a_dir), "phase4a_artifact_sha256": ca["artifact_sha256"],
        "canonical_sha256": canon0["combined"], "joint_fingerprint": AR.joint_fingerprint(ctx.joint),
        "input_proof": ca["input_proof"], "config": cfg4b.to_dict(), "sensor": noise.manifest(),
        "sigma_model_m": noise.sigma_model, "candidates": len(prop["candidates"]), "births": 0,
        "scan_complete": True, "views": len(sc.views), "recovery_views": recovery_views(ctx),
        "NOT_FOR_RESULTS": ctx.not_for_results})
    log(f"4B COMPLETE: {len(prop['candidates'])} candidate patches, 0 births")
    return {"complete": True, "dir": d, "payload": payload, "prop": prop}


def write_4b(d, prop, cfg4b, noise, roles, sc, trusted):
    from . import phase4b_surface as B
    S = prop["samples"]
    views = [{"camera": vr.cam.name, "frame": int(vr.frame)} for vr in sc.views]
    atomic_json(d / "scan_manifest.json", {
        "config": cfg4b.to_dict(), "sensor": noise.manifest(), "sigma_model_estimate": getattr(noise, "sigma_estimate", None),
        "roles": roles.manifest(), "codes": B.CODES, "birth_sources": "COVERAGE_GAP | FRONT_SURFACE, PROPOSAL role",
        "per_view": prop["stats"], "counters": prop["counters"], "report": prop["report"], "views": views,
        "renders_per_view": 3, "decodes": "one PreparedViews decode per view (no per-candidate I/O)",
        "uncertainty": {"measurement": "P_y propagated per point", "shared_joint": "not propagated per point; every "
                        "proposal is conditional on the fixed Phase-3 joint", "rendering": "3DGS kernel 3-sigma support, "
                        "used only for containment"}})
    AR.save_npz(d / "residual_taxonomy.npz", codes=torch.stack(prop["codes"]).numpy() if prop["codes"] else np.zeros(0),
                legend=json.dumps(B.CODES), views=json.dumps(views))
    t = sc.tensors()
    ix = trusted.nonzero()[:, 0].cpu().numpy()
    AR.save_npz(d / "moving_frontier.npz", trusted_uid=sc.g.lineage["gaussian_uid"][ix], trusted_row=ix,
                canonical_xyz=t["xyz"][trusted].cpu().numpy(),
                rule=f"trusted = CORE_MOVING_ANCHOR | MOVING_CONFIRMED; a patch is proposed if a cell lies within "
                     f"{cfg4b.frontier_reach_voxels} voxels of a trusted Gaussian's 3-sigma render support")
    AR.save_npz(d / "source_observations.npz", **{k: v for k, v in S.items()}, fragment=prop["fragments"])
    cells = prop["cells"]
    lab = prop["cell_patch"]
    co, corner = prop["co_edges"], prop["corner_edges"]
    pc = {}
    for a, b in corner:
        la, lb = int(lab[a]), int(lab[b])
        if la != lb:
            key = f"{min(la, lb)}-{max(la, lb)}"
            pc[key] = pc.get(key, 0) + 1
    atomic_json(d / "layer_graph.json", {
        "cells": len(cells), "co_layer_edges": int(len(co)), "corner_edges": int(len(corner)),
        "patches": {int(l): int((lab == l).sum()) for l in np.unique(lab)} if len(lab) else {},
        "corner_links_between_patches": pc,
        "rule": "co-layer: adjacent voxels, agreeing normals (sidedness), same per-view fragment when from one view, plane "
                "offsets within k sqrt(se_i^2 + se_j^2 + 2 sigma_layer^2) + tangential x normal_err; corner: separate graph"})
    cands = prop["candidates"]
    torch.save(cands, d / "candidates.pt")
    AR.save_npz(d / "candidates.npz", patch_uid=np.array([c["patch_uid"] for c in cands], dtype="U32"),
                layer_uid=np.array([c["layer_uid"] for c in cands], dtype="U32"),
                n_moving_cells=np.array([len(c["M_cells"]) for c in cands], np.int64),
                n_static_cells=np.array([len(c["S_cells"]) for c in cands], np.int64),
                support_groups=np.array([c["support_groups"] for c in cands], np.int64),
                n_sources=np.array([len(c["sources"]) for c in cands], np.int64),
                hypothesis="MOVING primary; STATIC / OMIT reserved for 4C")
    AR.write_jsonl(d / "candidate_provenance.jsonl", [{
        "patch_uid": c["patch_uid"], "layer_uid": c["layer_uid"], "search_mode": c["search_mode"],
        "frontier_parent_uids": c["frontier_parent_uids"], "n_frontier_parents": c["n_frontier_parents"],
        "support_groups": c["support_groups"], "evidence": c["evidence"], "views": c["views"], "frames": c["frames"],
        "first_observed_frame": c["first_observed_frame"], "source_rows_in_source_observations": c["sources"].tolist(),
        "source_motion_p90_m": c["source_motion_p90_m"], "split_cell_frac": c["split_cell_frac"],
        "proposal_role": "PROPOSAL pixels only (data_roles, assigned before discovery)"} for c in cands])
    atomic_json(d / "ambiguous_measurements.json", {
        "per_view": [{k: s[k] for k in ("camera", "frame", "MIXED_EDGE_AMBIGUOUS", "ORDINARY_GEOMETRY_RESIDUAL",
                                        "ORIGINAL_MEMBERSHIP_CONFLICT", "DEPTH_INVALID", "BEHIND_PREDICTION")}
                     for s in prop["stats"]],
        "note": "pixel-level maps in residual_taxonomy.npz; flagged measurements are kept for audit and never "
                "back-projected as surface evidence"})
    in_cand = np.zeros(len(S["view"]), bool)
    for c in cands:
        in_cand[c["sources"]] = True
    per_view = []
    for vi, s in enumerate(prop["stats"]):
        m = S["view"] == vi
        per_view.append({"camera": s["camera"], "frame": s["frame"], "gap_or_front_px": s["COVERAGE_GAP"] + s["FRONT_SURFACE"],
                         "proposal_role_sources": s["birth_source_px"], "kept_samples": int(m.sum()),
                         "in_candidates": int((m & in_cand).sum())})
    atomic_json(d / "unexplained_coverage_report.json", {
        "per_view": per_view, "counters": prop["counters"], "report": prop["report"],
        "not_in_candidates": int((~in_cand).sum()),
        "reasons": {"not PROPOSAL role": "pixels of FIT / VALIDATION / GUARD roles are never birth sources",
                    "on_existing_support": "ORDINARY_GEOMETRY_RESIDUAL (existing-geometry correction alternative)",
                    "membership_conflict": "ORIGINAL_MEMBERSHIP_CONFLICT (return to 4A)",
                    "too_small / not_frontier": "patch below min_samples / not connected to the trusted frontier"}})


# ====================================================================================== 4C
def stage_4c(ctx: Context, out, a_dir, b_dir, cfg4c, loss_cfg, log=print) -> dict:
    from . import phase4b_surface as B
    from . import phase4c_validate as C
    from .data_roles import Roles
    ca = AR.require_complete(a_dir, AR.STAGE_A)
    cb = AR.require_complete(b_dir, AR.STAGE_B)
    if cb["phase4a_artifact_sha256"] != ca["artifact_sha256"]:
        raise SystemExit("4C: the 4B artifact was built from a different 4A artifact")
    _check_ctx(ctx, ca, "4C", a_dir)
    if cb.get("recovery_views", {}).get("frame_stride", 1) != ctx.recovery_stride:
        raise SystemExit(f"4C: recovery frame stride {ctx.recovery_stride} != the 4B artifact's "
                         f"{cb.get('recovery_views', {}).get('frame_stride', 1)}")
    d = Path(out) / AR.STAGE_C
    if (d / "complete.json").is_file():
        log(f"4C: {d} is complete (immutable); not re-run")
        return {"complete": True, "skipped": True, "dir": d, "payload": AR.require_complete(d, AR.STAGE_C)}
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sc = ctx.scene
    dev = sc.device
    status, render_label, _ = load_membership(a_dir)
    cfg4b = B.Surface4B(**cb["config"])
    noise = B.SensorNoise(ctx.sensor_model, cfg4b, sigma_model=cb["sigma_model_m"])
    noise.sigma_model_source = "from the 4B artifact"
    roles = Roles(sc.views, cfg4b.roles())
    cands = torch.load(Path(b_dir) / "candidates.pt", weights_only=False)
    labels = torch.as_tensor(render_label, device=dev)
    canon0 = AR.canonical_hashes(sc.g)
    jf0 = AR.joint_fingerprint(ctx.joint)
    auth = C.Auth(sc, labels, status)
    E_before = C.full_audit(sc, labels, ctx.pv, loss_cfg)
    decisions, cycles, loo, secs = C.run_4c(auth, ctx.pv, roles, cands, cfg4c, loss_cfg, noise, cfg4b, log=log)
    E_after = C.full_audit(auth.sc, auth.labels, ctx.pv, loss_cfg)
    res = write_4c(d, auth, decisions, cycles, loo, E_before, E_after, status, cfg4c)
    AR.require_complete(a_dir, AR.STAGE_A)
    AR.require_complete(b_dir, AR.STAGE_B)
    reasons = []
    if AR.canonical_hashes(auth.g0)["combined"] != canon0["combined"] or canon0["combined"] != ca["canonical_sha256"]:
        reasons.append("the canonical original Gaussians changed")
    if not torch.equal(auth.labels[:auth.n0], labels):
        reasons.append("an original label changed inside 4C")
    if AR.joint_fingerprint(ctx.joint) != jf0:
        reasons.append("the joint changed during 4C")
    if res["counts"].get(C.UNTESTED, 0):
        reasons.append(f"{res['counts'][C.UNTESTED]} candidates UNTESTED_RESOURCE_LIMIT")
    if len(E_after) != len(sc.views):
        reasons.append("final all-view audit incomplete")
    if res["n_final"] != auth.n0 + res["n_births"]:
        reasons.append("N_final != N0 + N_accepted_births")
    atomic_json(d / "resource_report.json", {"seconds": time.time() - t0, "cycles": cycles,
                                             "views_decode": ctx.pv.counters.to_dict(), **_resources(log)})
    if reasons:
        AR.write_incomplete(d, reasons, {"counts": res["counts"]})
        log(f"4C INCOMPLETE: {reasons}")
        return {"complete": False, "dir": d, "reasons": reasons, "auth": auth, "decisions": decisions}
    payload = AR.write_complete(d, {
        "stage": "4c", "phase4a_artifact_sha256": ca["artifact_sha256"], "phase4b_artifact_sha256": cb["artifact_sha256"],
        "canonical_sha256_originals": canon0["combined"], "joint_fingerprint": jf0, "N0": auth.n0,
        "N_accepted_births": res["n_births"], "N_final": res["n_final"], "counts": res["counts"], "cycles": len(cycles),
        "static_births": 0, "all_view_audit": True, "recovery_views": recovery_views(ctx),
        "NOT_FOR_RESULTS": ctx.not_for_results, "input_proof": ca["input_proof"]})
    log(f"4C COMPLETE: {res['counts']}; {auth.n0} originals + {res['n_births']} moving births")
    return {"complete": True, "dir": d, "payload": payload, "auth": auth, "decisions": decisions, "cycles": cycles}


def write_4c(d, auth, decisions, cycles, loo, E_before, E_after, status4a, cfg4c):
    from . import phase4c_validate as C
    from .phase4a_exact import STATUS_CODES
    g = auth.sc.g
    g.check()
    n0 = auth.n0
    n = len(g)
    t = auth.tensors()
    births = auth.births
    patch_of = np.full(n, "", dtype="U32")
    for b in births:
        patch_of[b["rows"][0]:b["rows"][1]] = b["patch_uid"]
    AR.save_npz(d / "accepted_patches.npz", uid=g.lineage["gaussian_uid"][n0:], patch_uid=patch_of[n0:],
                xyz=g._xyz[n0:].cpu().numpy(), rotation=g._rotation[n0:].cpu().numpy(),
                scaling=g._scaling[n0:].cpu().numpy(), opacity=g._opacity[n0:].cpu().numpy(),
                features_dc=g._features_dc[n0:].cpu().numpy(), material=np.ones(n - n0, np.int8),
                note="canonical anchor coordinates; MOVING; transported by the one fixed joint at every frame")
    counts = {}
    for dd in decisions:
        counts[dd["status"]] = counts.get(dd["status"], 0) + 1
    vis_rows, ev_rows = [], []
    for dd in decisions:
        for key in ("_rows_M", "_rows_S"):
            for r in dd.pop(key, []) or []:
                vis_rows.append({"patch_uid": dd["patch_uid"], "hypothesis": key[-1], **r})
        ev_rows.append(dd)
    AR.write_jsonl(d / "per_patch_evidence.jsonl", ev_rows)
    AR.write_jsonl(d / "per_patch_visibility.jsonl", vis_rows)
    AR.write_jsonl(d / "candidate_statuses.jsonl", [{"patch_uid": dd["patch_uid"], "cycle": dd["cycle"],
                                                    "status": dd["status"], "reason": dd.get("reason")} for dd in decisions])
    prov = []
    for b in births:
        sv = b.get("source_view")
        prov.append({"patch_uid": b["patch_uid"], "layer_uid": b["layer_uid"], "cycle": b["cycle"],
                     "uid_range": [int(b["uids"][0]), int(b["uids"][-1]) + 1],
                     "frontier_parent_uids": b["frontier_parent_uids"],
                     "first_observed_frame_provenance": b["first_observed_frame"],
                     "source_pixels": None if sv is None else [
                         {"camera": auth.sc.views[int(v)].cam.name, "frame": int(auth.sc.views[int(v)].frame), "pixel": int(p)}
                         for v, p in zip(sv, b["source_pix"])][:5000],
                     "n_source_pixels": None if sv is None else int(len(sv)),
                     "creation_time": "none (exists at every frame; first observation is provenance only)"})
    AR.write_jsonl(d / "birth_provenance.jsonl", prov)
    pairs = []
    if len(births) > 1:
        items = [{"p": b["params"], "label": True} for b in births]
        rects = C.version_rects(auth, items)
        for i in range(len(births)):
            for j in range(i + 1, len(births)):
                ov = int(C.FP.rects_overlap(rects[i], rects[j]).sum())
                if ov:
                    pairs.append({"a": births[i]["patch_uid"], "b": births[j]["patch_uid"], "views_overlapping": ov})
    atomic_json(d / "interaction_audit.json", {
        "leave_one_out_final_scene": {k: {kk: vv for kk, vv in v.items() if kk != "_rows"} for k, v in loo.items()},
        "overlapping_accepted_pairs": pairs, "cycles": cycles,
        "note": "per-patch gains are not additive; every accepted patch was re-tested against the FINAL scene"})
    fc = d / "final_canonical"
    fc.mkdir(parents=True, exist_ok=True)
    g.lineage["material"] = auth.labels.cpu().numpy().astype(np.int8)
    g.save_ply(fc / "point_cloud.ply")
    g.save_lineage(fc)
    st = np.full(n, "PHASE4C_BIRTH_ACCEPTED_MOVING", dtype="U40")
    st[:n0] = np.asarray(status4a, dtype="U40")
    AR.save_npz(fc / "material_labels.npz", material=g.lineage["material"], gaussian_uid=g.lineage["gaussian_uid"],
                original_id=g.lineage["original_id"], provenance=g.lineage["provenance"], status=st,
                status_legend_4a=json.dumps(STATUS_CODES),
                unresolved_rendering_note="UNRESOLVED / SUSPECT originals keep their declared provisional render label")
    dup = 0
    if n > n0:
        from scipy.spatial import cKDTree
        X = t["xyz"].cpu().numpy()
        dd_, _ = cKDTree(X[:n0]).query(X[n0:])
        dup = int((dd_ < cfg4c.dup_tol).sum())
    early = []
    order = list(auth.sc.joint.frames)
    for k, v in loo.items():
        b = [x for x in births if x["patch_uid"] == k]
        if not b:
            continue
        f0 = b[0]["first_observed_frame"]
        for r in v["_rows"]:
            if r.get("contradiction") and order.index(r["frame"]) < order.index(f0):
                early.append({"patch_uid": k, "camera": r["camera"], "frame": r["frame"]})
    atomic_json(d / "full_sequence_audit.json", {
        "views": [{"camera": vr.cam.name, "frame": int(vr.frame)} for vr in auth.sc.views],
        "frame_order": [int(f) for f in order], "before_4C": E_before, "after_4C": E_after,
        "E_before": float(sum(r["E"] for r in E_before)), "E_after": float(sum(r["E"] for r in E_after)),
        "views_worse": [{"camera": a["camera"], "frame": a["frame"], "delta": b["E"] - a["E"]}
                        for a, b in zip(E_before, E_after) if b["E"] > a["E"] + 1e-12],
        "births_within_dup_tol_of_an_original": dup, "early_frame_contradictions": early,
        "offmask_covered_px_after": int(sum(r["offmask_covered_px"] for r in E_after)),
        "tolerance_note": "audit counts use an ASSUMED 1 cm tolerance"})
    atomic_json(d / "joint_unchanged_proof.json", {
        "joint_optimized": False, "fingerprint": AR.joint_fingerprint(auth.sc.joint), "joint": auth.sc.joint.to_dict(),
        "note": "no joint / camera tensor is created with requires_grad anywhere in Phase 4"})
    atomic_json(d / "static_alternatives_for_review.json", [
        {"patch_uid": dd["patch_uid"], "status": dd["status"], "reason": dd.get("reason"), "static": dd.get("static")}
        for dd in decisions if dd["status"] in (C.STATIC_ALT, C.AMBIG)])
    atomic_json(d / "return_to_4a_requests.json", [
        {"patch_uid": dd["patch_uid"], "undecided_original_overlap": dd.get("undecided_original_overlap")}
        for dd in decisions if dd["status"] == C.CONFLICT])
    return {"counts": counts, "n_births": n - n0, "n_final": n}


# ====================================================================================== orchestration
def run_all(ctx: Context, out, pol, loss_cfg, cfg4b, cfg4c, resume=False, log=print, recovery_ctx=None) -> dict:
    """4A -> 4B -> 4C, stopping at the first stage that does not complete. Never Phase 5.
    recovery_ctx: optional callable returning the 4B / 4C context (a frame-strided view set); 4A always uses ctx."""
    ra = stage_4a(ctx, out, pol, loss_cfg, resume=resume, log=log)
    if not ra["complete"]:
        return {"stopped_after": "4a", "4a": ra}
    if recovery_ctx is not None:
        ctx = recovery_ctx()
    rb = stage_4b(ctx, out, ra["dir"], cfg4b, log=log)
    if not rb["complete"]:
        return {"stopped_after": "4b", "4a": ra, "4b": rb}
    rc = stage_4c(ctx, out, ra["dir"], rb["dir"], cfg4c, loss_cfg, log=log)
    return {"stopped_after": None if rc["complete"] else "4c", "4a": ra, "4b": rb, "4c": rc}


# ====================================================================================== CLI context
SETUP_KEYS = ("source", "direction", "model", "phase2_dir", "phase3_dir", "joint_type", "cameras", "frame_window",
              "sensor_model", "view_subset", "max_host_gb", "sh_degree", "sh_mode", "seed", "results_root")


def context_from_setup(setup: dict, out_dir, log=print, repo=None, dry_run=False, recovery_stride: int = 1):
    """Build the Phase-4 context from upstream directories (identical for every stage: 4B / 4C read `setup` from the
    4A manifest, and every stage verifies the inputs against the 4A artifact's hashes)."""
    import argparse
    from .gaussians import GaussianTensors
    from .io_setup import git_state, setup_phase4
    from .material_completion import Phase4Scene
    from .rgbd_loss import LossConfig
    from .view_cache import write_cache_manifest
    stride = max(1, int(recovery_stride or 1))
    a = argparse.Namespace(**{**{k: None for k in SETUP_KEYS}, **setup, "dry_run": dry_run, "frame_stride": stride})
    tag = "" if stride == 1 else f"_recovery_stride{stride}"
    up, cams, views, pv, extra = setup_phase4(a, log=log, spill_dir=Path(out_dir) / f"view_spill{tag}")
    if dry_run:
        return up, cams, pv, extra
    cm = write_cache_manifest(Path(out_dir) / f"cache_manifest{tag}.json", pv)
    proof = {"upstream_sha256": dict(up.hashes), "views_combined_sha256": cm.get("combined_sha256"),
             "views": cm.get("views"), "joint": up.joint.to_dict()}
    g = GaussianTensors.load_ply(up.ply)
    if len(g) != up.num_gaussians:
        raise SystemExit(f"PLY has {len(g)} rows, header says {up.num_gaussians}")
    g.active_sh_degree = min(int(setup.get("sh_degree") or 0), g.max_sh_degree)
    scene = Phase4Scene(g, up.joint, views, LossConfig(), sh_mode=setup.get("sh_mode") or "local_SH",
                        sh_degree=g.active_sh_degree)
    return Context(scene=scene, pv=pv, core=np.asarray(up.members, np.int64), setup=dict(setup),
                   input_proof=json.loads(json.dumps(proof, default=str)), sensor_model=setup.get("sensor_model"),
                   not_for_results=bool(setup.get("view_subset")), code=git_state(repo) if repo else {},
                   recovery_stride=stride)
