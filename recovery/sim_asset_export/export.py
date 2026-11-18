"""export_phase5_sim_asset(...): the single entry point (spec 7).

Modes
  auto         (default) always the visual + kinematic asset; physics formats when a valid physics spec is given.
               Missing physics -> BLOCKED_MISSING_PHYSICS, invalid physics -> PHYSICS_REJECTED (no model files).
  required     FULL_PHYSICS_READY or an error (and the staging directory is discarded).
  visual_only  no physics on purpose -> VISUAL_ONLY.
  off          handled by the caller (the Phase-5 driver records it); never selected implicitly.
Read only on the Phase-5 run: only <out_root>/ (default <P5>/assets/) is written.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import artifacts as AF
from .collision import write_obj
from .load_phase5 import load_phase5
from .model import FORMAT_VERSION, asset_dict, build_asset, dynamics_dict, object_yaml_text
from .mjcf_export import NOT_REPRESENTED as MJCF_NOT_REPRESENTED
from .mjcf_export import write_mjcf
from .physics import resolve_physics
from .ply_split import sha256_file, split_articulated_ply
from .schema import (BLOCKED_MISSING_PHYSICS, FULL_PHYSICS_READY, LINK_NAMES, PHYSICS_INCOMPLETE_RUNTIME,
                     PHYSICS_PREVIEW_ASSUMPTIONS, PHYSICS_REJECTED, VISUAL_ONLY, PhysicsSpecError, parse_physics_spec,
                     parse_registration)
from .urdf_export import write_urdf
from .usd_export import NOT_REPRESENTED as USD_NOT_REPRESENTED
from .usd_export import write_usd
from . import validate as V

MODES = ("auto", "required", "visual_only")


class ExportError(RuntimeError):
    pass


@dataclass
class ExportResult:
    scene: str
    asset_root: Path
    status: str
    visual_complete: bool
    physics_complete: bool
    usd_complete: bool
    missing_inputs: list
    hashes: dict
    validation_report: dict
    NOT_FOR_RESULTS: bool
    reused: bool = False
    warnings: list = field(default_factory=list)
    manifest_sha256: str | None = None

    def summary(self) -> dict:
        return {"asset_export_status": self.status, "asset_export_dir": None, "visual_complete": self.visual_complete,
                "physics_complete": self.physics_complete, "usd_complete": self.usd_complete,
                "missing_inputs": self.missing_inputs, "reused": self.reused, "NOT_FOR_RESULTS": self.NOT_FOR_RESULTS}


def load_registration(path) -> object:
    import yaml
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise PhysicsSpecError(f"registration file not found: {p}")
    d = yaml.safe_load(p.read_text())
    if isinstance(d, dict) and "registration" in d and len(d) == 1:
        d = d["registration"]
    return parse_registration(d, f"{p.name}")


def _spec_files(spec) -> dict:
    out = {"spec": sha256_file(spec.path)}
    for s, c in spec.collision.items():
        out[f"collision.{s}.source"] = sha256_file(c.source)
        for i, p in enumerate(c.convex_parts):
            out[f"collision.{s}.convex_parts[{i}]"] = sha256_file(p)
    return out


def preflight_required(physics_spec, registration=None, joint=None, usd_python=None, mujoco_python=None) -> dict:
    """Before expensive Phase-5 training (mode required): spec parses and validates, registration is not an unverified
    gauge, both export runtimes import. Raises ExportError listing every problem."""
    problems = []
    if physics_spec is None:
        problems.append("--asset-physics-spec is required by --asset-export-mode required")
    spec = reg = None
    if physics_spec is not None:
        try:
            spec = parse_physics_spec(physics_spec)
        except PhysicsSpecError as e:
            problems.append(f"physics spec: {e}")
    if registration is not None:
        try:
            reg = load_registration(registration)
        except PhysicsSpecError as e:
            problems.append(f"registration: {e}")
    eff = reg or (spec.registration if spec is not None else None)
    if spec is not None and (eff is None or eff.gravity_alignment == "unverified"):
        problems.append("registration: FULL_PHYSICS_READY needs a measured / declared gravity alignment (registration "
                        "block or --asset-registration); the W = Wc gauge alone is unverified")
    if spec is not None and joint is not None:
        for s, c in spec.collision.items():
            if c.at_original_frame is not None and c.at_original_frame not in joint.frames:
                problems.append(f"collision.{s}.at_original_frame {c.at_original_frame} is not a Phase-3 frame")
        for lk, i in spec.inertial.items():
            if i.at_original_frame is not None and i.at_original_frame not in joint.frames:
                problems.append(f"links.{lk}.at_original_frame {i.at_original_frame} is not a Phase-3 frame")
        if spec.joint.limits_from_observed_range:
            problems.append("joint: limits_from_observed_range is a preview assumption, not FULL_PHYSICS_READY")
        for s, c in spec.collision.items():
            if c.approximation == "convex_hull" or c.source_kind == "reconstructed_rgbd":
                problems.append(f"collision.{s}: {c.approximation or c.source_kind} is a preview assumption")
    rt = {}
    for kind, ex in (("usd", usd_python), ("mujoco", mujoco_python)):
        try:
            rt[kind] = V.find_runtime(kind, ex)
        except V.ValidationError as e:
            rt[kind] = {"python": None}
            problems.append(str(e))
        if not rt[kind]["python"]:
            problems.append(f"no {kind} runtime (pass --asset-{kind}-python or set RIGEULER_{kind.upper()}_PYTHON)")
    if problems:
        raise ExportError("asset export preflight (mode required) failed:\n  - " + "\n  - ".join(problems))
    return {"spec": str(spec.path), "runtimes": rt}


def export_phase5_sim_asset(phase5_dir, scene: str | None = None, physics_spec=None, registration=None,
                            mode: str = "auto", out_root=None, usd_python=None, verify: bool = True,
                            mujoco_python=None, sapien_python=None, revision: str | None = None,
                            completion: dict | None = None, setup: dict | None = None, log=print) -> ExportResult:
    t0 = time.time()
    if mode not in MODES:
        raise ExportError(f"mode must be one of {MODES} (off is handled by the caller)")
    try:
        inp = load_phase5(phase5_dir, completion=completion, setup=setup, scene=scene)
    except Exception as e:
        raise ExportError(f"Phase-5 input: {e}") from e
    p5 = inp.phase5_dir
    out_root = Path(out_root) if out_root is not None else p5 / "assets"
    reg = None
    if registration is not None:
        try:
            reg = load_registration(registration)
        except PhysicsSpecError as e:
            raise ExportError(f"registration: {e}") from e

    # ---------------------------------------------------------------- physics (validated before anything is written)
    spec = phys = None
    missing, rejected = [], None
    if mode == "visual_only":
        pass
    elif physics_spec is None:
        missing = ["physics spec (--asset-physics-spec): collision meshes, mass / COM / inertia per link, physical "
                   "joint stops, joint velocity / effort limits, damping, Coulomb friction"]
        if reg is None:
            missing.append("registration (T_world_capture + gravity alignment)")
        if mode == "required":
            raise ExportError("mode required: " + "; ".join(missing))
    else:
        try:
            spec = parse_physics_spec(physics_spec)
            # the link frames are needed to resolve physics; built from the frozen joint alone (same rule as
            # model.build_asset, which re-derives and cross-checks them)
            from .frames import joint_frame
            T_B_J0 = (joint_frame("revolute", inp.joint.axis, inp.joint.point)[0] if inp.joint.type == "revolute" else
                      joint_frame("prismatic", inp.joint.axis, moving_xyz=inp.xyz[inp.part_id == 1])[0])
            phys = resolve_physics(spec, inp, T_B_J0, reg)
        except PhysicsSpecError as e:
            if mode == "required":
                raise ExportError(f"physics spec rejected: {e}") from e
            rejected = str(e)
            missing = [f"valid physics spec: {e}"]
            spec = phys = None

    rt = {"usd": {"python": None}, "mujoco": {"python": None}, "sapien": {"python": None}}
    if phys is not None:
        try:
            rt["usd"] = V.find_runtime("usd", usd_python)
            rt["mujoco"] = V.find_runtime("mujoco", mujoco_python)
            if verify:
                rt["sapien"] = V.find_runtime("sapien", sapien_python)
        except V.ValidationError as e:
            raise ExportError(str(e)) from e
        if mode == "required":
            miss = [k for k in ("usd", "mujoco") if not rt[k]["python"]]
            if miss:
                raise ExportError(f"mode required: no {miss} runtime")

    sig_obj = {"format": FORMAT_VERSION, "code_sha256": AF.code_sha256(), "mode": mode, "scene": inp.scene,
               "articulated_ply_sha256": inp.ply_sha256, "joint_fingerprint": inp.completion.get("joint_fingerprint"),
               "final_gaussians": int(len(inp.part_id)),
               "physics_files": None if spec is None else _spec_files(spec), "physics_rejected": rejected,
               "registration": None if reg is None else reg.to_dict(),
               "runtimes": {k: bool(v.get("python")) for k, v in rt.items()}}
    sig = AF.signature(sig_obj)
    dirname = AF.asset_dirname(inp.scene, revision)
    final = out_root / dirname
    if final.exists():
        try:
            ex = AF.verify_existing(final)
        except AF.ArtifactError as e:
            raise ExportError(str(e)) from e
        if ex["complete"].get("input_signature") != sig:
            raise ExportError(f"{final} holds an export of DIFFERENT inputs (status {ex['complete'].get('status')}); "
                              f"not replaced silently -- pass --asset-revision NAME to write a new revision")
        log(f"asset export: {final} already exported from identical inputs; verified and reused")
        man = ex["manifest"]
        return ExportResult(inp.scene, final, ex["complete"]["status"], man["readiness"]["visual_complete"],
                            man["readiness"]["physics_complete"], man["readiness"]["usd"] == V.PASS,
                            man["missing_inputs"], man["files"], man["validation"], inp.not_for_results, reused=True,
                            warnings=man.get("warnings", []), manifest_sha256=ex["complete"]["manifest_sha256"])

    stg = AF.Staging(out_root, dirname)
    try:
        S = stg.path
        split = split_articulated_ply(inp.ply_path, {"static_link": S / "visual" / "static.ply",
                                                     "moving_link": S / "visual" / "moving.ply"})
        asset = build_asset(inp, split, phys, reg)
        asset.source = {"phase5_run_name": p5.name, "completion_source": inp.completion_source,
                        "complete_json_sha256": inp.complete_sha256, "articulated_ply_sha256": inp.ply_sha256,
                        "train_mode": inp.completion.get("train_mode"), "final_gaussians": int(len(inp.part_id)),
                        "joint_fingerprint": {k: inp.completion["joint_fingerprint"][k]
                                              for k in ("files", "transforms_sha256")},
                        "joint_geometry_sha256": inp.joint.geometry_sha256,
                        "joint_trajectory_sha256": inp.joint.trajectory_sha256}
        # lineage / membership sidecar
        lin = {}
        for lk, r in split.rows.items():
            pre = lk.split("_")[0]
            lin[f"{pre}_source_row"] = r
            for k, v in inp.lineage.items():
                lin[f"{pre}_{k}"] = v[r]
        if inp.unresolved_originals is not None:
            for k, v in inp.unresolved_originals.items():
                lin[f"phase5_unresolved_originals_{k}"] = v
        lin["note"] = np.array("row k of visual/<link>.ply = row <link>_source_row[k] of the Phase-5 articulated PLY; "
                               "<link>_unresolved = 1: 4A left the row (or its ancestor) undecided -- its link is the "
                               "frozen RENDER assignment, membership UNCERTAIN, not an identified object part")
        np.savez(S / "lineage.npz", **lin)
        n_unres = {lk: int(inp.lineage["unresolved"][r].sum()) for lk, r in split.rows.items()}
        tr = inp.trajectory
        (S / "joint_trajectory.csv").write_text("original_frame,q,posterior_sd_q\n" + "".join(
            f"{int(f)},{float(q)!r},{float(s)!r}\n" for f, q, s in zip(tr["frame"], tr["q"], tr["posterior_sd_q"])))

        report = {"ply_split": {"status": V.PASS, **split.verified, "fields": split.fields, "dtypes": split.dtypes,
                                "sh_degree": split.sh_degree, "counts": split.counts,
                                "unresolved_rows": n_unres}}
        report["analytic_kinematics"] = V.check_analytic(asset, inp.joint, inp.xyz[inp.part_id == 1])
        fmt = {"urdf": "NOT_WRITTEN", "usd": "NOT_WRITTEN", "mjcf": "NOT_WRITTEN"}
        if phys is not None:
            cf = {}
            for lk in LINK_NAMES:
                pre = lk.split("_")[0]
                c = phys.collision[lk]
                surf_rel = f"physics/collision/{pre}.obj"
                hdr = [f"RigEuler3DGS sim asset {inp.scene}: {lk} collision surface, {lk} frame, metres",
                       f"source {c.report['source_file']} ({c.provenance}, {c.source_kind}); T_link_capture applied"]
                write_obj(c.surface, S / surf_rel, hdr)
                if c.approximation == "source_is_convex":
                    parts = [surf_rel]
                else:
                    parts = []
                    for k, part in enumerate(c.parts):
                        rel = f"physics/collision/{pre}_convex_{k:03d}.obj"
                        write_obj(part, S / rel, [f"{lk} convex proxy {k} ({c.approximation}), {lk} frame, metres"])
                        parts.append(rel)
                cf[lk] = {"surface": surf_rel, "parts": parts, "approximation": c.approximation}
            asset.collision_files = cf
            AF.write_json(S / "physics" / "collision_report.json",
                          {lk: {**phys.collision[lk].report, "files": cf[lk]} for lk in LINK_NAMES})
            import yaml
            (S / "physics" / "dynamics.yaml").write_text(
                "# GENERATED from the validated physics spec; the single source the three simulator files share\n" +
                yaml.safe_dump(dynamics_dict(asset), sort_keys=False, width=1000))
            write_urdf(asset, S / "maniskill" / "model.urdf")
            report["urdf"] = V.check_urdf(S / "maniskill" / "model.urdf", asset, inp.joint)
            write_mjcf(asset, S / "mujoco" / "model.xml")
            report["mjcf"] = V.check_mujoco(S / "mujoco" / "model.xml", asset, inp.joint, rt["mujoco"]["python"])
            if rt["usd"]["python"]:
                rb = write_usd(asset, S / "isaac" / "model.usd", rt["usd"]["python"])
                report["usd"] = V.check_usd(rb, asset, inp.joint, S)
            else:
                report["usd"] = V.check_usd(None, asset, inp.joint, S)
            fmt = {k: report[k]["status"] for k in ("urdf", "usd", "mjcf")}
            report["engine_parameter_mapping"] = {"mjcf_not_represented": MJCF_NOT_REPRESENTED,
                                                  "usd_not_represented": USD_NOT_REPRESENTED}
        # status
        if mode == "visual_only":
            status = VISUAL_ONLY
        elif phys is None:
            status = PHYSICS_REJECTED if rejected else BLOCKED_MISSING_PHYSICS
        elif V.NOT_RUN in fmt.values():
            status = PHYSICS_INCOMPLETE_RUNTIME
            missing += [f"{k} runtime (format written but not validated / not written)" for k, v in fmt.items()
                        if v == V.NOT_RUN]
        elif phys.assumptions:
            status = PHYSICS_PREVIEW_ASSUMPTIONS
        else:
            status = FULL_PHYSICS_READY
        (S / "object.yaml").write_text(object_yaml_text(asset, status))
        report["object_yaml"] = V.check_object_yaml(S / "object.yaml", asset)
        if phys is not None and verify:
            report["sapien_parttwin1d"] = V.check_sapien(S / "maniskill" / "model.urdf", S / "object.yaml", asset,
                                                         inp.joint, rt["sapien"]["python"],
                                                         {lk: S / asset.visual[lk]["ply"] for lk in LINK_NAMES})
        report["isaac_runtime"] = {"status": V.NOT_RUN, "reason": "Isaac Sim PhysX runtime is not started by the "
                                   "exporter; run tools/sim_runtimes/isaac_runtime_check.py with Isaac's python.sh"}
        fails = {k: v.get("errors") for k, v in report.items() if isinstance(v, dict) and v.get("status") == V.FAIL}
        if fails:
            raise ExportError(f"validation FAILED: {json.dumps(fails, default=str)[:3000]}")
        readiness = {"visual_complete": True, "kinematics_complete": report["analytic_kinematics"]["status"] == V.PASS,
                     "physics_complete": status == FULL_PHYSICS_READY, "urdf": fmt["urdf"], "usd": fmt["usd"],
                     "mjcf": fmt["mjcf"], "mujoco_compile": fmt["mjcf"],
                     "sapien_runtime": report.get("sapien_parttwin1d", {}).get("status", V.NOT_RUN),
                     "isaac_runtime": V.NOT_RUN,
                     "physics_assumptions": [] if phys is None else phys.assumptions,
                     "all_physics_measured": bool(phys is not None and phys.provenance_summary["all_measured"])}
        if mode == "required" and status != FULL_PHYSICS_READY:
            raise ExportError(f"mode required: export reached {status}, not {FULL_PHYSICS_READY} "
                              f"(assumptions {readiness['physics_assumptions']}, formats {fmt})")
        hashes_pre = AF.file_hashes(S)
        import yaml
        (S / "asset.yaml").write_text("# GENERATED canonical asset specification (RigEuler3DGS Phase-5 sim asset)\n" +
                                      yaml.safe_dump(asset_dict(asset, status, readiness, hashes_pre, missing),
                                                     sort_keys=False, width=1000))
        forbidden = sorted({str(p5.resolve()), str(out_root.resolve()), str(S.resolve()), str(Path.home())})
        report["portability"] = V.check_portable(S, forbidden)
        if report["portability"]["status"] == V.FAIL:
            raise ExportError(f"absolute paths in runtime files: {report['portability']['errors']}")
        warnings = list(inp.warnings) + ([] if phys is None else phys.warnings)
        manifest = {"format": FORMAT_VERSION, "scene": inp.scene, "status": status, "mode": mode,
                    "readiness": readiness, "missing_inputs": missing, "warnings": warnings,
                    "NOT_FOR_RESULTS": inp.not_for_results, "input_signature": sig, "inputs": sig_obj,
                    "provenance_paths": {"phase5_dir": str(p5.resolve()), "physics_spec": None if spec is None else
                                         str(spec.path), "registration": None if registration is None else
                                         str(Path(registration).resolve())},
                    "runtimes": rt, "validation": report, "seconds": time.time() - t0,
                    "files": AF.file_hashes(S)}
        AF.write_json(S / AF.MANIFEST, manifest)
        dst = stg.promote({"status": status, "input_signature": sig, "scene": inp.scene,
                           "requested_mode": mode, "validated": True})
    except BaseException as e:
        stg.abort(e)
        if isinstance(e, ExportError):
            raise
        raise ExportError(f"{type(e).__name__}: {e}") from e
    msha = sha256_file(dst / AF.MANIFEST)
    log(f"asset export [{status}] -> {dst} ({time.time() - t0:.1f} s): URDF {fmt['urdf']}, USD {fmt['usd']}, MJCF "
        f"{fmt['mjcf']}" + (f"; missing {missing}" if missing else ""))
    return ExportResult(inp.scene, dst, status, True, status == FULL_PHYSICS_READY, fmt["usd"] == V.PASS, missing,
                        manifest["files"], report, inp.not_for_results, warnings=warnings, manifest_sha256=msha)
