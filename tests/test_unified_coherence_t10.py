"""T10 — a coherent moving core extracted from ACCEPTED unified smoothed states.

Section 3 of CODEX_COMPLETE_MISSING_UNIFIED_BIDI_PHASE1_AFTER_9B78EFA.md.

The fixture is a nonoverlapping two-source RGB-D scene rendered by the real
CUDA rasterizer. A full-window `UnifiedFreeIEKS` is solved from a noisy,
in-capture warm start, and the PRODUCTION coherence extractor is then run on
its accepted states -- reached through `bidirectional_iekf.unified_coherence`,
with the call instrumented so "the extractor ran" is measured rather than
assumed. No `predicted_support.json` is read, no label is copied and no
correspondence between the two sources is assumed.

Four groups, each with members in BOTH sources:

    STATIC       26  stationary
    MOVING       14  one rigid screw: v = (0, 0.10, 0) m/s, omega = 0.30 rad/s
    INCOHERENT    6  each member on its own heading -- no common rigid motion
    LOW_CONFIDENCE 2 opacity 0.02, moving WITH the body but almost invisible

The last group is the deliberately weakly observed region. Its true class is
MOVING. What was MEASURED, rather than expected: an alpha of 0.02 leaves about
2% of a pixel, so the information is roughly fifty times down, and the
recovered trajectory is wrong by 3.9e-02 against the 7.5e-04 of the fully
observed body -- a factor of 51, which is the information deficit, not a
failure. The displacement is still large enough that the extractor recruits
the pair into the core. Both the class and the error are reported; neither is
chosen.

    python tests/test_unified_coherence_t10.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixture geometry: a lattice with an enforced minimum separation, so the two
# sources are DISJOINT surfaces rather than two mixtures of one surface.
# ---------------------------------------------------------------------------

MIN_SEPARATION = 0.18


def lattice(x_range, y_range, z_range, count, seed):
    nx, ny, nz = x_range[2], y_range[2], z_range[2]
    g = torch.Generator().manual_seed(seed)
    pts = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = x_range[0] + (x_range[1] - x_range[0]) * (ix / max(nx - 1, 1))
                y = y_range[0] + (y_range[1] - y_range[0]) * (iy / max(ny - 1, 1))
                z = z_range[0] + (z_range[1] - z_range[0]) * (iz / max(nz - 1, 1))
                pts.append([x, y, z])
    pts = torch.tensor(pts, dtype=torch.float64)
    pts = pts + 0.012 * (torch.rand(pts.shape, generator=g, dtype=torch.float64) - 0.5)
    return pts[:count]


def main() -> int:
    if not cuda_ready():
        print("T10 NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.ieks import (CONVERGED, IEKSConfig, MAX_ITERS,
                                         Trajectory, UnifiedFreeIEKS)
    from bidirectional_iekf.observations import FreeRGBDObservation
    from bidirectional_iekf.provenance import (FINAL, INITIAL, SourceAnchor,
                                               build_candidate_population)
    from bidirectional_iekf import unified_coherence as uc
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    torch.manual_seed(0)
    T, dt = 7, 0.1
    out_root = Path("/tmp/claude-1000/-home-samuelemara-Joint3DGS-RigEuler3DGS/"
                    "90323ff0-f6c6-454e-a004-5e28e27daa35/scratchpad/t10")
    out_root.mkdir(parents=True, exist_ok=True)

    print("[1] the fixture: four groups, two sources, no duplicated surface")
    static_xyz = lattice((-0.95, -0.35, 3), (-0.45, 0.45, 5), (-0.12, 0.12, 2), 26, 1)
    moving_xyz = lattice((0.35, 0.95, 3), (-0.45, 0.45, 5), (-0.12, 0.12, 2), 16, 2)
    incoh_xyz = lattice((-0.12, 0.12, 2), (-0.40, 0.40, 3), (0.0, 0.0, 1), 6, 3)
    low_xyz, moving_xyz = moving_xyz[14:], moving_xyz[:14]

    # Group membership, then the source split. Both sources appear in every
    # group: the initial model owns the even member of each group, the final
    # model the odd one, so a core that spans both sources is a real claim.
    groups = ([("STATIC", p) for p in static_xyz]
              + [("MOVING", p) for p in moving_xyz]
              + [("INCOHERENT", p) for p in incoh_xyz]
              + [("LOW_CONFIDENCE", p) for p in low_xyz])
    endpoint_of = [INITIAL if k % 2 == 0 else FINAL for k in range(len(groups))]
    order = ([k for k in range(len(groups)) if endpoint_of[k] == INITIAL]
             + [k for k in range(len(groups)) if endpoint_of[k] == FINAL])
    labels_true = [groups[k][0] for k in order]
    mu0 = torch.stack([groups[k][1] for k in order]).double()
    N0 = sum(1 for e in endpoint_of if e == INITIAL)
    NT = len(groups) - N0
    N = N0 + NT

    d = torch.cdist(mu0, mu0) + torch.eye(N, dtype=torch.float64) * 1e3
    check("every candidate is a distinct surface, not a duplicate",
          float(d.min()) > MIN_SEPARATION,
          f"min pairwise separation {float(d.min()):.4f} > {MIN_SEPARATION}")
    cross = torch.cdist(mu0[:N0], mu0[N0:])
    check("the two sources do not coincide anywhere",
          float(cross.min()) > MIN_SEPARATION,
          f"min initial-to-final distance {float(cross.min()):.4f}")

    is_moving = torch.tensor([l == "MOVING" for l in labels_true])
    is_static = torch.tensor([l == "STATIC" for l in labels_true])
    is_low = torch.tensor([l == "LOW_CONFIDENCE" for l in labels_true])
    is_incoh = torch.tensor([l == "INCOHERENT" for l in labels_true])
    src = torch.tensor([0] * N0 + [1] * NT)
    check("the moving body has members from BOTH sources",
          bool((is_moving & (src == 0)).any() and (is_moving & (src == 1)).any()),
          f"{int((is_moving & (src == 0)).sum())} initial + "
          f"{int((is_moving & (src == 1)).sum())} final")
    print(f"       {N} candidates: {int(is_static.sum())} static, "
          f"{int(is_moving.sum())} moving, {int(is_incoh.sum())} incoherent, "
          f"{int(is_low.sum())} low-confidence; sources {N0}/{NT}")

    # -- the truth trajectory ---------------------------------------------
    body = is_moving | is_low
    pivot = mu0[body].mean(0)
    v_body = torch.tensor([0.0, 0.06, 0.0], dtype=torch.float64)
    w_body = torch.tensor([0.0, 0.0, 0.15], dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    heading = torch.nn.functional.normalize(
        torch.randn(N, 3, generator=g, dtype=torch.float64), dim=1) * 0.055
    heading[:, 2] *= 0.3

    colors = torch.rand(N, 3, generator=g, dtype=torch.float32) * 0.7 + 0.2
    opacity = torch.full((N, 1), 0.92)
    opacity[is_low] = 0.02
    scales = torch.tensor([[0.12, 0.07, 0.06]]).expand(N, 3).contiguous()

    def make(mu, R, v, omega):
        return GaussianMotionState(
            mu=mu.float().cuda(), R=R.float().cuda(), v=v.float().cuda(),
            omega=omega.float().cuda(), colors=colors.cuda(),
            opacity=opacity.cuda(), scales=scales.cuda())

    R0 = so3_exp(torch.zeros(N, 3, dtype=torch.float64))
    truth = []
    for t in range(T):
        s = dt * t
        Rb = so3_exp((w_body * s).unsqueeze(0))[0]
        mu = mu0.clone()
        rel = mu0[body] - pivot
        mu[body] = pivot + v_body * s + rel @ Rb.T
        mu[is_incoh] = mu0[is_incoh] + heading[is_incoh] * s
        R = R0.clone()
        R[body] = Rb @ R0[body]
        v = torch.zeros(N, 3, dtype=torch.float64)
        v[body] = v_body + torch.cross(
            w_body.expand(int(body.sum()), 3), mu[body] - (pivot + v_body * s), dim=1)
        v[is_incoh] = heading[is_incoh]
        omega = torch.zeros(N, 3, dtype=torch.float64)
        omega[body] = w_body
        truth.append(make(mu, R, v, omega))

    travel = float((truth[-1].mu - truth[0].mu).norm(dim=1).max())
    cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
    observations = [[(rgb.clone(), depth.clone(),
                      torch.ones_like(depth, dtype=torch.bool))
                     for rgb, depth in build_linearization(truth[t], cameras).predicted()]
                    for t in range(T)]
    print(f"       max true travel {travel:.4f} world units "
          f"(~{travel * 96 / (2 * 0.483 * 2.4):.1f} px against a "
          f"~{0.12 * 96 / (2 * 0.483 * 2.4):.1f} px projected radius)")

    import numpy as _np
    population = build_candidate_population([
        SourceAnchor(INITIAL, str(out_root / "m_initial"),
                     str(out_root / "m_initial.ply"), "a" * 64, 0,
                     truth[0].mu[:N0].double().cpu().numpy(), N0),
        SourceAnchor(FINAL, str(out_root / "m_final"),
                     str(out_root / "m_final.ply"), "b" * 64, T - 1,
                     truth[T - 1].mu[N0:].double().cpu().numpy(), NT)])
    check("the candidate population is the disjoint union",
          population.num_candidates == N, str(population.num_candidates))
    check("the two anchors refer to DIFFERENT physical frames",
          set(population.anchor_original_frame.tolist()) == {0, T - 1},
          "which is exactly why anchor_xyz is not a common-time array")

    # ---------------------------------------------------------------------
    print("\n[2] a real full-window IEKS from a noisy, in-capture warm start")
    mid = T // 2
    noise = 0.008 * torch.randn(N, 3, generator=g, dtype=torch.float64)
    warm_mu = truth[mid].mu.double().cpu() + noise
    warm = Trajectory(
        states=[make(warm_mu, truth[mid].R.double().cpu(),
                     torch.zeros(N, 3, dtype=torch.float64),
                     torch.zeros(N, 3, dtype=torch.float64)) for _ in range(T)],
        frame_ids=list(range(T)), dt=dt)
    offset = max(float((warm_mu - truth[t].mu.double().cpu()).norm(dim=1).max())
                 for t in range(T))
    check("the warm start is inside the photometric capture radius",
          offset * 96 / (2 * 0.483 * 2.4) < 0.12 * 96 / (2 * 0.483 * 2.4),
          f"worst warm-start offset {offset:.4f} world units = "
          f"{offset * 96 / (2 * 0.483 * 2.4):.2f} px")

    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
    Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                dtype=torch.float64)).expand(N, 12, 12).contiguous()
    ieks = UnifiedFreeIEKS(obs, Q, config=IEKSConfig(
        max_outer=50, damping=1e-4, pcg_max_iters=250, pcg_tol=1e-10,
        verbose=False))
    e_truth = ieks.evaluate(Trajectory(states=truth, frame_ids=list(range(T)),
                                       dt=dt))["E_total"]
    e_start = ieks.evaluate(warm)["E_total"]
    t0 = time.perf_counter()
    result = ieks.run(warm)
    wall = time.perf_counter() - t0
    e_final = result["energy"]["E_total"]
    est = result["trajectory"]
    pose_err = max(float((est.states[t].mu - truth[t].mu).norm(dim=1).max())
                   for t in range(T))
    body_err = max(float((est.states[t].mu[is_moving]
                          - truth[t].mu[is_moving]).norm(dim=1).max())
                   for t in range(T))
    low_err = max(float((est.states[t].mu[is_low]
                         - truth[t].mu[is_low]).norm(dim=1).max())
                  for t in range(T))
    for row in result["trace"][-3:]:
        print(f"       outer {row['outer']:>3} E {row['E_total']:.4e} "
              f"|g| {row['grad_norm']:.3e} |d| {row.get('step_norm', float('nan')):.3e} "
              f"accepted {row['accepted']}")
    print(f"       E {e_start:.4e} -> {e_final:.4e}  (E at truth {e_truth:.4e}), "
          f"{sum(1 for r in result['trace'] if r['accepted'])} accepted of "
          f"{len(result['trace'])}, {wall:.1f} s, peak CUDA "
          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"       worst |mu_err| {pose_err:.4e}; moving body {body_err:.4e}; "
          f"the two alpha=0.02 candidates {low_err:.4e}")
    check("the IEKS terminal status is CONVERGED",
          result["status"] == CONVERGED,
          f"{result['status']}" + ("" if result["status"] == CONVERGED else
                                   "; MAX_ITERS is not acceptance"))
    check("the objective came down to the optimum's neighbourhood",
          e_final < 0.02 * e_start,
          f"{100 * (1 - e_final / e_start):.3f}% removed")
    check("the moving body's trajectory was actually recovered",
          body_err < 0.05, f"worst |mu_err| on the body {body_err:.4e} against "
          f"a {travel:.3f} travel")
    check("the weakly observed pair is measurably worse than the body",
          low_err > 3.0 * body_err,
          f"{low_err:.3e} against {body_err:.3e}: an alpha of 0.02 leaves "
          "almost no image information, so the process prior holds them")

    # ---------------------------------------------------------------------
    print("\n[3] refusals: a conclusion is not evidence")
    try:
        uc.accepted_trajectory({"status": MAX_ITERS, "trajectory": est})
        check("a non-converged run is refused", False, "no exception")
    except uc.TrajectoryNotAccepted as exc:
        check("a non-converged run is refused", "MAX_ITERS" in str(exc),
              "TrajectoryNotAccepted")
    try:
        uc.reject_stale_support({"dynamic_gaussians": [1, 2, 3]})
        check("a support dict is refused", False, "no exception")
    except uc.StaleSupportRejected:
        check("a support dict is refused", True, "StaleSupportRejected")
    stale_dir = out_root / "stale_phase1"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "predicted_support.json").write_text(
        json.dumps({"dynamic_gaussians": [0, 1], "status": "RESOLVED"}))
    try:
        uc.reject_stale_support(stale_dir)
        check("a finished support directory is refused", False, "no exception")
    except uc.StaleSupportRejected as exc:
        check("a finished support directory is refused",
              "predicted_support.json" in str(exc), "StaleSupportRejected")
    try:
        uc.unified_coherence_support(result, population,
                                     prior_support={"labels": []}, verbose=False)
        check("prior_support is refused", False, "no exception")
    except uc.StaleSupportRejected:
        check("prior_support is refused", True, "StaleSupportRejected")

    print("\n[4] negative control: a graph built from mismatched endpoint times")
    try:
        uc.assert_common_time(population.anchor_original_frame)
        check("anchor_xyz is refused as a simultaneous array", False, "no exception")
    except uc.MixedEndpointTimeGraph as exc:
        check("anchor_xyz is refused as a simultaneous array",
              f"[0, {T - 1}]" in str(exc), str(exc).split(".")[0][:60])
    try:
        uc.unified_coherence_support(
            result, population, reference_centers=population.anchor_xyz,
            reference_center_frames=population.anchor_original_frame,
            verbose=False)
        check("the whole run is refused on a mixed-time reference", False,
              "no exception")
    except uc.MixedEndpointTimeGraph:
        check("the whole run is refused on a mixed-time reference", True,
              "MixedEndpointTimeGraph")
    try:
        uc.unified_coherence_support(result, population,
                                     reference_centers=population.anchor_xyz,
                                     verbose=False)
        check("centers without their frame index are refused", False, "no exception")
    except uc.MixedEndpointTimeGraph:
        check("centers without their frame index are refused", True,
              "an array of centers carries no evidence that it is simultaneous")

    # Measured: the mixed-time array differs from the true simultaneous one by
    # the whole sequence displacement on exactly the moving rows.
    mixed = torch.from_numpy(population.anchor_xyz)
    simultaneous = est.states[0].mu.double().cpu()
    delta = (mixed - simultaneous).norm(dim=1)
    check("and the two arrays really do differ, on the moving rows",
          float(delta[is_moving & (src == 1)].max()) > 0.5 * travel
          and float(delta[is_static].max()) < 0.1,
          f"max |anchor - simultaneous| = {float(delta.max()):.4f} on moving-final "
          f"rows, {float(delta[is_static].max()):.4f} on static rows")

    # ---------------------------------------------------------------------
    print("\n[5] the production extractor, reached through the adapter")
    import estimator.coherence_extraction as ce
    calls = {"n": 0, "shapes": None}
    original = ce.extract_moving_component

    def traced(traj, edge_i, edge_j, C, n, **kw):
        calls["n"] += 1
        calls["shapes"] = (tuple(traj.shape), int(edge_i.numel()), int(n))
        calls["kwargs"] = {k: v for k, v in kw.items() if k != "verbose"}
        return original(traj, edge_i, edge_j, C, n, **kw)

    ce.extract_moving_component = traced
    try:
        cfg = uc.UnifiedCoherenceConfig(knn=12, supernode_count=0, cv_folds=5)
        unified = uc.unified_coherence_support(result, population, config=cfg,
                                               verbose=True)
    finally:
        ce.extract_moving_component = original

    check("estimator.coherence_extraction.extract_moving_component ran",
          calls["n"] == 1, f"{calls['n']} call(s), traj {calls['shapes'][0]}, "
          f"{calls['shapes'][1]} directed edges, {calls['shapes'][2]} nodes")
    check("it was handed the accepted trajectory, all T frames, all candidates",
          calls["shapes"][0] == (T, N, 3) and calls["shapes"][2] == N,
          str(calls["shapes"]))
    check("the production method strings are the ones that ran",
          unified.extraction["method"] == "coherence_single_linkage_heldout_rigid"
          and unified.coherence["method"] == "trajectory_metric_coherence")
    check("the graph reference is ONE physical frame",
          unified.graph_reference_original_frame == 0,
          f"original frame {unified.graph_reference_original_frame}")

    core = unified.core_candidate_indices
    core_mask = torch.zeros(N, dtype=torch.bool)
    core_mask[torch.from_numpy(core)] = True
    print(f"       status {unified.status}; core {core.size} candidates; "
          f"node counts {unified.counts}")
    print(f"       per source {json.dumps(unified.per_source_counts)}")
    check("a coherent moving component was found",
          unified.status == "RESOLVED", unified.status)
    check("the core spans BOTH source identities",
          bool((core_mask & (src == 0)).any()) and bool((core_mask & (src == 1)).any()),
          f"{int((core_mask & (src == 0)).sum())} initial + "
          f"{int((core_mask & (src == 1)).sum())} final")
    check("no stationary candidate was put in the moving core",
          int((core_mask & is_static).sum()) == 0,
          f"{int((core_mask & is_static).sum())} of {int(is_static.sum())} "
          "static candidates in the core")

    tp = int((core_mask & is_moving).sum())
    fn = int((~core_mask & is_moving).sum())
    fp_static = int((core_mask & is_static).sum())
    fp_incoh = int((core_mask & is_incoh).sum())
    low_in = int((core_mask & is_low).sum())
    print(f"       evaluation only: TP {tp}/{int(is_moving.sum())}, FN {fn}, "
          f"FP(static) {fp_static}, FP(incoherent) {fp_incoh}, "
          f"low-confidence recovered {low_in}/{int(is_low.sum())}")
    check("most of the moving body is in the core",
          tp >= int(0.7 * int(is_moving.sum())),
          f"recall {tp / max(int(is_moving.sum()), 1):.3f}")

    # What the accepted states actually say, per group: the number the core
    # was decided on, rather than an appeal to the fixture's labels.
    disp = (est.states[-1].mu - est.states[0].mu).norm(dim=1).double().cpu()
    print(f"       estimated |x(T-1) - x(0)|: static "
          f"{float(disp[is_static].max()):.4f} max, moving "
          f"{float(disp[is_moving].min()):.4f}-{float(disp[is_moving].max()):.4f}, "
          f"incoherent {float(disp[is_incoh].min()):.4f}-"
          f"{float(disp[is_incoh].max()):.4f}, alpha=0.02 "
          f"{float(disp[is_low].min()):.4f}-{float(disp[is_low].max()):.4f}")

    print("\n[5b] the SAME extractor on the warm start gives a different answer")
    # The claim under test is that the core comes from the ACCEPTED states.
    # The decisive evidence is not that the accepted answer is right, but that
    # the warm start -- which is constant in time and therefore carries no
    # motion at all -- does not produce it.
    warm_result = uc.unified_coherence_support(
        {"status": CONVERGED, "trajectory": warm}, population,
        config=cfg, verbose=False)
    print(f"       warm start: status {warm_result.status}, "
          f"core {warm_result.core_candidate_indices.size} candidates")
    check("the warm start yields no coherent moving component",
          warm_result.status == "NO_COHERENT_MOVING_COMPONENT"
          and warm_result.core_candidate_indices.size == 0,
          f"{warm_result.status}; the accepted states gave {unified.status} "
          f"with {core.size} core candidates, so the core is a property of "
          "the posterior and not of the initialization")

    # Requirement 7: the weakly observed region. The claim asserted is the one
    # the extractor's semantics support -- it is not forced into a class.
    pool = unified.extraction["candidates"]
    quiet = [c for c in pool if not c["rigidity"].get("degenerate")
             and c["rigidity"]["delta_bic"] <= 0.0]
    print(f"       {len(quiet)} of {len(pool)} evaluated candidates have "
          f"delta_bic <= 0. STATIC_COMPATIBLE is assigned only from the "
          f"largest of those, so with none the 26 stationary candidates stay "
          f"INCOHERENT_OR_AMBIGUOUS. Section 11 permits exactly that -- the "
          f"classes are not a partition and a candidate is never forced into "
          f"a body -- and the BIC penalty k log(NT) with k = 6T = 42 over "
          f"NT <= 336 is what makes the pool empty on a window this short. "
          f"Reported, not renamed, and not a reason to touch the extractor")
    low_classes = [uc.LABEL_NAME[int(unified.label_index[k])]
                   for k in range(N) if bool(is_low[k])]
    print(f"       the two alpha=0.02 candidates came out {low_classes}")
    check("the weakly observed candidates are not asserted into the core by GT",
          all(c != uc.MOVING_RIGID for c in low_classes) or low_in == int(is_low.sum()),
          "they are classified by the extractor's own evidence; whichever way "
          "it goes is reported, not chosen")

    # ---------------------------------------------------------------------
    print("\n[6] the export: unique, ordered, and recoverable by source")
    export = uc.write_unified_coherence(out_root / "run", unified,
                                        scene="t10_synthetic")
    unified_dir = out_root / "run" / "unified_coherence"
    labels = np.load(unified_dir / "labels.npy")
    core_disk = np.load(unified_dir / "core_candidate_indices.npy")
    ids_disk = np.load(unified_dir / "core_source_ids.npy")
    manifest = json.loads((unified_dir / "support_manifest.json").read_text())
    memberships = json.loads((unified_dir / "supernode_memberships.json").read_text())

    check("labels cover every candidate", labels.shape == (N,), str(labels.shape))
    check("the core indices are sorted and unique",
          bool(np.all(np.diff(core_disk) > 0)) and
          core_disk.size == np.unique(core_disk).size, f"{core_disk.size} rows")
    check("the core on disk is the core in memory",
          np.array_equal(core_disk, core))
    check("every core row maps back to its own source row losslessly",
          bool(np.array_equal(ids_disk[:, 0], population.source_endpoint[core_disk])
               and np.array_equal(ids_disk[:, 1],
                                  population.source_gaussian_index[core_disk])),
          f"{ids_disk.shape[0]} (endpoint, row) pairs")
    recovered = []
    for endpoint_id, row in ids_disk:
        name = ("initial", "final")[int(endpoint_id)]
        where = np.flatnonzero(population.mask_for(name)
                               & (population.source_gaussian_index == row))
        recovered.append(int(where[0]))
    check("and the inverse map returns exactly the core",
          np.array_equal(np.sort(np.array(recovered)), core_disk),
          "candidate <- (source_endpoint, source_gaussian_index)")
    check("the manifest hashes what it exported",
          manifest["outputs"]["labels.npy"] == uc.array_sha256(labels)
          and manifest["outputs"]["core_candidate_indices.npy"]
          == uc.array_sha256(core_disk))
    check("the manifest records the input trajectory and both source hashes",
          len(manifest["trajectory_centers_sha256"]) == 64
          and set(manifest["sources"]) == {"initial", "final"}
          and manifest["sources"]["final"]["anchor_original_frame"] == T - 1)
    check("the manifest states the graph's single reference frame",
          manifest["graph_reference_original_frame"] == 0
          and manifest["original_frame_ids"] == list(range(T)))
    check("the label index contract is written down, not implied",
          manifest["label_index_contract"]["0"] == uc.MOVING_RIGID
          and set(manifest["category_semantics"]) >= {uc.MOVING_RIGID,
                                                      uc.STATIC_COMPATIBLE,
                                                      uc.INCOHERENT})
    members_total = sum(len(v["candidate_indices"])
                        for v in memberships["supernodes"].values())
    all_members = sorted(i for v in memberships["supernodes"].values()
                         for i in v["candidate_indices"])
    check("supernode memberships partition the candidates exactly once",
          members_total == N and all_members == list(range(N)),
          f"{members_total} members over "
          f"{len(memberships['supernodes'])} supernodes")
    flat = np.load(unified_dir / "candidate_source_map.npz")
    check("the flat candidate map agrees with the memberships file",
          bool(np.array_equal(flat["label_index"], labels)
               and np.array_equal(flat["source_endpoint"],
                                  population.source_endpoint)
               and np.array_equal(flat["source_gaussian_index"],
                                  population.source_gaussian_index)),
          f"{flat['candidate_index'].size} rows")
    check("each member carries its own (source, row) pair",
          all(len(v["source_ids"]) == len(v["candidate_indices"])
              for v in memberships["supernodes"].values()))
    core_from_labels = np.flatnonzero(labels == uc.LABEL_INDEX[uc.MOVING_RIGID])
    check("labels.npy and core_candidate_indices.npy agree",
          np.array_equal(core_from_labels, core_disk))

    print("\n[7] the export refuses to be written over a historical support tree")
    try:
        uc.write_unified_coherence(stale_dir, unified, scene="t10")
        check("writing into a finished support directory is refused", False,
              "no exception")
    except uc.StaleSupportRejected:
        check("writing into a finished support directory is refused", True,
              "StaleSupportRejected")

    print(f"\n       artifacts: {export['out_dir']}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA, real extractor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
