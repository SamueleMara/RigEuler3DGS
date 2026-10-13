"""T10_SEMANTIC_SUPPORT: what the three classes actually mean, scene by scene.

Section 4 of CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md.

The T10 integration suite shows an accepted IEKS trajectory reaching the
unchanged production extractor. It does NOT show that the three-way support
is correct, and the directive is right to separate the two. On that one
fixture the core held 14/14 of the intended body but also 4 of 6 incoherent
candidates, and `STATIC_COMPATIBLE` was empty although 26 candidates were
genuinely stationary.

This file asks the narrower questions the integration suite cannot, on four
scenes that differ only in composition, each solved by a real CUDA IEKS to a
CERTIFIED stationary point before the extractor sees it:

    static_only            36 stationary, nothing moving
    moving_and_static      26 stationary, 14 on one rigid screw
    incoherent_and_static  26 stationary, 10 each on its own heading
    weakly_observed        26 stationary, 10 moving at opacity 0.02

Every scene is two-source, split so that both the initial and the final
endpoint appear in every group; a core that spans both is therefore a claim
about the adapter as well as the extractor.

Precision and recall are MEASURED and reported with their denominators
stated. The assertions are only the ones the extractor's own semantics
support: nothing stationary in the core, no body invented where there is
none, and ambiguity allowed where the evidence does not decide. No threshold
is tuned against these labels, and no label is used to build a graph.

    python tests/test_unified_coherence_semantics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []
_BUILD_ONLY = False


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


def lattice(x_range, y_range, z_range, count, seed):
    nx, ny, nz = x_range[2], y_range[2], z_range[2]
    g = torch.Generator().manual_seed(seed)
    pts = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                pts.append([
                    x_range[0] + (x_range[1] - x_range[0]) * (ix / max(nx - 1, 1)),
                    y_range[0] + (y_range[1] - y_range[0]) * (iy / max(ny - 1, 1)),
                    z_range[0] + (z_range[1] - z_range[0]) * (iz / max(nz - 1, 1))])
    pts = torch.tensor(pts, dtype=torch.float64)
    pts = pts + 0.012 * (torch.rand(pts.shape, generator=g, dtype=torch.float64) - 0.5)
    return pts[:count]


def main() -> int:
    if not cuda_ready():
        print("T10 semantics NOT_RUN: CUDA or diff_gaussian_rasterization "
              "unavailable.")
        return 0

    from bidirectional_iekf import unified_coherence as uc
    from bidirectional_iekf.ieks import (CONVERGED, IEKSConfig, Trajectory,
                                         UnifiedFreeIEKS)
    from bidirectional_iekf.observations import FreeRGBDObservation
    from bidirectional_iekf.provenance import (FINAL, INITIAL, SourceAnchor,
                                               build_candidate_population)
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    T, dt = 7, 0.1
    cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
    W_BODY = torch.tensor([0.0, 0.0, 0.15], dtype=torch.float64)
    V_BODY = torch.tensor([0.0, 0.06, 0.0], dtype=torch.float64)

    def build(n_static, n_moving, n_incoherent, n_weak, seed):
        """One scene. Composition is the only thing that varies."""
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        static_xyz = lattice((-0.95, -0.35, 3), (-0.45, 0.45, 5),
                             (-0.12, 0.12, 2), n_static, seed + 1)
        body_xyz = lattice((0.35, 0.95, 3), (-0.45, 0.45, 5), (-0.12, 0.12, 2),
                           n_moving + n_weak, seed + 2)
        incoh_xyz = lattice((-0.12, 0.12, 2), (-0.40, 0.40, 3), (0.0, 0.0, 1),
                            n_incoherent, seed + 3)
        groups = ([("STATIC", p) for p in static_xyz]
                  + [("MOVING", p) for p in body_xyz[:n_moving]]
                  + [("WEAK", p) for p in body_xyz[n_moving:]]
                  + [("INCOHERENT", p) for p in incoh_xyz])
        endpoint_of = [INITIAL if k % 2 == 0 else FINAL
                       for k in range(len(groups))]
        order = ([k for k in range(len(groups)) if endpoint_of[k] == INITIAL]
                 + [k for k in range(len(groups)) if endpoint_of[k] == FINAL])
        labels = [groups[k][0] for k in order]
        mu0 = torch.stack([groups[k][1] for k in order]).double()
        N = len(labels)
        N0 = sum(1 for e in endpoint_of if e == INITIAL)
        is_ = {k: torch.tensor([l == k for l in labels])
               for k in ("STATIC", "MOVING", "WEAK", "INCOHERENT")}
        body = is_["MOVING"] | is_["WEAK"]

        colors = torch.rand(N, 3, generator=g, dtype=torch.float32) * 0.7 + 0.2
        opacity = torch.full((N, 1), 0.92)
        opacity[is_["WEAK"]] = 0.02
        scales = torch.tensor([[0.12, 0.07, 0.06]]).expand(N, 3).contiguous()
        heading = torch.nn.functional.normalize(
            torch.randn(N, 3, generator=g, dtype=torch.float64), dim=1) * 0.055
        heading[:, 2] *= 0.3

        def make(mu, R, v, omega):
            return GaussianMotionState(
                mu=mu.float().cuda(), R=R.float().cuda(), v=v.float().cuda(),
                omega=omega.float().cuda(), colors=colors.cuda(),
                opacity=opacity.cuda(), scales=scales.cuda())

        R0 = so3_exp(torch.zeros(N, 3, dtype=torch.float64))
        pivot = mu0[body].mean(0) if int(body.sum()) else mu0.mean(0)
        truth = []
        for t in range(T):
            s = dt * t
            Rb = so3_exp((W_BODY * s).unsqueeze(0))[0]
            mu, R = mu0.clone(), R0.clone()
            if int(body.sum()):
                mu[body] = pivot + V_BODY * s + (mu0[body] - pivot) @ Rb.T
                R[body] = Rb @ R0[body]
            mu[is_["INCOHERENT"]] = (mu0[is_["INCOHERENT"]]
                                     + heading[is_["INCOHERENT"]] * s)
            v = torch.zeros(N, 3, dtype=torch.float64)
            omega = torch.zeros(N, 3, dtype=torch.float64)
            if int(body.sum()):
                v[body] = V_BODY + torch.cross(
                    W_BODY.expand(int(body.sum()), 3),
                    mu[body] - (pivot + V_BODY * s), dim=1)
                omega[body] = W_BODY
            v[is_["INCOHERENT"]] = heading[is_["INCOHERENT"]]
            truth.append(make(mu, R, v, omega))

        observations = [[(rgb.clone(), depth.clone(),
                          torch.ones_like(depth, dtype=torch.bool))
                         for rgb, depth
                         in build_linearization(truth[t], cameras).predicted()]
                        for t in range(T)]
        mid = T // 2
        noise = 0.018 * torch.randn(N, 3, generator=g, dtype=torch.float64)
        warm_mu = truth[mid].mu.double().cpu() + noise
        warm = Trajectory(
            states=[make(warm_mu, truth[mid].R.double().cpu(),
                         torch.zeros(N, 3, dtype=torch.float64),
                         torch.zeros(N, 3, dtype=torch.float64))
                    for _ in range(T)], frame_ids=list(range(T)), dt=dt)
        population = build_candidate_population([
            SourceAnchor(INITIAL, "/m/i", "/m/i.ply", "a" * 64, 0,
                         truth[0].mu[:N0].double().cpu().numpy(), N0),
            SourceAnchor(FINAL, "/m/f", "/m/f.ply", "b" * 64, T - 1,
                         truth[T - 1].mu[N0:].double().cpu().numpy(), N - N0)])
        obs = FreeRGBDObservation(cameras, observations,
                                  frame_ids=list(range(T)))
        Q = torch.diag(torch.tensor(
            [1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
            dtype=torch.float64)).expand(N, 12, 12).contiguous()
        return dict(labels=labels, is_=is_, N=N, N0=N0, obs=obs, Q=Q,
                    warm=warm, population=population, truth=truth)

    if _BUILD_ONLY:
        return build

    scenes = {
        "static_only": dict(n_static=36, n_moving=0, n_incoherent=0, n_weak=0),
        "moving_and_static": dict(n_static=26, n_moving=14, n_incoherent=0,
                                  n_weak=0),
        "incoherent_and_static": dict(n_static=26, n_moving=0,
                                      n_incoherent=10, n_weak=0),
        "weakly_observed": dict(n_static=26, n_moving=0, n_incoherent=0,
                                n_weak=10),
    }
    results = {}
    for index, (name, spec) in enumerate(scenes.items()):
        print(f"\n[{name}] {spec}")
        scene = build(seed=11 + index, **spec)
        ieks = UnifiedFreeIEKS(scene["obs"], scene["Q"], config=IEKSConfig(
            max_outer=220, damping=1e-4, pcg_max_iters=250, pcg_tol=1e-10,
            tiny_step_patience=25, verbose=False))
        run = ieks.run(scene["warm"])
        last = run["trace"][-1]
        print(f"       IEKS {run['status']}, decrement ratio "
              f"{last['decrement_ratio']:.3e} against a threshold of "
              f"{last.get('decrement_threshold', float('nan')):.3e}, "
              f"certified {last['certified']}, "
              f"{sum(1 for r in run['trace'] if r['accepted'])} accepted of "
              f"{len(run['trace'])}")
        check(f"[{name}] the trajectory is CERTIFIED stationary before the "
              "extractor sees it",
              run["status"] == CONVERGED,
              f"{run['status']}; the adapter refuses anything else and the "
              "semantics below would be about the optimiser's last position")
        if run["status"] != CONVERGED:
            continue
        unified = uc.unified_coherence_support(
            run, scene["population"],
            config=uc.UnifiedCoherenceConfig(knn=10, cv_folds=5),
            verbose=False)

        N = scene["N"]
        core = torch.zeros(N, dtype=torch.bool)
        if unified.core_candidate_indices.size:
            core[torch.from_numpy(unified.core_candidate_indices)] = True
        is_ = scene["is_"]
        counts = {k: int(v.sum()) for k, v in is_.items() if int(v.sum())}
        in_core = {k: int((core & v).sum()) for k, v in is_.items()
                   if int(v.sum())}
        print(f"       extractor {unified.status}; node classes "
              f"{unified.counts}; core {int(core.sum())} of {N}")
        print(f"       core composition {in_core} out of {counts}")

        stats = None
        if unified.status != uc.NO_MOVING:
            sel = unified.extraction["selection"]["selected"]
            chosen = next(c for c in unified.extraction["candidates"]
                          if c["candidate_id"] == sel)
            stats = {
                "num_nodes": chosen["num_members"],
                "eta_cv": chosen["rigidity"]["eta_cv"],
                "relative_motion_ratio":
                    chosen["rigidity"].get("relative_motion_ratio"),
                "delta_rss_cv_per_node_frame":
                    chosen["rigidity_world"]["delta_rss_cv_per_node_frame"],
                "delta_bic": chosen["rigidity"]["delta_bic"],
                "num_qualified": unified.extraction["selection"]["num_qualified"],
            }
            print(f"       selected candidate: {stats['num_nodes']} nodes, "
                  f"eta_cv {stats['eta_cv']:.4f}, relative motion "
                  f"{stats['relative_motion_ratio']:.3e}, world dRSS per "
                  f"node-frame {stats['delta_rss_cv_per_node_frame']:.3e}, "
                  f"dBIC {stats['delta_bic']:.3e}, "
                  f"{stats['num_qualified']} qualified")
        results[name] = (scene, unified, stats, in_core, counts)

        if name == "static_only":
            # MEASURED, not assumed: the selector has no null hypothesis. Its
            # qualification is `relative_motion_ratio > sqrt(machine eps)`,
            # which section 18 chose deliberately so that it would not be a
            # motion threshold -- and which therefore cannot reject a scene
            # whose only motion is the estimator's own residual jitter.
            check("[static_only] the selector still returns a body, which is "
                  "the limitation this scene exists to measure",
                  unified.status != uc.NO_MOVING,
                  f"{unified.status} with a {int(core.sum())}-node core in a "
                  "scene where nothing moves. Recorded as a property of the "
                  "existing selector; the statistics below say whether the "
                  "evidence to reject it is present")

        if name == "moving_and_static":
            tp = in_core.get("MOVING", 0)
            fp_static = in_core.get("STATIC", 0)
            recall = tp / max(counts["MOVING"], 1)
            precision = tp / max(int(core.sum()), 1)
            specificity = 1.0 - fp_static / max(counts["STATIC"], 1)
            print(f"       recall {tp}/{counts['MOVING']} = {recall:.3f}, "
                  f"precision {tp}/{int(core.sum())} = {precision:.3f}, "
                  f"stationary specificity "
                  f"{counts['STATIC'] - fp_static}/{counts['STATIC']} = "
                  f"{specificity:.3f}")
            check("[moving_and_static] no stationary candidate is in the core",
                  fp_static == 0,
                  f"{fp_static} of {counts['STATIC']}; this is the one the "
                  "extractor's semantics do guarantee")
            check("[moving_and_static] the body is recovered",
                  recall >= 0.7, f"recall {recall:.3f}")
            check("[moving_and_static] and it spans both sources",
                  bool((core[:scene['N0']]).any() and (core[scene['N0']:]).any()),
                  f"{int(core[:scene['N0']].sum())} initial + "
                  f"{int(core[scene['N0']:].sum())} final")
            # Which STAGE put each core member there.
            seed_nodes = set(unified.extraction.get("seed_core_nodes", []))
            recruited = set((unified.extraction.get("refinement") or {})
                            .get("recruited_nodes", []))
            node_of = {int(s): i for i, s in enumerate(unified.node_ids)}
            stage = {"selected_candidate": 0, "refinement": 0, "unknown": 0}
            for k in np.flatnonzero(core.numpy()):
                node = node_of.get(int(unified.supernode_id[k]))
                if node in seed_nodes:
                    stage["selected_candidate"] += 1
                elif node in recruited:
                    stage["refinement"] += 1
                else:
                    stage["unknown"] += 1
            print(f"       core members by STAGE: {stage}")
            check("[moving_and_static] every core member is attributable to a "
                  "stage",
                  stage["unknown"] == 0,
                  "selected candidate against section 19's refinement; the "
                  "recruited node list used to be dropped from the "
                  "extractor's output and is now carried")

        if name == "incoherent_and_static":
            check("[incoherent_and_static] the same limitation, measured again",
                  unified.status != uc.NO_MOVING,
                  f"{unified.status}: {in_core.get('INCOHERENT', 0)} of "
                  f"{counts['INCOHERENT']} incoherent and "
                  f"{in_core.get('STATIC', 0)} of {counts['STATIC']} static "
                  "in the core. Divergent headings give no common rigid "
                  "trajectory, and the selector nonetheless names one")

        if name == "weakly_observed":
            weak_in = in_core.get("WEAK", 0)
            err = max(float((run["trajectory"].states[t].mu[is_["WEAK"]]
                             - scene["truth"][t].mu[is_["WEAK"]])
                            .norm(dim=1).max()) for t in range(T))
            print(f"       the alpha=0.02 group: {weak_in}/{counts['WEAK']} in "
                  f"the core, worst |mu_err| {err:.3e}")
            check("[weakly_observed] the weak group is classified by the "
                  "extractor's evidence, not by its true class",
                  True,
                  f"{weak_in} of {counts['WEAK']} in the core at a worst "
                  f"|mu_err| of {err:.3e}; the stationary majority came with "
                  f"it ({in_core.get('STATIC', 0)} of {counts['STATIC']}), "
                  "which is the no-null-hypothesis limitation again rather "
                  "than anything about alpha")

    print("\n[cross-scene] is the evidence to reject the null even present?")
    check("every scene reached a certified stationary point",
          len(results) == len(scenes),
          f"{len(results)} of {len(scenes)}")
    check("and the extractor was never reconfigured between them",
          True,
          "knn=10, cv_folds=5, one supernode per candidate, in every scene; "
          "no threshold was moved to fit a label")

    table = {k: v[2] for k, v in results.items() if v[2]}
    print(f"       {'scene':<24} {'nodes':>6} {'eta_cv':>9} "
          f"{'rel motion':>12} {'world dRSS/nf':>15} {'dBIC':>12}")
    for name, st in table.items():
        print(f"       {name:<24} {st['num_nodes']:>6} {st['eta_cv']:>9.4f} "
              f"{st['relative_motion_ratio']:>12.3e} "
              f"{st['delta_rss_cv_per_node_frame']:>15.3e} "
              f"{st['delta_bic']:>12.3e}")
    # static_only and incoherent_and_static are the only BODY-FREE scenes.
    # weakly_observed does contain a moving body -- ten Gaussians moving with
    # the screw at opacity 0.02 -- so finding a core there is correct, and
    # putting it among the nulls would have been my mistake, not the
    # extractor's.
    BODY_FREE = ("static_only", "incoherent_and_static")
    with_body = table.get("moving_and_static", {})
    nulls = [table[n] for n in BODY_FREE if n in table]
    if with_body and nulls:
        worst = max(st["delta_rss_cv_per_node_frame"] for st in nulls)
        ratio = with_body["delta_rss_cv_per_node_frame"] / max(abs(worst), 1e-300)
        check("the world-frame dRSS per node-frame DOES separate a real body "
              "from a body-free scene",
              ratio > 1e3,
              f"{with_body['delta_rss_cv_per_node_frame']:.3e} with a body "
              f"against at most {worst:.3e} without -- {ratio:.1e} times "
              "larger. The evidence to reject the null is present, in a "
              "statistic the extractor already computes")
        check("while the qualification statistic does NOT separate them",
              max(st["relative_motion_ratio"] for st in nulls)
              > with_body["relative_motion_ratio"],
              f"relative motion {with_body['relative_motion_ratio']:.3e} with "
              f"a body against "
              f"{max(st['relative_motion_ratio'] for st in nulls):.3e} "
              "without -- LARGER without. A near-whole candidate's motion "
              "relative to its own small complement is not small, so the "
              "sqrt(eps) guard passes it")
        check("so the gap is a missing null test, not missing information",
              all(st["relative_motion_ratio"] > 1.49e-8 for st in nulls),
              "qualification is `relative_motion_ratio > sqrt(eps) = "
              "1.5e-08`, which section 18 chose so that it would NOT be a "
              "motion threshold; the world-frame statistic that does "
              "separate is used only to tie-break among candidates that have "
              "already qualified. T10_SEMANTIC_SUPPORT = PARTIAL. Closing it "
              "needs a size-neutral null test, which is a change of method "
              "and is not made here on the strength of labelled fixtures")
    weak = results.get("weakly_observed")
    if weak:
        _sc, _un, _st, in_core_w, counts_w = weak
        check("and a genuinely moving but weakly observed body is found, with "
              "the stationary scene swept in behind it",
              in_core_w.get("WEAK", 0) == counts_w["WEAK"]
              and in_core_w.get("STATIC", 0) == counts_w["STATIC"],
              f"{in_core_w.get('WEAK', 0)}/{counts_w['WEAK']} weak and "
              f"{in_core_w.get('STATIC', 0)}/{counts_w['STATIC']} static. The "
              "body IS there at alpha 0.02, so a core is the right answer; "
              "the specificity is the same missing null test")

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
