"""Pixel-mediated final-frame Gaussian matching: required tests 31-37.

Runs against the real modified CUDA rasterizer, not a stand-in.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.pixel_matching import (
    PixelCandidateEvidence,
    PixelContributorMap,
    accumulate_pair_affinity,
    candidate_pixel_evidence,
    covariance_from_scale_rotation,
    current_pixel_attraction_loss,
    normalized_match_confidence,
    pixel_candidate_loss,
    pixel_margin_loss,
    project_gaussians,
    target_coverage_loss,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# A minimal camera with exactly the fields the projection/raster path reads.
# ---------------------------------------------------------------------------
class Cam:
    def __init__(self, eye, target=(0.0, 0.0, 0.0), h=48, w=48, fov=60.0):
        self.image_height, self.image_width = h, w
        self.FoVx = self.FoVy = math.radians(fov)
        eye = torch.tensor(eye, dtype=torch.float32, device=DEV)
        tgt = torch.tensor(target, dtype=torch.float32, device=DEV)
        fwd = torch.nn.functional.normalize(tgt - eye, dim=0)
        up = torch.tensor([0.0, 1.0, 0.0], device=DEV)
        if abs(float(torch.dot(fwd, up))) > 0.99:
            up = torch.tensor([1.0, 0.0, 0.0], device=DEV)
        right = torch.nn.functional.normalize(torch.cross(up, fwd, dim=0), dim=0)
        true_up = torch.cross(fwd, right, dim=0)
        r = torch.stack((right, true_up, fwd), dim=1)      # world->cam basis in columns
        view = torch.eye(4, device=DEV)
        view[:3, :3] = r
        view[3, :3] = -(eye @ r)
        self.world_view_transform = view.contiguous()
        znear, zfar = 0.01, 100.0
        tan = math.tan(self.FoVx * 0.5)
        proj = torch.zeros((4, 4), device=DEV)
        proj[0, 0] = 1.0 / tan
        proj[1, 1] = 1.0 / tan
        proj[2, 2] = zfar / (zfar - znear)
        proj[2, 3] = 1.0
        proj[3, 2] = -(zfar * znear) / (zfar - znear)
        self.full_proj_transform = (view @ proj).contiguous()
        self.camera_center = eye.contiguous()


class Model:
    """Stands in for GaussianModel with only the properties the raster path uses."""

    def __init__(self, xyz, scale=0.05, opacity=0.95):
        n = int(xyz.shape[0])
        self._xyz = xyz.to(DEV).float()
        self._scale = torch.full((n, 3), float(scale), device=DEV)
        self._rot = torch.zeros((n, 4), device=DEV)
        self._rot[:, 0] = 1.0
        self._op = torch.full((n, 1), float(opacity), device=DEV)

    get_xyz = property(lambda s: s._xyz)
    get_scaling = property(lambda s: s._scale)
    get_rotation = property(lambda s: s._rot)
    get_opacity = property(lambda s: s._op)


def _maps(cameras, model, top_l=4):
    from rigid_parts.pixel_matching import render_contributors

    return [render_contributors(c, model, top_l=top_l) for c in cameras]


def _evidence(cand_xyz, cameras, target_maps, final_xyz, *, scale=0.05, bbox=2.0, window=3):
    n, k = cand_xyz.shape[0], cand_xyz.shape[1]
    scales = torch.full((n * k, 3), scale, device=DEV)
    rots = torch.zeros((n * k, 4), device=DEV)
    rots[:, 0] = 1.0
    cov = covariance_from_scale_rotation(scales, rots).reshape(n, k, 3, 3)
    return candidate_pixel_evidence(
        candidate_xyz=cand_xyz, candidate_cov3d=cov,
        opacity=torch.full((n, 1), 0.95, device=DEV),
        cameras=cameras, target_maps=target_maps, final_xyz=final_xyz,
        bbox_diag=bbox, window=window,
    )


def _require_cuda():
    if not HAS_CUDA:
        raise RuntimeError("these tests require the CUDA rasterizer")


# ---------------------------------------------------------------------------
# 31: rasterizer contributor mode (A-D)
# ---------------------------------------------------------------------------
def test_31A_single_contributor():
    _require_cuda()
    cam = Cam((0.0, 0.0, -4.0))
    model = Model(torch.tensor([[0.0, 0.0, 0.0]]), scale=0.2)
    m = _maps([cam], model)[0]
    cy = cx = 24
    assert int(m.ids[cy, cx, 0]) == 0
    assert float(m.weights[cy, cx, 0]) > 0.0
    assert abs(float(m.depths[cy, cx, 0]) - 4.0) < 0.2, float(m.depths[cy, cx, 0])


def test_31B_occluding_pair_ranks_by_responsibility_not_alpha():
    _require_cuda()
    cam = Cam((0.0, 0.0, -4.0))
    # Same ray, both nearly opaque; the front one must dominate.
    model = Model(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]), scale=0.2, opacity=0.99)
    m = _maps([cam], model)[0]
    cy = cx = 24
    slots = {int(m.ids[cy, cx, l]): float(m.weights[cy, cx, l]) for l in range(4)}
    front, back = slots.get(0, 0.0), slots.get(1, 0.0)
    assert front > 0.0
    assert front > 5.0 * max(back, 1e-9), f"front {front:.4f} vs back {back:.4f}"


def test_31C_top_l_is_sorted_by_responsibility():
    _require_cuda()
    cam = Cam((0.0, 0.0, -6.0))
    zs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    model = Model(torch.tensor([[0.0, 0.0, z] for z in zs]), scale=0.2, opacity=0.4)
    m = _maps([cam], model, top_l=4)[0]
    cy = cx = 24
    w = [float(m.weights[cy, cx, l]) for l in range(4)]
    assert all(w[i] >= w[i + 1] for i in range(3)), w
    assert all(int(m.ids[cy, cx, l]) >= 0 for l in range(4))


def test_31D_contributor_mode_does_not_change_the_training_path():
    from diff_gaussian_rasterization import GaussianRasterizationSettings

    assert GaussianRasterizationSettings._fields == (
        "image_height", "image_width", "tanfovx", "tanfovy", "bg", "scale_modifier",
        "viewmatrix", "projmatrix", "sh_degree", "campos", "prefiltered", "debug",
        "antialiasing",
    ), "the differentiable rasteriser interface changed"
    from gaussian_renderer import render

    assert "means3D_override" in render.__code__.co_varnames


# ---------------------------------------------------------------------------
# 32: pixel affinity
# ---------------------------------------------------------------------------
def test_32_affinity_is_high_when_colocated_and_falls_when_moved():
    _require_cuda()
    cams = [Cam((0.0, 0.0, -4.0)), Cam((4.0, 0.0, 0.0)), Cam((0.0, 4.0, 0.01))]
    src = Model(torch.tensor([[0.0, 0.0, 0.0]]), scale=0.15)
    tgt_same = Model(torch.tensor([[0.0, 0.0, 0.0]]), scale=0.15)
    tgt_moved = Model(torch.tensor([[0.9, 0.0, 0.0]]), scale=0.15)

    smaps = _maps(cams, src)
    a_same = accumulate_pair_affinity(smaps, _maps(cams, tgt_same), num_target=1, sigma_z=0.1)[2].sum()
    a_moved = accumulate_pair_affinity(smaps, _maps(cams, tgt_moved), num_target=1, sigma_z=0.1)[2].sum()
    assert float(a_same) > 0.0
    assert float(a_same) > 10.0 * float(a_moved), f"same {float(a_same):.4f} moved {float(a_moved):.4f}"


# ---------------------------------------------------------------------------
# 33: depth disambiguation
# ---------------------------------------------------------------------------
def test_33_depth_disambiguates_two_targets_on_the_same_pixel():
    _require_cuda()
    cam = Cam((0.0, 0.0, -4.0))
    # Target 0 at the candidate depth, target 1 far behind on the same ray.
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], device=DEV)
    tmap = _maps([cam], Model(targets, scale=0.15, opacity=0.35))
    cand = torch.tensor([[[0.0, 0.0, 0.0]]], device=DEV)      # [1,1,3]
    ev = _evidence(cand, [cam], tmap, targets, scale=0.15)

    # Rebuild the per-target affinity directly to compare j_correct vs j_wrong.
    smap = _maps([cam], Model(cand.reshape(1, 3), scale=0.15))
    si, tj, a = accumulate_pair_affinity(smap, tmap, num_target=2, sigma_z=0.05)
    got = {int(t): float(v) for t, v in zip(tj, a)}
    assert got.get(0, 0.0) > got.get(1, 0.0), got
    assert float(ev.confidence[0, 0]) > 0.0


# ---------------------------------------------------------------------------
# 34: label hypothesis -- pixel energy prefers the aligned candidate
# ---------------------------------------------------------------------------
def _hypothesis_case(target_matches_moving: bool):
    cams = [Cam((0.0, 0.0, -4.0)), Cam((4.0, 0.0, 0.0)), Cam((0.0, 4.0, 0.01)), Cam((-4.0, 0.0, 0.0))]
    static_pos = torch.tensor([0.0, 0.0, 0.0], device=DEV)
    moving_pos = torch.tensor([0.8, 0.0, 0.0], device=DEV)
    final = (moving_pos if target_matches_moving else static_pos).reshape(1, 3)
    tmaps = _maps(cams, Model(final, scale=0.15))
    cand = torch.stack((static_pos, moving_pos)).reshape(1, 2, 3)      # [N=1, K+1=2, 3]
    return _evidence(cand, cams, tmaps, final, scale=0.15), cand


def test_34_pixel_energy_prefers_the_hypothesis_that_lands_on_the_final_gaussians():
    _require_cuda()
    ev_moving, _ = _hypothesis_case(target_matches_moving=True)
    assert float(ev_moving.energy[0, 1]) < float(ev_moving.energy[0, 0]), (
        f"E_moving {float(ev_moving.energy[0,1]):.4f} !< E_static {float(ev_moving.energy[0,0]):.4f}"
    )
    ev_static, _ = _hypothesis_case(target_matches_moving=False)
    assert float(ev_static.energy[0, 0]) < float(ev_static.energy[0, 1]), (
        f"E_static {float(ev_static.energy[0,0]):.4f} !< E_moving {float(ev_static.energy[0,1]):.4f}"
    )


def test_34b_optimizing_only_logits_moves_probability_to_the_preferred_class():
    _require_cuda()
    ev, _ = _hypothesis_case(target_matches_moving=True)
    logits = torch.zeros((1, 2), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.2)
    before = torch.softmax(logits, dim=-1)[0, 1].item()
    for _ in range(60):
        opt.zero_grad()
        p = torch.softmax(logits, dim=-1)
        pixel_candidate_loss(p, ev).backward()
        opt.step()
    after = torch.softmax(logits, dim=-1)[0, 1].item()
    assert after > before, f"p_moving did not rise ({before:.4f} -> {after:.4f})"
    assert after > 0.5, f"p_moving only reached {after:.4f}"


# ---------------------------------------------------------------------------
# 35: multiview disambiguation
# ---------------------------------------------------------------------------
def test_35_multiview_resolves_what_a_single_view_cannot():
    _require_cuda()
    # Along the -z camera's ray, static and moving differ only in depth, so that
    # single view is ambiguous; the side camera separates them in image space.
    ambiguous = Cam((0.0, 0.0, -5.0))
    informative = Cam((5.0, 0.0, 0.0))
    static_pos = torch.tensor([0.0, 0.0, 0.0], device=DEV)
    moving_pos = torch.tensor([0.0, 0.0, 0.9], device=DEV)
    final = moving_pos.reshape(1, 3)
    cand = torch.stack((static_pos, moving_pos)).reshape(1, 2, 3)

    single = _evidence(cand, [ambiguous], _maps([ambiguous], Model(final, scale=0.2)), final, scale=0.2)
    both = _evidence(cand, [ambiguous, informative],
                     _maps([ambiguous, informative], Model(final, scale=0.2)), final, scale=0.2)

    gap_single = float(single.energy[0, 0] - single.energy[0, 1])
    gap_both = float(both.energy[0, 0] - both.energy[0, 1])
    assert gap_both > 0.0, "multiview failed to prefer the correct candidate"
    assert gap_both > gap_single, f"multiview did not sharpen the margin ({gap_single:.4f} -> {gap_both:.4f})"


# ---------------------------------------------------------------------------
# 36: low confidence must not mean static
# ---------------------------------------------------------------------------
def test_36_unsupported_candidate_gets_near_zero_confidence_and_no_label_pressure():
    _require_cuda()
    cams = [Cam((0.0, 0.0, -4.0)), Cam((4.0, 0.0, 0.0))]
    final = torch.tensor([[0.0, 0.0, 0.0]], device=DEV)
    tmaps = _maps(cams, Model(final, scale=0.15))
    # A candidate far outside any final splat, under BOTH hypotheses.
    cand = torch.tensor([[[8.0, 8.0, 8.0], [8.5, 8.0, 8.0]]], device=DEV)
    ev = _evidence(cand, cams, tmaps, final, scale=0.15)

    assert float(ev.confidence.max()) < 1e-6, f"confidence {float(ev.confidence.max()):.3e}"
    logits = torch.zeros((1, 2), device=DEV, requires_grad=True)
    p = torch.softmax(logits, dim=-1)
    pixel_candidate_loss(p, ev).backward()
    assert float(logits.grad.abs().max()) < 1e-6, "unsupported candidate still pushed the label"
    # And it must NOT be nudged toward static.
    assert float(ev.confidence[0, 0]) == float(ev.confidence[0, 1]) == 0.0


# ---------------------------------------------------------------------------
# 37: no ground truth may be read
# ---------------------------------------------------------------------------
def test_37_module_never_reads_ground_truth():
    import rigid_parts.pixel_matching as pm

    src = inspect.getsource(pm)
    for forbidden in ("points3D_parts", "gt_part", "ground_truth", "part_labels_gt", "_parts.ply"):
        assert forbidden not in src, f"pixel matching references ground truth: {forbidden}"
    # Its public entry points take only models, cameras and frozen trajectories.
    params = set(inspect.signature(candidate_pixel_evidence).parameters)
    assert not any("gt" in p or "label" in p for p in params), params


# ---------------------------------------------------------------------------
# 12 / 13 / 28: normalisation, attraction, coverage
# ---------------------------------------------------------------------------
def test_12_normalized_match_confidence_shapes_and_symmetry():
    si = torch.tensor([0, 0, 1], device=DEV)
    tj = torch.tensor([0, 1, 1], device=DEV)
    w = torch.tensor([3.0, 1.0, 4.0], device=DEV)
    out = normalized_match_confidence(si, tj, w, num_source=2, num_target=2)
    assert torch.allclose(out["p_j_given_i"][:2].sum(), torch.tensor(1.0, device=DEV), atol=1e-6)
    # A target claimed by many sources is discounted by the symmetric form.
    assert float(out["symmetric"][2]) <= float(out["p_j_given_i"][2])


def test_13_current_attraction_pulls_toward_the_weighted_barycentre():
    cur = torch.tensor([[0.0, 0.0, 0.0]], device=DEV)
    final = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device=DEV)
    si = torch.tensor([0, 0], device=DEV)
    tj = torch.tensor([0, 1], device=DEV)
    w = torch.tensor([3.0, 1.0], device=DEV)
    loss, mass = current_pixel_attraction_loss(cur, final, si, tj, w, bbox_diag=2.0)
    # barycentre = (3*1 + 1*3)/4 = 1.5 ; residual = 2.25 ; /(4 * 4) * 4
    assert abs(float(loss) - (2.25 / 4.0)) < 1e-6, float(loss)
    assert abs(float(mass[0]) - 4.0) < 1e-6


def test_28_target_coverage_is_scale_free_and_weak():
    p = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=DEV)
    q_src = torch.tensor([0, 2], device=DEV)      # flattened (i,k) indices
    q_tgt = torch.tensor([0, 1], device=DEV)
    q_w = torch.tensor([1.0, 1.0], device=DEV)
    balanced = target_coverage_loss(p, q_src, q_tgt, q_w, torch.tensor([1.0, 1.0], device=DEV), num_target=2)
    skewed = target_coverage_loss(p, q_src, q_tgt, q_w, torch.tensor([10.0, 0.1], device=DEV), num_target=2)
    assert float(balanced) < float(skewed)
    # Scaling the support uniformly must not change the loss (it is normalised).
    same = target_coverage_loss(p, q_src, q_tgt, q_w, torch.tensor([100.0, 100.0], device=DEV), num_target=2)
    assert abs(float(balanced) - float(same)) < 1e-6


# ---------------------------------------------------------------------------
# 23: margin neutrality on missing evidence
# ---------------------------------------------------------------------------
def test_23_missing_evidence_is_neutral_in_the_margin():
    e = torch.tensor([[0.1, 5.0]], device=DEV)
    confident = PixelCandidateEvidence(
        energy=e, confidence=torch.ones_like(e), overlap=torch.ones_like(e),
        xyz_energy=e, depth_energy=e, num_views=torch.ones_like(e))
    unknown = PixelCandidateEvidence(
        energy=e, confidence=torch.zeros_like(e), overlap=torch.zeros_like(e),
        xyz_energy=e, depth_energy=e, num_views=torch.zeros_like(e))
    p = torch.full((1, 2), 0.5, device=DEV)
    assert float(pixel_margin_loss(p, unknown)) < float(pixel_margin_loss(p, confident)), (
        "missing pixel evidence was treated as strong evidence"
    )


def test_15_projection_matches_the_rasteriser_footprint():
    _require_cuda()
    cam = Cam((0.0, 0.0, -4.0))
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.2, 0.0]], device=DEV)
    model = Model(xyz, scale=0.1)
    proj = project_gaussians(cam, xyz, covariance_from_scale_rotation(model.get_scaling, model.get_rotation))
    m = _maps([cam], model)[0]
    for i in range(2):
        u, v = proj["uv"][i].round().long().tolist()
        assert 0 <= u < cam.image_width and 0 <= v < cam.image_height
        # The rasteriser must list this Gaussian among the contributors there.
        assert i in {int(x) for x in m.ids[v, u]}, (
            f"projected pixel ({u},{v}) does not contain gaussian {i}: {m.ids[v,u].tolist()}"
        )
        assert abs(float(proj["depth"][i]) - float(m.depths[v, u, 0])) < 0.3


def test_28_sparse_triples_feed_the_real_target_coverage():
    """section 28 must use the actual Q_ikj, not a placeholder."""
    _require_cuda()
    cams = [Cam((0.0, 0.0, -4.0)), Cam((4.0, 0.0, 0.0))]
    final = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], device=DEV)
    tmaps = _maps(cams, Model(final, scale=0.15))
    cand = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], device=DEV)
    n, k = 1, 2
    scales = torch.full((n * k, 3), 0.15, device=DEV)
    rots = torch.zeros((n * k, 4), device=DEV); rots[:, 0] = 1.0
    cov = covariance_from_scale_rotation(scales, rots).reshape(n, k, 3, 3)
    ev, sparse = candidate_pixel_evidence(
        candidate_xyz=cand, candidate_cov3d=cov,
        opacity=torch.full((n, 1), 0.95, device=DEV),
        cameras=cams, target_maps=tmaps, final_xyz=final,
        bbox_diag=2.0, window=3, collect_sparse=True,
    )
    for key in ("source", "klass", "target", "weight"):
        assert key in sparse, key
    assert sparse["source"].numel() > 0, "no sparse (i,k,j) triples were emitted"
    assert int(sparse["klass"].max()) < k and int(sparse["target"].max()) < final.shape[0]

    p = torch.full((n, k), 0.5, device=DEV)
    loss = target_coverage_loss(
        p, sparse["source"], sparse["target"], sparse["weight"],
        torch.ones(final.shape[0], device=DEV), num_target=int(final.shape[0]),
        q_class=sparse["klass"],
    )
    assert torch.isfinite(loss) and float(loss) >= 0.0


def test_8_current_sparse_matches_are_produced_by_the_pipeline():
    """sections 8-13 + 40: the pipeline must build and save current matches."""
    import rigid_parts.nested_pipeline as np_mod

    src = inspect.getsource(np_mod.run_nested_part_embedding_killing_discovery)
    for needed in ("render_contributors", "accumulate_pair_affinity",
                   "normalized_match_confidence", "current_pixel_attraction_loss",
                   "current_sparse_matches", "target_coverage_loss"):
        assert needed in src, f"section 8-13/28/40 not wired: {needed}"
    assert "_summary.json" in src, "section 40 summary.json not written"



if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [
        (name, fn)
        for name, fn in sorted(vars(module).items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS pixel_mediated_matching ({len(tests)} tests)")
