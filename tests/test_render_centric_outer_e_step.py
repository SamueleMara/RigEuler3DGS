"""Render-centric outer E-step: §36-§42 required tests, plus §29 frame alignment.

The claim under test: with motion frozen, the VIDEO residual alone can move a
Gaussian's label, while the forward physical scene stays hard.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts import nested_pipeline
from rigid_parts.embeddings import PartEmbeddingModel
from rigid_parts.neighborhood import build_neighbor_graph
from rigid_parts.outer_render import (
    align_quaternion_hemisphere,
    candidate_state_at_frame,
    compose_straight_through_scene,
    compute_detached_3d_gradient_cap,
    kinetic_label_loss,
    motion_aware_label_graph_loss,
    outer_render_membership_loss,
    straight_through_membership,
)

DT = torch.float64


# ---------------------------------------------------------------------------
# A synthetic "video": each camera observes the true positions of the Gaussians
# it can see. The render loss is therefore a genuine image-space residual whose
# only free variable is the membership.
# ---------------------------------------------------------------------------
def _make_render_fn(true_xyz, visible_mask=None):
    def render_fn(camera, gaussians, pipe, background, dataset, args, positions, rotations):
        mask = camera["visible"] if visible_mask is None else visible_mask
        w = mask.to(positions.dtype)
        residual = ((positions - true_xyz.to(positions)) * w[:, None]).square().sum(-1)
        rgb = residual.sum() / w.sum().clamp_min(1.0)
        zero = rgb.new_tensor(0.0)
        return rgb, rgb, zero, zero, False, 1.0

    return render_fn


def _identity_quats(n):
    q = torch.zeros((n, 4), dtype=DT)
    q[:, 0] = 1.0
    return q


def _frozen_trajectory(frames, shift, n_parts=2):
    """T_0 = I for all t; T_1(t) translates progressively. Returns [F,K+1,...]."""
    rot = torch.eye(3, dtype=DT)[None, None].repeat(frames, n_parts, 1, 1)
    trans = torch.zeros((frames, n_parts, 3), dtype=DT)
    for f in range(frames):
        trans[f, 1] = shift * (f / max(frames - 1, 1))
    quat = torch.zeros((frames, n_parts, 4), dtype=DT)
    quat[..., 0] = 1.0
    return rot, trans, quat


def _setup(n=60, frames=4, n_cameras=3, moving_fraction=0.5):
    torch.manual_seed(0)
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    rot0 = _identity_quats(n)
    shift = torch.tensor([1.5, 0.0, 0.0], dtype=DT)
    truly_moving = torch.zeros(n, dtype=torch.bool)
    truly_moving[: int(n * moving_fraction)] = True
    R, T, Q = _frozen_trajectory(frames, shift)
    cameras_by_frame = [
        [{"visible": torch.ones(n, dtype=torch.bool)} for _ in range(n_cameras)]
        for _ in range(frames)
    ]
    return xyz0, rot0, R, T, Q, truly_moving, shift, cameras_by_frame


def _true_positions_at(xyz0, truly_moving, T, frame):
    return torch.where(truly_moving[:, None], xyz0 + T[frame, 1], xyz0)


def _run_outer(model, xyz0, rot0, R, T, Q, cameras_by_frame, truly_moving, *, epochs, lr=0.5, tau=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        p_soft = model.probabilities(tau).to(DT)
        p_st = straight_through_membership(p_soft)
        total = p_st.sum() * 0.0
        for f in range(len(cameras_by_frame)):
            true_xyz = _true_positions_at(xyz0, truly_moving, T, f)
            out = outer_render_membership_loss(
                xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
                p_st=p_st, cameras_by_frame=cameras_by_frame, gaussians=None, pipe=None,
                background=None, dataset=None, args=None, frames=[f],
                render_fn=_make_render_fn(true_xyz),
            )
            total = total + out["loss"]
        total.backward()
        opt.step()
    return model.probabilities(tau).detach()


# ---------------------------------------------------------------------------
# §38 the critical straight-through test
# ---------------------------------------------------------------------------
def test_38_forward_state_is_hard_while_logits_still_receive_gradient():
    n = 4
    xyz0 = torch.rand((n, 3), dtype=DT)
    shift = torch.tensor([5.0, 0.0, 0.0], dtype=DT)
    cand_xyz = torch.stack((xyz0, xyz0 + shift), dim=1)
    cand_quat = torch.stack((_identity_quats(n), _identity_quats(n)), dim=1)

    for probs, expect_moving in ([0.51, 0.49], False), ([0.49, 0.51], True):
        logits = torch.log(torch.tensor([probs] * n, dtype=DT)).requires_grad_(True)
        p_soft = torch.softmax(logits, dim=-1)
        p_st = straight_through_membership(p_soft)
        xyz, quat = compose_straight_through_scene(cand_xyz, cand_quat, p_st)

        expected = xyz0 + shift if expect_moving else xyz0
        assert torch.allclose(xyz, expected, atol=1e-14), (
            f"p={probs} produced a blended coordinate, not a hard state"
        )
        # There must never be a 49/51 blend.
        blend = 0.49 * xyz0 + 0.51 * (xyz0 + shift)
        assert not torch.allclose(xyz, blend, atol=1e-6)

        # ...and the render loss still reaches the logits.
        (xyz.sum() + quat.sum()).backward()
        assert logits.grad is not None and float(logits.grad.abs().max()) > 0.0, (
            "straight-through gave no gradient to the logits"
        )


def test_38b_straight_through_forward_equals_onehot_backward_equals_softmax():
    torch.manual_seed(1)
    logits = torch.randn((20, 3), dtype=DT, requires_grad=True)
    p_soft = torch.softmax(logits, dim=-1)
    p_st = straight_through_membership(p_soft)

    hard = torch.nn.functional.one_hot(p_soft.argmax(-1), 3).to(DT)
    assert torch.allclose(p_st.detach(), hard, atol=1e-15), "forward is not one-hot"

    target = torch.randn((20, 3), dtype=DT)
    (p_st * target).sum().backward()
    g_st = logits.grad.detach().clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    (torch.softmax(logits2, dim=-1) * target).sum().backward()
    assert torch.allclose(g_st, logits2.grad, atol=1e-14), "backward is not the softmax gradient"


def test_5_quaternion_hemispheres_are_aligned_before_combination():
    n = 5
    q = torch.zeros((n, 2, 4), dtype=DT)
    q[:, 0, 0] = 1.0
    q[:, 1, 0] = -1.0            # same rotation, opposite sign
    aligned = align_quaternion_hemisphere(q)
    assert torch.allclose(aligned[:, 1], q[:, 0], atol=1e-15), "hemisphere not aligned"
    # A hard selection of either class must still give a unit quaternion.
    p_st = torch.zeros((n, 2), dtype=DT)
    p_st[:, 1] = 1.0
    _, quat = compose_straight_through_scene(torch.zeros((n, 2, 3), dtype=DT), q, p_st)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(n, dtype=DT), atol=1e-12)


# ---------------------------------------------------------------------------
# §36 / §37 render alone corrects a wrong label
# ---------------------------------------------------------------------------
def test_36_render_corrects_a_truly_moving_gaussian_wrongly_labelled_static():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup()
    n = int(xyz0.shape[0])
    logits = torch.zeros((n, 2), dtype=torch.float32)
    logits[:, 0] = 2.0                                   # everything starts STATIC
    model = PartEmbeddingModel(n, 2, 4, temperature=1.0,
                               static_anchor_mask=torch.zeros(n, dtype=torch.bool),
                               init_logits=logits)
    before = model.probabilities().detach().clone()
    after = _run_outer(model, xyz0, rot0, R, T, Q, cams, truly_moving, epochs=60)

    assert float(after[truly_moving, 1].mean()) > float(before[truly_moving, 1].mean()), (
        "p_moving did not increase for truly-moving Gaussians"
    )
    recovered = float((after[truly_moving].argmax(-1) == 1).float().mean())
    assert recovered > 0.9, f"render recovered only {recovered*100:.1f}% of moving Gaussians"
    # No endpoint NN term was used anywhere in this test.


def test_37_render_corrects_a_truly_static_gaussian_wrongly_labelled_moving():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup()
    n = int(xyz0.shape[0])
    logits = torch.zeros((n, 2), dtype=torch.float32)
    logits[:, 1] = 2.0                                   # everything starts MOVING
    model = PartEmbeddingModel(n, 2, 4, temperature=1.0,
                               static_anchor_mask=torch.zeros(n, dtype=torch.bool),
                               init_logits=logits)
    before = model.probabilities().detach().clone()
    after = _run_outer(model, xyz0, rot0, R, T, Q, cams, truly_moving, epochs=60)

    truly_static = ~truly_moving
    assert float(after[truly_static, 0].mean()) > float(before[truly_static, 0].mean()), (
        "p_static did not increase for truly-static Gaussians"
    )
    recovered = float((after[truly_static].argmax(-1) == 0).float().mean())
    assert recovered > 0.9, f"render recovered only {recovered*100:.1f}% of static Gaussians"


# ---------------------------------------------------------------------------
# §39 motion is frozen during the E-step
# ---------------------------------------------------------------------------
def test_39_only_membership_parameters_receive_outer_gradients():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup(n=30, frames=2, n_cameras=2)
    n = int(xyz0.shape[0])
    # Everything that must stay frozen, made grad-capable so a leak would show.
    R = R.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)
    Q = Q.clone().requires_grad_(True)
    control_velocity = torch.randn((1, 8, 3), dtype=DT, requires_grad=True)
    gaussian_xyz = xyz0.clone().requires_grad_(True)
    gaussian_opacity = torch.rand((n, 1), dtype=DT, requires_grad=True)

    model = PartEmbeddingModel(n, 2, 4, temperature=1.0,
                               static_anchor_mask=torch.zeros(n, dtype=torch.bool),
                               init_logits=torch.randn((n, 2), dtype=torch.float32))
    p_st = straight_through_membership(model.probabilities().to(DT))
    out = outer_render_membership_loss(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        p_st=p_st, cameras_by_frame=cams, gaussians=None, pipe=None, background=None,
        dataset=None, args=None, render_fn=_make_render_fn(_true_positions_at(xyz0, truly_moving, T.detach(), 1)),
    )
    out["loss"].backward()

    assert model.logits.grad is not None and float(model.logits.grad.abs().max()) > 0.0
    for name, tensor in (("part_rotation", R), ("part_translation", T), ("part_quaternion", Q),
                         ("control_velocity", control_velocity), ("gaussian_xyz", gaussian_xyz),
                         ("gaussian_opacity", gaussian_opacity)):
        assert tensor.grad is None, f"{name} received a gradient during the E-step"


def test_23_candidate_states_carry_no_motion_graph():
    """§28: candidate states are differentiable only w.r.t. labels."""
    xyz0, rot0, R, T, Q, *_ = _setup(n=20, frames=3, n_cameras=1)
    R = R.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)
    xyz, quat = candidate_state_at_frame(xyz0, rot0, R, T, Q, 2)
    assert not xyz.requires_grad and not quat.requires_grad, "candidate state retained a motion graph"


# ---------------------------------------------------------------------------
# §40 video dominates the 3D regularizers
# ---------------------------------------------------------------------------
def test_40_three_d_regularizer_gradient_is_capped_below_video():
    torch.manual_seed(3)
    logits = torch.randn((40, 2), dtype=DT, requires_grad=True)
    p = torch.softmax(logits, dim=-1)
    video = (p[:, 1] * 0.01).sum()          # weak-looking but the primary signal
    reg = (p[:, 0] * 50.0).sum()            # deliberately overwhelming 3D term

    scale, g_video, g_reg = compute_detached_3d_gradient_cap(video, reg, [logits], rho=0.30)
    assert not scale.requires_grad, "the 3D scale must be detached"
    assert g_reg > g_video, "test vacuous: the 3D term was not dominant to begin with"
    assert float(scale) < 1.0

    scaled = float(scale) * g_reg
    assert scaled <= 0.30 * g_video + 1e-9, (
        f"scaled 3D gradient {scaled:.4e} exceeds rho * video {0.30 * g_video:.4e}"
    )

    # The net label gradient must follow the video, not the regularizer.
    total = video + scale * reg
    logits.grad = None
    total.backward()
    net = logits.grad.detach()
    logits2 = logits.detach().clone().requires_grad_(True)
    (torch.softmax(logits2, dim=-1)[:, 1] * 0.01).sum().backward()
    video_only = logits2.grad.detach()
    cos = torch.nn.functional.cosine_similarity(net.reshape(-1), video_only.reshape(-1), dim=0)
    assert float(cos) > 0.0, f"net label gradient opposes the video gradient (cos {float(cos):.3f})"


def test_40b_cap_is_inactive_when_video_already_dominates():
    logits = torch.randn((20, 2), dtype=DT, requires_grad=True)
    p = torch.softmax(logits, dim=-1)
    scale, g_v, g_r = compute_detached_3d_gradient_cap((p[:, 1] * 10.0).sum(), (p[:, 0] * 0.01).sum(),
                                                       [logits], rho=0.30)
    assert float(scale) == 1.0, "the cap must not amplify or interfere when video dominates"


# ---------------------------------------------------------------------------
# §41 3D evidence fills in where the video is blind
# ---------------------------------------------------------------------------
def test_41_three_d_regularizer_can_move_a_gaussian_invisible_to_every_camera():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup(n=24, frames=2, n_cameras=2)
    n = int(xyz0.shape[0])
    hidden = torch.zeros(n, dtype=torch.bool)
    hidden[0] = True
    visible = ~hidden
    for frame in cams:
        for cam in frame:
            cam["visible"] = visible

    model = PartEmbeddingModel(n, 2, 4, temperature=1.0,
                               static_anchor_mask=torch.zeros(n, dtype=torch.bool),
                               init_logits=torch.zeros((n, 2), dtype=torch.float32))
    p_st = straight_through_membership(model.probabilities().to(DT))
    out = outer_render_membership_loss(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        p_st=p_st, cameras_by_frame=cams, gaussians=None, pipe=None, background=None,
        dataset=None, args=None, render_fn=_make_render_fn(_true_positions_at(xyz0, truly_moving, T, 1)),
    )
    out["loss"].backward()
    video_grad = model.logits.grad.detach().clone()
    assert float(video_grad[0].abs().max()) < 1e-12, "an invisible Gaussian got a video gradient"
    assert float(video_grad[visible].abs().max()) > 0.0, "visible Gaussians got no video gradient"

    # A 3D candidate term favouring static must still be able to move it.
    model.logits.grad = None
    p = model.probabilities().to(DT)
    candidate_energy = torch.zeros((n, 2), dtype=DT)
    candidate_energy[0, 1] = 1.0                      # moving is a bad endpoint for it
    (p * candidate_energy).sum().backward()
    assert float(model.logits.grad[0].abs().max()) > 0.0, (
        "3D evidence could not update the invisible Gaussian"
    )


# ---------------------------------------------------------------------------
# §42 no acceptance gate
# ---------------------------------------------------------------------------
def test_42_no_global_acceptance_gate_in_the_outer_e_step():
    source = inspect.getsource(nested_pipeline.run_nested_part_embedding_killing_discovery)
    outer = source[source.index("D. OUTER E-STEP"):]
    for forbidden in ("reject", "accepted = False", "evaluate_inner_solve", "if.*>.*previous_video"):
        assert forbidden not in outer, f"a gate crept into the E-step: {forbidden}"
    # The only rollback allowed is numerical.
    assert "torch.isfinite(total.detach())" in outer, "numerical backtrack missing"
    assert "optimizer.step()" in outer, "the label optimizer must always run"


# ---------------------------------------------------------------------------
# §17 / §18 local label regularizers
# ---------------------------------------------------------------------------
def test_17_label_graph_allows_a_boundary_where_motion_evidence_disagrees():
    torch.manual_seed(4)
    n = 200
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=10)
    p = torch.softmax(torch.randn((n, 2), dtype=DT), dim=-1)

    same = torch.zeros(n, dtype=DT)                       # identical motion evidence
    split = torch.where(x[:, 0] < 0.5, -5.0, 5.0).to(DT)  # incompatible across a boundary
    l_same = motion_aware_label_graph_loss(p, same, graph, sigma_e=1.0)
    l_split = motion_aware_label_graph_loss(p, split, graph, sigma_e=1.0)
    assert float(l_split) < float(l_same), (
        "the graph term did not relax across incompatible motion evidence"
    )
    # It is a real coherence term: identical labels cost nothing.
    uniform = torch.zeros((n, 2), dtype=DT)
    uniform[:, 0] = 1.0
    assert float(motion_aware_label_graph_loss(uniform, same, graph)) < 1e-15


def test_18_kinetic_label_loss_penalises_an_incompatible_moving_label():
    n = 10
    p = torch.zeros((n, 2), dtype=DT)
    p[:, 1] = 1.0                                    # everything labelled moving
    energy = torch.zeros((n, 1), dtype=DT)
    energy[:5] = 3.0                                 # first half does not behave rigidly
    loss = kinetic_label_loss(p, energy)
    assert abs(float(loss) - 1.5) < 1e-12, float(loss)

    p_static = torch.zeros((n, 2), dtype=DT)
    p_static[:, 0] = 1.0
    assert float(kinetic_label_loss(p_static, energy)) == 0.0, (
        "the static class must not be charged moving-part kinetic energy"
    )


# ---------------------------------------------------------------------------
# §29 frame alignment
# ---------------------------------------------------------------------------
def test_29_frame_alignment_state_t_renders_against_image_t():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup(n=12, frames=5, n_cameras=1)
    # T_k(0) must be identity.
    assert torch.allclose(R[0], torch.eye(3, dtype=DT).expand_as(R[0]), atol=1e-15)
    assert float(T[0].abs().max()) == 0.0

    n = int(xyz0.shape[0])
    p_st = torch.zeros((n, 2), dtype=DT)
    p_st[:, 1] = 1.0                                  # everything on the moving hypothesis

    seen: list[int] = []
    for frame in range(5):
        cand_xyz, _ = candidate_state_at_frame(xyz0, rot0, R, T, Q, frame)
        state = (p_st[..., None] * cand_xyz).sum(dim=1)
        expected = xyz0 + T[frame, 1]
        assert torch.allclose(state, expected, atol=1e-14), f"frame {frame} state mismatch"
        # Off-by-one would match frame+1 instead.
        if frame + 1 < 5:
            assert not torch.allclose(state, xyz0 + T[frame + 1, 1], atol=1e-9)
        seen.append(frame)
    assert seen == [0, 1, 2, 3, 4]


def test_8_one_epoch_visits_every_frame_and_every_camera():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup(n=16, frames=6, n_cameras=3)
    n = int(xyz0.shape[0])
    p_st = straight_through_membership(torch.softmax(torch.randn((n, 2), dtype=DT), dim=-1))
    out = outer_render_membership_loss(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        p_st=p_st, cameras_by_frame=cams, gaussians=None, pipe=None, background=None,
        dataset=None, args=None, render_fn=_make_render_fn(xyz0),
    )
    assert out["num_frames"] == 6 and out["num_views"] == 18
    assert len(out["per_camera"]) == 18
    assert sorted({r["frame"] for r in out["per_camera"]}) == list(range(6))


def test_8b_gradient_accumulation_matches_the_single_graph_loss():
    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup(n=16, frames=3, n_cameras=2)
    n = int(xyz0.shape[0])
    logits = torch.randn((n, 2), dtype=DT, requires_grad=True)
    fn = _make_render_fn(_true_positions_at(xyz0, truly_moving, T, 2))

    p_st = straight_through_membership(torch.softmax(logits, dim=-1))
    single = outer_render_membership_loss(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        p_st=p_st, cameras_by_frame=cams, gaussians=None, pipe=None, background=None,
        dataset=None, args=None, render_fn=fn,
    )
    single["loss"].backward()
    reference = logits.grad.detach().clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    p_st2 = straight_through_membership(torch.softmax(logits2, dim=-1))
    outer_render_membership_loss(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        p_st=p_st2, cameras_by_frame=cams, gaussians=None, pipe=None, background=None,
        dataset=None, args=None, render_fn=fn,
        accumulate=lambda t: t.backward(retain_graph=True),
    )
    assert torch.allclose(logits2.grad, reference, atol=1e-12), (
        f"accumulation differs from the single-graph gradient by "
        f"{float((logits2.grad - reference).abs().max()):.3e}"
    )


def test_32_video_is_the_principal_label_gradient_by_construction():
    """§32/§43: no hand-tuned constant may let a regularizer outrank the video.

    The first real run had lambda_graph=0.05 against a graph loss 25x the video
    loss, so the graph term became 1.27x video. The fix is the same detached
    subordination used for the 3D block, applied to the local regularizers.
    """
    source = inspect.getsource(nested_pipeline.run_nested_part_embedding_killing_discovery)
    assert "scale_local" in source and "rho_local_gradient_cap" in source, (
        "local regularizers are not subordinated to the video gradient"
    )
    assert "scale_local * local_reg" in source

    # The cap itself: a regularizer 100x stronger than the video is pulled under rho.
    torch.manual_seed(9)
    logits = torch.randn((50, 2), dtype=DT, requires_grad=True)
    p = torch.softmax(logits, dim=-1)
    video = (p[:, 1] * 0.01).sum()
    local = (p[:, 0] * 1.0).sum()
    scale, g_v, g_l = compute_detached_3d_gradient_cap(video, local, [logits], rho=0.50)
    assert g_l > g_v, "test vacuous"
    assert float(scale) * g_l <= 0.50 * g_v + 1e-9, (
        f"scaled local gradient {float(scale) * g_l:.3e} exceeds 0.5 * video {0.5 * g_v:.3e}"
    )


# ---------------------------------------------------------------------------
# §26 temperature must not fall while label transitions are still rising
# ---------------------------------------------------------------------------
def test_26_temperature_does_not_fall_while_transitions_are_rising():
    from rigid_parts.nested_pipeline import adaptive_temperature

    tau = 1.0
    # Transitions climbing 117 -> 213 (the real run's trajectory).
    prev = None
    for count in (117, 136, 150, 166, 174, 213):
        tau, blockers = adaptive_temperature(
            tau, motion_accepted=True, label_churn_fraction=0.008,
            candidate_separation=1.0, geometry_stable=True,
            transitions=count, previous_transitions=prev,
        )
        prev = count
        assert tau == 1.0, f"tau fell to {tau} while transitions rose to {count}"
        assert any("transitions" in b for b in blockers), blockers

    # Once transitions genuinely decrease, tau may finally move.
    tau, blockers = adaptive_temperature(
        tau, motion_accepted=True, label_churn_fraction=0.0,
        candidate_separation=1.0, geometry_stable=True,
        transitions=90, previous_transitions=213,
    )
    assert tau < 1.0 and not blockers, (tau, blockers)


# ---------------------------------------------------------------------------
# §12 relative view degradation
# ---------------------------------------------------------------------------
def test_12_view_degradation_is_relative_not_absolute():
    from rigid_parts.outer_render import view_degradation_penalty

    # Two cameras with very different irreducible scales, both unchanged.
    base = {(0, 0): 0.001, (0, 1): 0.100}
    same = {(0, 0): torch.tensor(0.001, dtype=DT), (0, 1): torch.tensor(0.100, dtype=DT)}
    assert float(view_degradation_penalty(same, base)) == 0.0, (
        "cameras with different absolute scales were penalised for being different"
    )
    # Worsening the cheap camera by 50% is penalised like worsening the dear one.
    worse_cheap = {(0, 0): torch.tensor(0.0015, dtype=DT), (0, 1): torch.tensor(0.100, dtype=DT)}
    worse_dear = {(0, 0): torch.tensor(0.001, dtype=DT), (0, 1): torch.tensor(0.150, dtype=DT)}
    a = float(view_degradation_penalty(worse_cheap, base))
    b = float(view_degradation_penalty(worse_dear, base))
    # eps in the denominator leaves a ~1e-7 scale-dependent residue; both are 0.25.
    assert abs(a - b) < 1e-5 and a > 0.0, (a, b)
    # Improving a camera is never penalised.
    better = {(0, 0): torch.tensor(0.0005, dtype=DT), (0, 1): torch.tensor(0.050, dtype=DT)}
    assert float(view_degradation_penalty(better, base)) == 0.0


# ---------------------------------------------------------------------------
# §35 counterfactual rendering agrees with the label gradient direction
# ---------------------------------------------------------------------------
def test_35_counterfactual_render_agrees_with_the_label_gradient():
    from rigid_parts.outer_render import counterfactual_label_deltas

    xyz0, rot0, R, T, Q, truly_moving, shift, cams = _setup(n=12, frames=2, n_cameras=2)
    n = int(xyz0.shape[0])
    true_xyz = _true_positions_at(xyz0, truly_moving, T, 1)
    fn = _make_render_fn(true_xyz)

    # Everything wrongly static; pick a genuinely moving Gaussian.
    hard = torch.zeros(n, dtype=torch.long)
    gid = int(truly_moving.nonzero()[0])
    records = counterfactual_label_deltas(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        hard_label=hard, gaussian_ids=[gid], cameras_by_frame=cams,
        gaussians=None, pipe=None, background=None, dataset=None, args=None,
        frames=[1], render_fn=fn,
    )
    delta_moving = records[0]["delta_label_1"]
    assert delta_moving < 0.0, f"switching a moving Gaussian to moving should help ({delta_moving:.3e})"

    # The label gradient must point the same way: increase p_i,moving.
    logits = torch.zeros((n, 2), dtype=DT, requires_grad=True)
    p_st = straight_through_membership(torch.softmax(logits, dim=-1))
    out = outer_render_membership_loss(
        xyz0=xyz0, rot0=rot0, part_rotation=R, part_translation=T, part_quaternion=Q,
        p_st=p_st, cameras_by_frame=cams, gaussians=None, pipe=None, background=None,
        dataset=None, args=None, frames=[1], render_fn=fn,
    )
    out["loss"].backward()
    # Descending the gradient raises the moving logit relative to static.
    assert float(logits.grad[gid, 1]) < float(logits.grad[gid, 0]), (
        "§35 violated: gradient direction disagrees with the counterfactual render"
    )



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
    print(f"PASS render_centric_outer_e_step ({len(tests)} tests)")
