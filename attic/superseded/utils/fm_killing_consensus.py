"""The inner physical loop: flow matching <-> consensus velocity <-> Killing.

For fixed responsibilities this module answers one question: what is the best
rigid physical motion consistent with the flow-matching transport and the
rasterized multiview velocity observations?

Three pieces, in the order the loop uses them.

``compute_consensus_velocity``
    Fuses the FM proposal, the camera-supported observation and the current
    Killing prediction into one velocity per Gaussian per interval, weighted by
    the responsibilities.  A 3x3 solve, closed form.

``refit_killing_fields``
    Projects that consensus back onto the rigid manifold, one generator plus one
    signed speed per interval per part, weighted by ``r^gamma`` so uncertain
    points move the field less than trusted ones.

``update_core_shell_hysteresis``
    Decides which points are trusted enough to fit from.  A point that has just
    become probable enters the shell and only joins the fitting core after it
    has stayed probable, which is what stops the field from confirming itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.rigid_killing_motion import (
    RigidKillingMode,
    integrate_rigid_mode_exact,
    predict_rigid_cell_velocity,
    solve_rigid_killing_mode_all_intervals,
)

_EPS = 1.0e-8


@dataclass
class ConsensusWeights:
    """Weights of the four terms in the consensus objective.

    ``gamma`` is the exponent on the responsibility in the Killing and static
    pulls: above one it makes an uncertain point contribute much less than a
    confident one, which is what keeps a half-assigned Gaussian from dragging
    the consensus toward a part it may not belong to.
    """

    fm: float = 1.0
    observed: float = 1.0
    killing: float = 1.0
    static: float = 1.0
    gamma: float = 2.0
    ridge: float = 1.0e-6


@torch.no_grad()
def compute_consensus_velocity(
    *,
    fm_velocity: torch.Tensor,
    fm_confidence: torch.Tensor,
    observed_velocity: torch.Tensor,
    observed_normal_matrix: torch.Tensor,
    observed_valid: torch.Tensor,
    killing_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    killing_valid=None,
    weights: ConsensusWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Fuse FM, observation and Killing into one velocity per Gaussian/interval.

    Minimises, per Gaussian ``i`` and interval ``t``,

        lambda_F c_FM ||v - v_FM||^2
      + lambda_V (v - v_obs)^T N (v - v_obs)
      + lambda_K sum_k r_k^gamma ||v - v_K,k||^2
      + lambda_0 r_0^gamma ||v||^2

    which is quadratic in ``v``, so the minimum is the solution of a 3x3 SPD
    system.  The last term is the static hypothesis pulling toward zero: it is
    not a fitted field, it is the literal statement that a static point does not
    move, and it is why a static-dominant Gaussian gets a consensus near zero
    even when FM proposes something.

    The observation term uses the rasterized normal matrix rather than an
    isotropic weight, so a velocity the cameras only constrain in two directions
    is only pulled in those two directions.

    ``killing_valid`` marks which parts have actually been fitted.  A part that
    has not been drops out of *both* sides of the system rather than
    contributing a zero prediction: a zero-velocity target with a real weight is
    the static hypothesis, and letting an unfitted part assert it would halve
    the consensus on genuinely moving points before the first Killing fit ever
    happens.
    """
    weights = weights or ConsensusWeights()
    velocity_fm = torch.as_tensor(fm_velocity)
    device, dtype = velocity_fm.device, velocity_fm.dtype
    intervals, n = velocity_fm.shape[0], velocity_fm.shape[1]
    labels = responsibilities.shape[-1]
    if killing_velocity.shape[0] != labels - 1:
        raise ValueError("killing_velocity must have one entry per moving part")

    eye = torch.eye(3, device=device, dtype=dtype).expand(intervals, n, 3, 3)
    confidence_fm = fm_confidence.to(device=device, dtype=dtype).clamp_min(0.0)
    valid = observed_valid.to(device=device)
    normal = observed_normal_matrix.to(device=device, dtype=dtype)
    normal = torch.where(valid[..., None, None], normal, torch.zeros_like(normal))
    velocity_obs = torch.where(
        valid[..., None], observed_velocity.to(device=device, dtype=dtype),
        torch.zeros_like(velocity_fm),
    )

    static_share = responsibilities[:, 0].to(device=device, dtype=dtype).clamp_min(0.0)
    moving_share = responsibilities[:, 1:].to(device=device, dtype=dtype).clamp_min(0.0)
    static_weight = static_share.pow(weights.gamma)[None]           # [1,N]
    moving_weight = moving_share.pow(weights.gamma).T[:, None]      # [K,1,N]
    if killing_valid is not None:
        gate = torch.as_tensor(
            [bool(item) for item in killing_valid], device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        if gate.shape[0] != moving_weight.shape[0]:
            raise ValueError("killing_valid must have one entry per moving part")
        moving_weight = moving_weight * gate

    hessian = weights.fm * confidence_fm[..., None, None] * eye
    hessian = hessian + weights.observed * normal
    hessian = hessian + weights.killing * moving_weight.sum(0)[..., None, None] * eye
    hessian = hessian + weights.static * static_weight[..., None, None] * eye

    rhs = weights.fm * confidence_fm[..., None] * velocity_fm
    rhs = rhs + weights.observed * torch.einsum("tnij,tnj->tni", normal, velocity_obs)
    rhs = rhs + weights.killing * (
        moving_weight[..., None] * killing_velocity.to(device=device, dtype=dtype)
    ).sum(0)
    # The static term contributes nothing to the right-hand side: its target is
    # exactly zero, which is the whole content of the static hypothesis.

    trace = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1)
    ridge = weights.ridge * (trace / 3.0).clamp_min(_EPS) + _EPS
    regularised = hessian + ridge[..., None, None] * torch.eye(3, device=device, dtype=dtype)
    consensus = torch.linalg.solve(regularised, rhs[..., None]).squeeze(-1)

    # A Gaussian with no evidence at all in an interval has only the ridge
    # holding the system up; report that rather than letting a numerically tiny
    # solve pass as a measurement.
    support = (
        weights.fm * confidence_fm
        + weights.observed * torch.diagonal(normal, dim1=-2, dim2=-1).sum(-1) / 3.0
        + weights.killing * moving_weight.sum(0)
        + weights.static * static_weight
    )
    informative = support > _EPS
    consensus = torch.where(informative[..., None], consensus, torch.zeros_like(consensus))
    return {
        "consensus_velocity": consensus,
        "support": support,
        "informative": informative,
    }


@torch.no_grad()
def refit_killing_fields(
    *,
    canonical_fg: torch.Tensor,
    consensus_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    core_masks: torch.Tensor,
    informative: torch.Tensor | None = None,
    gamma: float = 2.0,
    previous_modes: list | None = None,
    previous_speeds: torch.Tensor | None = None,
    solver_kwargs: dict | None = None,
) -> dict:
    """Project the consensus velocity onto one rigid Killing field per part.

    The fit weight is ``r_ik^gamma`` restricted to that part's trusted core, so
    the M-step does not treat every active point equally: a point that has only
    just become probable is in the shell and carries no fitting weight at all
    until it has persisted.  That asymmetry is what breaks the self-confirming
    loop where a field recruits a point and the point then vouches for the
    field.

    A part whose core is too small to constrain a generator keeps its previous
    mode rather than being refitted from noise.
    """
    points = torch.as_tensor(canonical_fg).detach().cpu().double()
    velocity = torch.as_tensor(consensus_velocity).detach().cpu().double()
    share = torch.as_tensor(responsibilities).detach().cpu().double()
    cores = torch.as_tensor(core_masks).detach().cpu().bool()
    intervals, n = velocity.shape[0], velocity.shape[1]
    k = cores.shape[0]
    kwargs = dict(solver_kwargs or {})
    minimum = int(kwargs.get("min_valid_cells", 8))

    base_valid = (
        torch.ones((intervals, n), dtype=torch.bool)
        if informative is None
        else torch.as_tensor(informative).detach().cpu().bool()
    )

    def keep_previous_speed(part: int) -> torch.Tensor:
        """Retaining a mode must retain its speeds.

        Returning zeros here would silently freeze the part at the canonical
        pose while its geometry still looked correct -- the axis stays right and
        the motion quietly disappears, which is very hard to read off a
        diagnostic that only reports the axis.
        """
        if previous_speeds is None:
            return torch.zeros(intervals, dtype=torch.float64)
        return torch.as_tensor(previous_speeds[part]).detach().cpu().double().reshape(-1)

    modes, speeds, diagnostics = [], [], []
    for part in range(k):
        weight_row = share[:, part + 1].clamp(0.0, 1.0).pow(float(gamma))
        weight_row = weight_row * cores[part].double()
        valid = base_valid & (weight_row > 0)[None]
        previous = None if previous_modes is None else previous_modes[part]
        if int(valid.any(dim=0).sum()) < max(3, minimum):
            modes.append(previous or _null_mode(points))
            speeds.append(keep_previous_speed(part))
            diagnostics.append({
                "part": part + 1,
                "refitted": False,
                "reason": "core_too_small",
                "core_size": int(cores[part].sum()),
                "weighted_points": int((weight_row > 0).sum()),
            })
            continue
        weights = weight_row[None].expand(intervals, n).clone()
        mode, speed, detail = solve_rigid_killing_mode_all_intervals(
            points[None].expand(intervals, n, 3).clone(),
            velocity,
            weights=weights,
            valid_mask=valid,
            reference_point=None if previous is None else previous.reference_point,
            **kwargs,
        )
        if not mode.valid and previous is not None:
            modes.append(previous)
            speeds.append(keep_previous_speed(part))
            diagnostics.append({
                "part": part + 1, "refitted": False, "reason": "solver_rejected_the_fit",
                "core_size": int(cores[part].sum()),
            })
            continue
        modes.append(mode)
        speeds.append(torch.as_tensor(speed).detach().cpu().double().reshape(-1))
        diagnostics.append({
            "part": part + 1,
            "refitted": True,
            "core_size": int(cores[part].sum()),
            "weighted_points": int((weight_row > 0).sum()),
            "rigid_rms": float(mode.diagnostics.get("rigid_rms", float("nan"))),
            "static_rms": float(mode.diagnostics.get("static_rms", float("nan"))),
            "converged": bool(mode.diagnostics.get("converged", False)),
        })
    return {
        "modes": modes,
        "interval_speeds": torch.stack(speeds),
        "diagnostics": diagnostics,
    }


def _null_mode(points: torch.Tensor) -> RigidKillingMode:
    return RigidKillingMode(
        reference_point=points.mean(dim=0),
        reference_velocity=torch.zeros(3, dtype=torch.float64),
        angular_mode=torch.zeros(3, dtype=torch.float64),
        mode_type="translational",
        normalization_scale=0.0,
        valid=False,
        diagnostics={"reason": "no_valid_fit"},
    )


@torch.no_grad()
def predict_all_killing_velocities(
    modes: list, interval_speeds: torch.Tensor, canonical_fg: torch.Tensor
) -> torch.Tensor:
    """Killing velocity of every part at every canonical point and interval.

    Returns ``[K, T-1, N_F, 3]``.  An invalid part contributes exactly zero,
    which is the same thing the static hypothesis contributes, so a part that
    failed to fit cannot pull the consensus anywhere.
    """
    points = torch.as_tensor(canonical_fg).detach()
    speeds = torch.as_tensor(interval_speeds)
    out = []
    for part, mode in enumerate(modes):
        if not mode.valid:
            out.append(torch.zeros((speeds.shape[1],) + tuple(points.shape), device=points.device, dtype=points.dtype))
            continue
        scaled = speeds[part].to(device=points.device, dtype=points.dtype)
        out.append(predict_rigid_cell_velocity(mode.to(device=points.device, dtype=points.dtype), points, scaled))
    return torch.stack(out)


@torch.no_grad()
def integrate_killing_positions(
    modes: list, interval_speeds: torch.Tensor, canonical_fg: torch.Tensor, frame_times
) -> torch.Tensor:
    """Exact SE(3) transport of every canonical point under every part.

    Returns ``[K, T, N_F, 3]`` with frame 0 equal to the canonical positions.
    The coordinate is the running integral of the interval speeds, and the
    transform is the exact exponential of the generator, never a linearisation.
    """
    points = torch.as_tensor(canonical_fg).detach()
    # The coordinate integral is done on the CPU in float64 regardless of where
    # the speeds live: the state tensors are on the GPU, the frame times are
    # plain Python floats, and mixing the two silently is a device error the
    # moment this leaves a CPU-only test.
    speeds = torch.as_tensor(interval_speeds).detach().cpu().double()
    times = torch.as_tensor([float(value) for value in frame_times], dtype=torch.float64)
    coordinates = torch.cat(
        (torch.zeros((speeds.shape[0], 1), dtype=torch.float64),
         torch.cumsum(speeds * torch.diff(times)[None], dim=-1)),
        dim=-1,
    )
    tracks = []
    for part, mode in enumerate(modes):
        frames = [points]
        for index in range(1, coordinates.shape[1]):
            if not mode.valid:
                frames.append(points)
                continue
            moved = integrate_rigid_mode_exact(
                points,
                mode.to(device=points.device, dtype=points.dtype),
                coordinates[part, index].to(points),
            )["positions"]
            frames.append(moved)
        tracks.append(torch.stack(frames))
    return torch.stack(tracks)


@torch.no_grad()
def integrate_killing_transforms(
    modes: list,
    interval_speeds: torch.Tensor,
    frame_times,
    *,
    device=None,
    dtype=None,
) -> dict[str, torch.Tensor]:
    """Exact cumulative SE(3) transforms of every part at every frame.

    ``integrate_killing_positions`` stores the transported Gaussian positions.
    The debug artifact named ``killing_transforms.pt`` should store the actual
    rigid transforms, otherwise the filename is misleading and Stage-2 audits
    cannot inspect axis/translation drift without re-integrating the modes.
    """
    speeds = torch.as_tensor(interval_speeds).detach().cpu().double()
    times = torch.as_tensor([float(value) for value in frame_times], dtype=torch.float64)
    coordinates = torch.cat(
        (
            torch.zeros((speeds.shape[0], 1), dtype=torch.float64),
            torch.cumsum(speeds * torch.diff(times)[None], dim=-1),
        ),
        dim=-1,
    )
    out_device = torch.device("cpu") if device is None else torch.device(device)
    out_dtype = torch.float32 if dtype is None else dtype
    rotations = []
    translations = []
    quaternions = []
    identity = torch.eye(3, device=out_device, dtype=out_dtype)
    zero = torch.zeros(3, device=out_device, dtype=out_dtype)
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=out_device, dtype=out_dtype)
    probe = torch.zeros((1, 3), device=out_device, dtype=out_dtype)
    for part, mode in enumerate(modes):
        part_rot, part_trans, part_quat = [], [], []
        for frame in range(coordinates.shape[1]):
            if not mode.valid:
                part_rot.append(identity)
                part_trans.append(zero)
                part_quat.append(identity_q)
                continue
            payload = integrate_rigid_mode_exact(
                probe,
                mode.to(device=out_device, dtype=out_dtype),
                coordinates[part, frame].to(device=out_device, dtype=out_dtype),
            )
            part_rot.append(payload["rotation_matrix"])
            part_trans.append(payload["translation"])
            part_quat.append(payload["rotation_quaternion"])
        rotations.append(torch.stack(part_rot))
        translations.append(torch.stack(part_trans))
        quaternions.append(torch.stack(part_quat))
    return {
        "rotations": torch.stack(rotations),
        "translations": torch.stack(translations),
        "rotation_quaternions": torch.stack(quaternions),
        "coordinates": coordinates.to(device=out_device, dtype=out_dtype),
    }


@torch.no_grad()
def build_discrete_physical_proposal(
    *, canonical_fg: torch.Tensor, hard_labels: torch.Tensor, killing_positions: torch.Tensor
) -> torch.Tensor:
    """Assemble the current physical hypothesis, one exact transform per point.

    Returns ``[T, N_F, 3]``.  A point labelled static stays exactly canonical; a
    point labelled ``k`` takes part ``k``'s exact transform.  Deliberately not a
    soft spatial blend of the two: responsibilities are inference weights, and a
    Gaussian sitting at a probability-weighted average of two hypotheses is a
    place the object never is, which would make the render residual meaningless.
    """
    points = torch.as_tensor(canonical_fg).detach()
    labels = torch.as_tensor(hard_labels).to(device=points.device).long()
    frames = killing_positions.shape[1]
    proposal = points[None].expand(frames, *points.shape).clone()
    for part in range(killing_positions.shape[0]):
        mask = labels == (part + 1)
        if bool(mask.any()):
            proposal[:, mask] = killing_positions[part][:, mask].to(proposal)
    return proposal


@torch.no_grad()
def compute_candidate_physical_displacements(
    canonical_fg: torch.Tensor, killing_positions: torch.Tensor
) -> torch.Tensor:
    """Per-label exact displacement candidates, with static exactly zero.

    Returns ``[K+1,T,N_F,3]``.  Label 0 is the null/static hypothesis and is
    therefore identically zero; labels 1..K are the exact integrated Killing
    transports minus the canonical position.
    """
    points = torch.as_tensor(canonical_fg).detach()
    positions = torch.as_tensor(killing_positions).detach().to(points)
    if positions.ndim != 4 or positions.shape[-2:] != points.shape:
        raise ValueError("killing_positions must be [K,T,N_F,3]")
    static = torch.zeros(
        (1, positions.shape[1], points.shape[0], 3),
        device=points.device,
        dtype=points.dtype,
    )
    moving = positions - points[None, None]
    return torch.cat((static, moving), dim=0)


@torch.no_grad()
def compute_fm_physical_endpoint_energy(
    *,
    fm_current_displacement: torch.Tensor,
    fm_confidence: torch.Tensor,
    candidate_displacements: torch.Tensor,
    frame_index: int = -1,
    huber_delta: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Direct 3-D energy between current FM endpoint transport and candidates.

    Static pays ``||d_FM||``.  Moving part ``k`` pays
    ``||d_FM - d_K,k||`` at the requested endpoint frame.  This is the
    bidirectional assignment term: a point currently static can still move if
    the FM transport matches a Killing candidate better than zero.
    """
    fm = torch.as_tensor(fm_current_displacement).detach()
    device, dtype = fm.device, fm.dtype
    candidates = torch.as_tensor(candidate_displacements).detach().to(device=device, dtype=dtype)
    if candidates.ndim != 4 or candidates.shape[-2:] != fm.shape:
        raise ValueError("candidate_displacements must be [K+1,T,N_F,3]")
    frame = int(frame_index)
    if frame < 0:
        frame += int(candidates.shape[1])
    frame = max(0, min(frame, int(candidates.shape[1]) - 1))
    confidence = torch.as_tensor(fm_confidence).detach().to(device=device, dtype=dtype)
    if confidence.ndim == 2:
        confidence = confidence.mean(0)
    confidence = confidence.reshape(-1).clamp(0.0, 1.0)
    if confidence.numel() != fm.shape[0]:
        raise ValueError("fm_confidence must be [N_F] or [T-1,N_F]")

    residual = candidates[:, frame].transpose(0, 1) - fm[:, None]
    norm = residual.norm(dim=-1)
    delta = max(float(huber_delta), _EPS)
    cost = torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta))
    energy = confidence[:, None] * cost
    best = candidates[:, frame, :, :].transpose(0, 1)
    hard = energy.argmin(-1)
    chosen = best[torch.arange(fm.shape[0], device=device), hard]
    return {
        "energy": energy,
        "reliability": confidence,
        "best_residual_norm": (fm - chosen).norm(dim=-1),
    }


@torch.no_grad()
def build_current_physical_scene_xyz(
    canonical_all: torch.Tensor, foreground_idx: torch.Tensor, physical_fg: torch.Tensor
) -> torch.Tensor:
    """Scatter moved foreground positions back into the full canonical cloud."""
    full = torch.as_tensor(canonical_all).detach().clone()
    idx = torch.as_tensor(foreground_idx, device=full.device).long()
    moved = torch.as_tensor(physical_fg).detach().to(full)
    if moved.shape != (idx.numel(), 3):
        raise ValueError("physical_fg must be [N_F,3]")
    full.index_copy_(0, idx, moved)
    return full


@torch.no_grad()
def bootstrap_moving_core_from_observations(
    *,
    observed_velocity: torch.Tensor,
    observed_confidence: torch.Tensor,
    observed_valid: torch.Tensor,
    canonical_fg: torch.Tensor,
    num_moving_parts: int,
    generator: torch.Generator | None = None,
) -> dict:
    """Seed the very first moving core straight from the observed velocity.

    Without this the loop deadlocks: a part's core admits only points with high
    responsibility, responsibility needs a fitted field to prefer, and the field
    is fitted only from the core.  Nothing in that cycle can start.

    The seed is the one piece of evidence that exists before any hypothesis --
    the rasterized multiview velocity.  The split between "moving" and "still"
    comes from Otsu on the observed speed distribution of this scene, so it is
    read off the data rather than set: a scene whose part barely moves and one
    whose part swings open both get a split placed between their own two modes.

    With ``K > 1`` the candidates are divided by k-means on their velocity
    vectors, because two parts of the same object are told apart by moving
    *differently*, not by being far apart in space.
    """
    velocity = torch.as_tensor(observed_velocity)
    valid = torch.as_tensor(observed_valid)
    weight = torch.as_tensor(observed_confidence).clamp_min(0.0) * valid.to(velocity.dtype)
    total = weight.sum(0)
    speed = (weight * velocity.norm(dim=-1)).sum(0) / total.clamp_min(_EPS)
    observed = total > 0
    n = int(speed.shape[0])
    k = int(num_moving_parts)
    core = torch.zeros((k, n), dtype=torch.bool, device=speed.device)
    if not bool(observed.any()):
        return {"core_masks": core, "candidates": observed, "split": None,
                "reason": "no_valid_velocity_observations"}

    from utils.kpart_velocity_field_discovery import _otsu_threshold

    scores = speed[observed]
    # Otsu works on a normalised score; the scale is restored afterwards so the
    # reported split is in the physical units of the observation.
    peak = float(scores.max())
    if peak <= _EPS:
        return {"core_masks": core, "candidates": observed, "split": 0.0,
                "reason": "no_observed_motion"}
    split = _otsu_threshold(scores / peak)
    if split is None:
        return {"core_masks": core, "candidates": observed, "split": None,
                "reason": "observed_speed_has_no_split"}
    threshold = float(split) * peak
    candidates = observed & (speed >= threshold)
    if not bool(candidates.any()):
        return {"core_masks": core, "candidates": candidates, "split": threshold,
                "reason": "no_candidate_above_the_split"}

    if k == 1:
        core[0] = candidates
    else:
        mean_velocity = (weight[..., None] * velocity).sum(0) / total.clamp_min(_EPS)[:, None]
        labels = _kmeans_labels(mean_velocity[candidates], k, generator=generator)
        index = torch.nonzero(candidates, as_tuple=False).flatten()
        for part in range(k):
            core[part, index[labels == part]] = True
    return {
        "core_masks": core,
        "candidates": candidates,
        "split": threshold,
        "observed_speed": speed,
        "reason": None,
    }


def _kmeans_labels(values: torch.Tensor, k: int, *, iterations: int = 25, generator=None):
    """Small deterministic k-means, seeded farthest-point for reproducibility."""
    data = torch.as_tensor(values).double()
    if data.shape[0] <= k:
        return torch.arange(data.shape[0]) % k
    centres = [data[int(data.norm(dim=-1).argmax())]]
    for _ in range(1, k):
        distance = torch.stack([(data - c).norm(dim=-1) for c in centres]).amin(0)
        centres.append(data[int(distance.argmax())])
    centre = torch.stack(centres)
    labels = torch.zeros(data.shape[0], dtype=torch.long)
    for _ in range(int(iterations)):
        labels = torch.cdist(data, centre).argmin(-1)
        for part in range(k):
            mask = labels == part
            if bool(mask.any()):
                centre[part] = data[mask].mean(0)
    return labels


@dataclass
class CoreShellConfig:
    """Hysteresis for promoting a point into a part's trusted fitting core."""

    enter: float = 0.90
    exit: float = 0.30
    persistence_cycles: int = 2


@torch.no_grad()
def update_core_shell_hysteresis(
    *,
    responsibilities: torch.Tensor,
    moving_streak: torch.Tensor,
    static_streak: torch.Tensor,
    previous_core: torch.Tensor | None = None,
    previous_static_core: torch.Tensor | None = None,
    config: CoreShellConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Promote persistent points into the core; demote only those that fell away.

    Genuine hysteresis, with two different thresholds doing two different jobs:

        enter  -- a point must exceed this for ``persistence_cycles`` in a row
                  before it is trusted enough to fit from
        exit   -- an existing core member is only expelled once it falls below
                  this, which is much lower

    The band between them is the point: a member that dips to 0.5 keeps its
    place, so the fitting set does not churn every time the assignment breathes.
    Treating anything below ``enter`` as an expulsion -- which is a hard gate,
    not hysteresis -- empties the core on the first soft cycle and the field
    never gets fitted again.
    """
    config = config or CoreShellConfig()
    share = torch.as_tensor(responsibilities)
    static = share[:, 0]
    moving = share[:, 1:]

    above = moving >= float(config.enter)
    below = moving < float(config.exit)
    streak = torch.where(above, moving_streak + 1, torch.zeros_like(moving_streak))
    streak = torch.where(below, torch.zeros_like(streak), streak)
    entering = streak >= int(config.persistence_cycles)
    if previous_core is None:
        core = entering
    else:
        retained = previous_core.T.to(entering.device) & ~below
        core = retained | entering

    static_above = static >= float(config.enter)
    static_below = static < float(config.exit)
    new_static_streak = torch.where(
        static_above, static_streak + 1, torch.zeros_like(static_streak)
    )
    new_static_streak = torch.where(
        static_below, torch.zeros_like(new_static_streak), new_static_streak
    )
    static_entering = new_static_streak >= int(config.persistence_cycles)
    static_core = (
        static_entering
        if previous_static_core is None
        else (previous_static_core.to(static_entering.device) & ~static_below) | static_entering
    )

    # The shell is "probable enough to evaluate, not yet trusted enough to fit".
    shell = (moving >= float(config.exit)) & ~core
    return {
        "core_masks": core.T.contiguous(),
        "shell_masks": shell.T.contiguous(),
        "static_core_mask": static_core,
        "moving_streak": streak,
        "static_streak": new_static_streak,
    }


@torch.no_grad()
def cross_fit_core_validation(
    *,
    canonical_fg: torch.Tensor,
    consensus_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    core_masks: torch.Tensor,
    part: int,
    gamma: float = 2.0,
    solver_kwargs: dict | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Split a part's core in half, fit on each half, score the other.

    A Gaussian that helped fit a field will of course agree with it.  Fitting on
    half the core and measuring the residual on the other half removes that
    circularity: if the two halves describe the same rigid motion the held-out
    residual matches the fitted one, and if the core is contaminated it does not.
    """
    cores = torch.as_tensor(core_masks).detach().cpu().bool()
    index = torch.nonzero(cores[part], as_tuple=False).flatten()
    if index.numel() < 2 * max(3, int((solver_kwargs or {}).get("min_valid_cells", 8))):
        return {"available": False, "core_size": int(index.numel())}
    permutation = torch.randperm(index.numel(), generator=generator)
    halves = (index[permutation[: index.numel() // 2]], index[permutation[index.numel() // 2 :]])

    scores = []
    for fit_ids, test_ids in (halves, halves[::-1]):
        mask = torch.zeros(cores.shape[1], dtype=torch.bool)
        mask[fit_ids] = True
        result = refit_killing_fields(
            canonical_fg=canonical_fg,
            consensus_velocity=consensus_velocity,
            responsibilities=responsibilities,
            core_masks=mask[None],
            gamma=gamma,
            solver_kwargs=solver_kwargs,
        )
        mode = result["modes"][0]
        if not mode.valid:
            return {"available": False, "reason": "half_core_fit_failed"}
        points = torch.as_tensor(canonical_fg).detach().cpu().double()
        velocity = torch.as_tensor(consensus_velocity).detach().cpu().double()
        predicted = predict_rigid_cell_velocity(
            mode, points[None].expand_as(velocity), result["interval_speeds"][0]
        )
        residual = torch.linalg.norm(velocity - predicted, dim=-1)
        scores.append(
            (float(residual[:, fit_ids].mean()), float(residual[:, test_ids].mean()))
        )
    fitted = sum(item[0] for item in scores) / len(scores)
    held_out = sum(item[1] for item in scores) / len(scores)
    return {
        "available": True,
        "core_size": int(index.numel()),
        "fitted_residual": fitted,
        "held_out_residual": held_out,
        # Above one means the field does not generalise inside its own core.
        "generalisation_ratio": held_out / max(fitted, _EPS),
    }
