"""Residual diffusion around the current physical hypothesis.

The denoiser is trained and run *around the proposal the closed loop currently
believes*, not around canonical.  That single change is what turns it from a
motion predictor into a critic: the residual it leaves behind,

    r_i = x_final - x_physical_proposal,

is near zero when the label and the motion are both right, and otherwise points
almost exactly along the displacement that would fix the label.

Two halves.

``corrupt_physical_proposal``
    Builds deliberately wrong hypotheses to train on -- wrong labels, wrong
    axis, wrong speed, wrong FM transport -- because a critic only ever trained
    on nearly-correct proposals never learns what being wrong looks like, and
    would answer "no correction needed" to everything.

``run_physics_conditioned_diffusion``
    Runs the recurrent renderer-supervised denoiser over several stochastic
    trials and reduces them to a mean residual, a covariance and a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from scene.physics_conditioned_gaussian_denoiser import assemble_features

_EPS = 1.0e-8


# ----------------------------------------------------------------------
# Structured physical corruption (spec 18)
# ----------------------------------------------------------------------
@dataclass
class CorruptionConfig:
    """How wrong the training hypotheses are allowed to be.

    Every probability is the chance of applying that corruption to a given
    sample.  They are independent, so a sample can carry several at once, which
    is the realistic case: an early closed-loop cycle usually has a slightly
    wrong axis *and* a mislabelled shell at the same time.
    """

    label_flip_probability: float = 0.25
    static_to_moving_probability: float = 0.15
    moving_to_static_probability: float = 0.15
    axis_perturbation_radians: float = 0.15
    translation_perturbation_ratio: float = 0.15
    speed_scale_log_sigma: float = 0.25
    fm_perturbation_ratio: float = 0.20
    enabled: bool = True


@torch.no_grad()
def corrupt_physical_proposal(
    *,
    canonical_fg: torch.Tensor,
    hard_labels: torch.Tensor,
    killing_modes: list,
    interval_speeds: torch.Tensor,
    frame_times,
    config: CorruptionConfig,
    generator: torch.Generator | None = None,
) -> dict:
    """Produce a deliberately wrong physical proposal and record what was broken.

    Returns the corrupted per-frame positions together with the corrupted labels
    and modes, so the caller can compute the *true* residual the denoiser should
    have produced: the difference between the clean proposal and the corrupted
    one.  That is the supervision signal for teaching the critic what each kind
    of error looks like.
    """
    from utils.fm_killing_consensus import (
        build_discrete_physical_proposal,
        integrate_killing_positions,
    )
    from utils.rigid_killing_motion import RigidKillingMode

    device = canonical_fg.device
    labels = hard_labels.clone()
    k = len(killing_modes)
    applied: list[str] = []

    if not config.enabled:
        positions = integrate_killing_positions(
            killing_modes, interval_speeds, canonical_fg, frame_times
        )
        return {
            "positions": build_discrete_physical_proposal(
                canonical_fg=canonical_fg, hard_labels=labels,
                killing_positions=positions,
            ),
            "labels": labels,
            "modes": killing_modes,
            "interval_speeds": interval_speeds,
            "corruptions": [],
        }

    def uniform(*shape):
        return torch.rand(shape, generator=generator, device="cpu").to(device)

    # --- assignment corruption ---------------------------------------
    static = labels == 0
    moving = ~static
    if k >= 1 and bool(static.any()):
        flip = static & (uniform(labels.shape[0]) < config.static_to_moving_probability)
        if bool(flip.any()):
            part = torch.randint(1, k + 1, (int(flip.sum()),), generator=generator).to(device)
            labels[flip] = part
            applied.append("static_to_moving")
    if bool(moving.any()):
        drop = moving & (uniform(labels.shape[0]) < config.moving_to_static_probability)
        if bool(drop.any()):
            labels[drop] = 0
            applied.append("moving_to_static")
    if k >= 2:
        swap = (labels > 0) & (uniform(labels.shape[0]) < config.label_flip_probability)
        if bool(swap.any()):
            other = torch.randint(1, k + 1, (int(swap.sum()),), generator=generator).to(device)
            labels[swap] = other
            applied.append("part_to_other_part")

    # --- geometry and speed corruption -------------------------------
    modes, speeds = [], interval_speeds.clone()
    for part, mode in enumerate(killing_modes):
        if not mode.valid:
            modes.append(mode)
            continue
        angular = mode.angular_mode.clone()
        linear = mode.reference_velocity.clone()
        if config.axis_perturbation_radians > 0 and float(angular.norm()) > _EPS:
            # Rotate the axis by a small random rotation: a wrong hinge
            # direction is the error the render residual is most able to see.
            axis = angular / angular.norm()
            random = torch.randn(3, generator=generator, dtype=angular.dtype)
            perpendicular = random - torch.dot(random, axis) * axis
            if float(perpendicular.norm()) > _EPS:
                perpendicular = perpendicular / perpendicular.norm()
                angle = float(torch.randn(1, generator=generator)) * config.axis_perturbation_radians
                rotated = axis * torch.cos(torch.tensor(angle, dtype=angular.dtype)) + \
                    perpendicular * torch.sin(torch.tensor(angle, dtype=angular.dtype))
                angular = rotated * angular.norm()
                applied.append("axis")
        if config.translation_perturbation_ratio > 0:
            magnitude = float(linear.norm()) or 1.0
            linear = linear + torch.randn(3, generator=generator, dtype=linear.dtype) * (
                magnitude * config.translation_perturbation_ratio
            )
            applied.append("translation")
        modes.append(
            RigidKillingMode(
                reference_point=mode.reference_point.clone(),
                reference_velocity=linear,
                angular_mode=angular,
                mode_type=mode.mode_type,
                normalization_scale=mode.normalization_scale,
                valid=True,
                diagnostics={"corrupted": True},
            )
        )
        if config.speed_scale_log_sigma > 0:
            factor = float(
                torch.exp(torch.randn(1, generator=generator) * config.speed_scale_log_sigma)
            )
            speeds[part] = speeds[part] * factor
            applied.append("speed")

    positions = integrate_killing_positions(modes, speeds, canonical_fg, frame_times)
    return {
        "positions": build_discrete_physical_proposal(
            canonical_fg=canonical_fg, hard_labels=labels, killing_positions=positions
        ),
        "labels": labels,
        "modes": modes,
        "interval_speeds": speeds,
        "corruptions": sorted(set(applied)),
    }


@torch.no_grad()
def corrupt_fm_velocity(
    fm_velocity: torch.Tensor, ratio: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Perturb the FM transport the critic is conditioned on.

    The denoiser must not learn to trust FM blindly; showing it FM that
    disagrees with the truth is how the ``fm_minus_killing`` feature becomes
    informative rather than decorative.
    """
    if ratio <= 0:
        return fm_velocity
    magnitude = fm_velocity.norm(dim=-1, keepdim=True).mean().clamp_min(_EPS)
    noise = torch.randn(fm_velocity.shape, generator=generator, device="cpu").to(fm_velocity)
    return fm_velocity + noise * magnitude * float(ratio)


# ----------------------------------------------------------------------
# Recurrent denoising and residual statistics (spec 14, 17)
# ----------------------------------------------------------------------
@dataclass
class DiffusionConfig:
    """Settings for one residual-diffusion pass."""

    num_trials: int = 8
    denoise_steps: int = 6
    noise_sigma_ratio: float = 0.15
    step_ratio: float = 0.5
    residual_magnitude_weight: float = 0.01
    residual_spatial_weight: float = 0.0
    seed: int = 20_260_910


@dataclass
class DiffusionResult:
    mean: torch.Tensor                 # [T,N_F,3]
    cov: torch.Tensor                  # [T,N_F,3,3]
    confidence: torch.Tensor           # [T,N_F]
    per_trial: torch.Tensor | None     # [S,T,N_F,3] when retained
    diagnostics: dict = field(default_factory=dict)


def denoise_once(
    denoiser,
    *,
    spec,
    current_positions: torch.Tensor,
    physical_positions: torch.Tensor,
    canonical_fg: torch.Tensor,
    fm_velocity: torch.Tensor,
    killing_velocity: torch.Tensor,
    responsibilities: torch.Tensor,
    assignment_entropy: torch.Tensor,
    hard_labels: torch.Tensor,
    knn_idx: torch.Tensor,
    neighbour_scale: float,
    step_ratio: float,
    render_evidence: dict | None = None,
    observation_confidence: dict | None = None,
) -> torch.Tensor:
    """One recurrent denoising step, returning the updated positions.

    Kept separate from the trial loop so the training path can backpropagate
    through a step while the inference path runs it under ``no_grad``.
    """
    features = assemble_features(
        spec,
        current_positions=current_positions,
        physical_positions=physical_positions,
        canonical_fg=canonical_fg,
        fm_velocity=fm_velocity,
        killing_velocity=killing_velocity,
        responsibilities=responsibilities,
        assignment_entropy=assignment_entropy,
        hard_labels=hard_labels,
        neighbour_scale=neighbour_scale,
        render_evidence=render_evidence,
        observation_confidence=observation_confidence,
    )
    delta = denoiser(features, knn_idx, step_scale=neighbour_scale * float(step_ratio))
    return current_positions + delta


def bootstrap_responsibilities_from_neutral_denoiser(
    denoiser,
    *,
    spec,
    state,
    target_frames,
    knn_idx: torch.Tensor,
    neighbour_scale: float,
    config: DiffusionConfig,
    render_loss_fn,
    render_evidence_fn=None,
    observation_confidence_fn=None,
    iterations: int = 200,
    learning_rate: float = 1.0e-3,
    static_prior: float = 0.7,
    num_moving_parts: int = 1,
    log=print,
) -> dict:
    """Spec 5.1: a short physics-neutral pass to initialise the responsibilities.

    Everything here is deliberately ignorant of the Killing fields.  The
    proposal is exactly canonical, so ``x_init = x_0 + eps`` and the only thing
    supervising the correction is the renderer.  A Gaussian the images want moved
    accumulates a large residual; one already in the right place does not.

    The reason this matters is coverage.  On the USB scene only 4542 of 11263
    foreground Gaussians ever receive a valid multiview velocity observation, so
    the velocity energies leave 4493 genuinely moving Gaussians with no evidence
    at all and the move-activation cost sends every one of them to static --
    a hard recall ceiling of 27.6% that no amount of energy tuning moves.  The
    renderer sees 10733 of them.  This is the evidence channel with full
    coverage, which is why the specification puts it before the loop rather than
    inside it.

    The residual is turned into a probability by its own robust spread, not by a
    threshold: the split is wherever this scene's residuals separate.
    """
    from utils.closed_loop_motion_losses import (
        ResidualRegularizerWeights,
        residual_regularizers,
    )

    canonical = state.canonical_fg
    frames = [int(value) for value in target_frames]
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(config.seed) + 7)
    sigma = float(config.noise_sigma_ratio) * float(neighbour_scale)
    regularizers = ResidualRegularizerWeights(
        magnitude=config.residual_magnitude_weight, spatial=0.0
    )
    # Physics-neutral means physics-neutral: no part is assigned, so every
    # feature that would leak a hypothesis is zero or uniform.
    neutral_labels = torch.zeros(canonical.shape[0], dtype=torch.long, device=canonical.device)
    neutral_share = torch.full(
        (canonical.shape[0], int(num_moving_parts) + 1),
        1.0 / (int(num_moving_parts) + 1),
        device=canonical.device, dtype=canonical.dtype,
    )
    neutral_entropy = -(neutral_share * neutral_share.clamp_min(_EPS).log()).sum(-1)
    zero_velocity = torch.zeros_like(canonical)

    history = []
    for step in range(max(1, int(iterations))):
        frame = frames[step % len(frames)]
        noise = torch.randn(canonical.shape, generator=generator, device="cpu").to(canonical)
        current = canonical + noise * sigma
        optimizer.zero_grad(set_to_none=True)
        for _ in range(max(1, int(config.denoise_steps))):
            evidence = (
                None if render_evidence_fn is None
                else render_evidence_fn(current.detach(), frame)
            )
            confidence = (
                None if observation_confidence_fn is None
                else observation_confidence_fn(max(0, frame - 1))
            )
            current = denoise_once(
                denoiser, spec=spec, current_positions=current,
                physical_positions=canonical, canonical_fg=canonical,
                fm_velocity=zero_velocity, killing_velocity=zero_velocity,
                responsibilities=neutral_share, assignment_entropy=neutral_entropy,
                hard_labels=neutral_labels, knn_idx=knn_idx,
                neighbour_scale=neighbour_scale, step_ratio=config.step_ratio,
                render_evidence=evidence, observation_confidence=confidence,
            )
        penalty = residual_regularizers(
            current - canonical, neighbour_scale=neighbour_scale,
            knn_idx=knn_idx, weights=regularizers,
        )
        total = render_loss_fn(current, frame) + penalty["total"]
        if not bool(torch.isfinite(total)):
            raise RuntimeError("the neutral bootstrap loss became non-finite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        if step % max(1, int(iterations) // 4) == 0 or step == int(iterations) - 1:
            history.append({"step": step, "frame": frame, "total": float(total.detach())})
            log(f"    neutral bootstrap step {step:4d} frame {frame:3d} loss {float(total.detach()):.6f}")

    # Measure the residual the trained neutral critic leaves at every target.
    with torch.no_grad():
        magnitude = torch.zeros(canonical.shape[0], device=canonical.device)
        for frame in frames:
            noise = torch.randn(canonical.shape, generator=generator, device="cpu").to(canonical)
            current = canonical + noise * sigma
            for _ in range(max(1, int(config.denoise_steps))):
                evidence = (
                    None if render_evidence_fn is None
                    else render_evidence_fn(current, frame)
                )
                current = denoise_once(
                    denoiser, spec=spec, current_positions=current,
                    physical_positions=canonical, canonical_fg=canonical,
                    fm_velocity=zero_velocity, killing_velocity=zero_velocity,
                    responsibilities=neutral_share, assignment_entropy=neutral_entropy,
                    hard_labels=neutral_labels, knn_idx=knn_idx,
                    neighbour_scale=neighbour_scale, step_ratio=config.step_ratio,
                    render_evidence=evidence, observation_confidence=None,
                )
            magnitude = magnitude + (current - canonical).norm(dim=-1)
        magnitude = magnitude / max(1, len(frames))

    # Map the residual to a soft moving probability by this scene's own spread.
    finite = magnitude[torch.isfinite(magnitude)]
    scale = float(finite.quantile(0.75)) if finite.numel() else 0.0
    if scale <= 0.0:
        share = torch.full(
            (canonical.shape[0], int(num_moving_parts) + 1),
            (1.0 - static_prior) / int(num_moving_parts), device=canonical.device,
        )
        share[:, 0] = static_prior
        return {"responsibilities": share, "residual": magnitude, "scale": None,
                "reason": "neutral residual has no spread"}
    moving = torch.sigmoid((magnitude - scale) / max(scale, _EPS))
    # Keep it soft and fully mutable, as spec 5 requires: nothing is frozen and
    # no label is made unreachable by the first softmax update.
    moving = moving.clamp(0.05, 0.95)
    share = torch.empty(
        (canonical.shape[0], int(num_moving_parts) + 1),
        device=canonical.device, dtype=canonical.dtype,
    )
    share[:, 0] = 1.0 - moving
    share[:, 1:] = (moving / int(num_moving_parts))[:, None]
    share = share / share.sum(-1, keepdim=True)
    return {
        "responsibilities": share,
        "residual": magnitude,
        "scale": scale,
        "history": history,
        "reason": None,
    }


def train_physics_conditioned_denoiser(
    denoiser,
    *,
    spec,
    state,
    clean_positions: torch.Tensor,
    target_frames,
    knn_idx: torch.Tensor,
    neighbour_scale: float,
    config: DiffusionConfig,
    corruption: CorruptionConfig,
    iterations: int = 100,
    learning_rate: float = 1.0e-3,
    render_evidence_fn=None,
    observation_confidence_fn=None,
    render_loss_fn=None,
    log_every: int = 0,
) -> dict:
    """Stage A of spec 31: teach the critic what a wrong hypothesis looks like.

    Without this the critic is untrained, its residual is just the injected
    noise walked back by an untitialised network, and the transition energy of
    spec 20 reads that noise as "no correction wanted" -- which is a full-weight
    vote for whatever label the point already has.  Measured on the USB scene:
    with an untrained critic voting, every one of 11263 Gaussians collapsed to
    static and the trusted core decayed from 686 to 60; with its vote removed
    the assignment moved normally.  So the critic has to be trained before it is
    trusted, not merely run.

    The supervision is the corruption itself.  ``corrupt_physical_proposal``
    breaks a known-good proposal in a known way, so the exact correction the
    critic should have produced is available in closed form -- no renderer
    needed, and it is a much stronger signal than the render residual for
    teaching that a mislabelled point needs a *whole* displacement undone.  The
    render term is added on top when a renderer is supplied, because the
    corruption distribution is not the same as the distribution of the loop's
    own mistakes.
    """
    from utils.closed_loop_motion_losses import (
        ResidualRegularizerWeights,
        residual_regularizers,
        supervised_residual_loss,
    )

    frames = [int(value) for value in target_frames]
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(config.seed) + 1)
    sigma = float(config.noise_sigma_ratio) * float(neighbour_scale)
    history = []
    regularizers = ResidualRegularizerWeights(
        magnitude=config.residual_magnitude_weight,
        spatial=config.residual_spatial_weight,
    )

    for step in range(max(1, int(iterations))):
        frame = frames[step % len(frames)]
        broken = corrupt_physical_proposal(
            canonical_fg=state.canonical_fg,
            hard_labels=state.hard_labels,
            killing_modes=state.killing_modes,
            interval_speeds=state.killing_interval_speeds,
            frame_times=state.frame_times,
            config=corruption,
            generator=generator,
        )
        proposal = broken["positions"][frame].detach()
        target = clean_positions[frame].detach()
        interval = max(0, min(frame - 1, state.fm_velocity.shape[0] - 1))
        # The critic is conditioned on the *corrupted* labels, because that is
        # what a wrong hypothesis looks like from the inside.
        labels = broken["labels"]
        killing = _killing_velocity_for_labels(state, interval, labels)
        fm = corrupt_fm_velocity(
            state.fm_velocity[interval], corruption.fm_perturbation_ratio, generator
        ).detach()

        noise = torch.randn(proposal.shape, generator=generator, device="cpu").to(proposal)
        current = proposal + noise * sigma
        optimizer.zero_grad(set_to_none=True)
        for _ in range(max(1, int(config.denoise_steps))):
            evidence = (
                None if render_evidence_fn is None
                else render_evidence_fn(current.detach(), frame)
            )
            confidence = (
                None if observation_confidence_fn is None
                else observation_confidence_fn(interval)
            )
            current = denoise_once(
                denoiser, spec=spec, current_positions=current,
                physical_positions=proposal, canonical_fg=state.canonical_fg,
                fm_velocity=fm, killing_velocity=killing,
                responsibilities=state.responsibilities,
                assignment_entropy=state.assignment_entropy,
                hard_labels=labels, knn_idx=knn_idx,
                neighbour_scale=neighbour_scale, step_ratio=config.step_ratio,
                render_evidence=evidence, observation_confidence=confidence,
            )
        supervised = supervised_residual_loss(
            current, target, neighbour_scale=neighbour_scale
        )
        penalty = residual_regularizers(
            current - proposal, neighbour_scale=neighbour_scale,
            knn_idx=knn_idx, weights=regularizers,
        )
        total = supervised + penalty["total"]
        if render_loss_fn is not None:
            total = total + render_loss_fn(current, frame)
        if not bool(torch.isfinite(total)):
            raise RuntimeError("the residual-diffusion loss became non-finite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        if log_every and (step % int(log_every) == 0 or step == int(iterations) - 1):
            history.append({
                "step": step, "frame": frame,
                "total": float(total.detach()),
                "supervised": float(supervised.detach()),
                "corruptions": broken["corruptions"],
                "labels_corrupted": int((labels != state.hard_labels).sum()),
            })
    return {"history": history, "iterations": int(iterations)}


@torch.no_grad()
def run_physics_conditioned_diffusion(
    denoiser,
    *,
    spec,
    state,
    physical_positions: torch.Tensor,
    target_frames,
    knn_idx: torch.Tensor,
    neighbour_scale: float,
    config: DiffusionConfig,
    render_evidence_fn=None,
    observation_confidence_fn=None,
    keep_trials: bool = False,
) -> DiffusionResult:
    """Run several stochastic denoising trials and summarise the residual.

    Each trial perturbs the physical proposal with fresh noise and lets the
    denoiser walk it back.  Where the hypothesis is right the trials all return
    to the proposal and the residual mean is near zero with small covariance;
    where it is wrong they agree on a *systematic* offset, which is exactly the
    signal that survives averaging.  The spread across trials is what separates
    the two, so the covariance is not a diagnostic afterthought -- the
    assignment energies weight by it.
    """
    device = physical_positions.device
    frames = list(int(value) for value in target_frames)
    n = int(state.canonical_fg.shape[0])
    total_frames = int(physical_positions.shape[0])
    generator = torch.Generator().manual_seed(int(config.seed))

    accumulated = torch.zeros((total_frames, n, 3), device=device)
    accumulated_outer = torch.zeros((total_frames, n, 3, 3), device=device)
    counts = torch.zeros(total_frames, device=device)
    trials = [] if keep_trials else None

    sigma = float(config.noise_sigma_ratio) * float(neighbour_scale)
    for trial in range(max(1, int(config.num_trials))):
        per_frame = torch.zeros((total_frames, n, 3), device=device)
        for frame in frames:
            proposal = physical_positions[frame]
            noise = torch.randn(proposal.shape, generator=generator, device="cpu").to(device)
            current = proposal + noise * sigma
            interval = max(0, min(frame - 1, state.fm_velocity.shape[0] - 1))
            fm = state.fm_velocity[interval]
            label = state.hard_labels
            killing = _killing_velocity_for_labels(state, interval, label)
            for _ in range(max(1, int(config.denoise_steps))):
                evidence = (
                    None if render_evidence_fn is None
                    else render_evidence_fn(current, frame)
                )
                confidence = (
                    None if observation_confidence_fn is None
                    else observation_confidence_fn(interval)
                )
                current = denoise_once(
                    denoiser, spec=spec, current_positions=current,
                    physical_positions=proposal, canonical_fg=state.canonical_fg,
                    fm_velocity=fm, killing_velocity=killing,
                    responsibilities=state.responsibilities,
                    assignment_entropy=state.assignment_entropy,
                    hard_labels=label, knn_idx=knn_idx,
                    neighbour_scale=neighbour_scale, step_ratio=config.step_ratio,
                    render_evidence=evidence, observation_confidence=confidence,
                )
            per_frame[frame] = current - proposal
        accumulated[frames] += per_frame[frames]
        accumulated_outer[frames] += torch.einsum(
            "fni,fnj->fnij", per_frame[frames], per_frame[frames]
        )
        counts[frames] += 1.0
        if trials is not None:
            trials.append(per_frame)

    divisor = counts.clamp_min(1.0)[:, None, None]
    mean = accumulated / divisor
    second = accumulated_outer / divisor[..., None]
    cov = second - torch.einsum("fni,fnj->fnij", mean, mean)
    # A single trial has no spread to measure; report zero covariance rather
    # than a negative number from the unbiased correction.
    cov = torch.where(counts[:, None, None, None] > 1, cov, torch.zeros_like(cov))

    spread = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1).clamp_min(0.0).sqrt()
    # Confidence falls as the trials disagree, measured against the noise the
    # trials were started from: a residual reproduced to within less than the
    # injected noise is a systematic finding, one that varies by more is not.
    confidence = torch.exp(-spread / max(sigma, _EPS))
    confidence = torch.where(counts[:, None] > 0, confidence, torch.zeros_like(confidence))

    return DiffusionResult(
        mean=mean,
        cov=cov,
        confidence=confidence,
        per_trial=torch.stack(trials) if trials else None,
        diagnostics={
            "target_frames": frames,
            "num_trials": int(config.num_trials),
            "denoise_steps": int(config.denoise_steps),
            "noise_sigma": sigma,
            "residual_norm_mean": float(mean[frames].norm(dim=-1).mean()),
            "trial_spread_mean": float(spread[frames].mean()),
            "confidence_mean": float(confidence[frames].mean()),
        },
    )


def _killing_velocity_for_labels(state, interval: int, labels: torch.Tensor) -> torch.Tensor:
    """The Killing velocity of whichever part each Gaussian is currently assigned.

    Static Gaussians get exactly zero, which is the static hypothesis, not a
    missing value.
    """
    out = torch.zeros_like(state.canonical_fg)
    for part in range(state.num_moving_parts):
        mask = labels == (part + 1)
        if bool(mask.any()):
            out[mask] = state.killing_velocity[part, interval][mask].to(out)
    return out


# ----------------------------------------------------------------------
# Residual decomposition (spec 26)
# ----------------------------------------------------------------------
@torch.no_grad()
def decompose_residual_against_field(
    residual_mean: torch.Tensor,
    *,
    canonical_fg: torch.Tensor,
    mode,
    responsibilities_for_part: torch.Tensor,
) -> dict:
    """Split the residual into components along and across the part's own field.

    The distinction diagnoses *which* thing is wrong, and they need different
    repairs:

        coherent parallel over the core     ->  speed or amplitude is wrong
        coherent orthogonal over the core   ->  the axis or geometry is wrong
        sparse orthogonal                   ->  those points are mislabelled

    ``phi`` is the unit-speed spatial field of the part, so the parallel share is
    the part of the correction that could be absorbed by changing how fast the
    part moves, and the orthogonal share is the part that could not.
    """
    if mode is None or not mode.valid:
        return {"available": False}
    points = torch.as_tensor(canonical_fg)
    reference = mode.reference_point.to(points)
    phi = mode.reference_velocity.to(points) + torch.cross(
        mode.angular_mode.to(points).expand_as(points), points - reference, dim=-1
    )
    norm_squared = (phi * phi).sum(-1, keepdim=True).clamp_min(_EPS)
    residual = torch.as_tensor(residual_mean).to(points)
    if residual.ndim == 3:
        residual = residual.mean(0)
    parallel = ((residual * phi).sum(-1, keepdim=True) / norm_squared) * phi
    orthogonal = residual - parallel
    weight = responsibilities_for_part.to(points).clamp(0.0, 1.0)
    total = weight.sum().clamp_min(_EPS)
    parallel_norm = parallel.norm(dim=-1)
    orthogonal_norm = orthogonal.norm(dim=-1)
    return {
        "available": True,
        "parallel": parallel,
        "orthogonal": orthogonal,
        "parallel_norm_mean": float((weight * parallel_norm).sum() / total),
        "orthogonal_norm_mean": float((weight * orthogonal_norm).sum() / total),
        # Above one means the correction is mostly along the field, i.e. a speed
        # error rather than a geometry error.
        "parallel_over_orthogonal": float(
            (weight * parallel_norm).sum() / (weight * orthogonal_norm).sum().clamp_min(_EPS)
        ),
        # A coherent orthogonal correction points the same way everywhere; a
        # mislabelling scatters.  This separates the two.
        "orthogonal_coherence": float(
            (weight[:, None] * orthogonal).sum(0).norm()
            / (weight * orthogonal_norm).sum().clamp_min(_EPS)
        ),
    }


@torch.no_grad()
def residual_speed_correction(
    residual_mean: torch.Tensor,
    *,
    canonical_fg: torch.Tensor,
    mode,
    responsibilities_for_part: torch.Tensor,
    frame_times,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Least-squares speed nudge implied by the parallel residual (spec 27).

    Projects the residual onto the part's own field, which gives the change in
    interval speed that would absorb it.  The caller applies it with a learning
    rate, or not at all: this is deliberately conservative because a residual
    caused by mislabelling would otherwise be laundered into a speed change.
    """
    if mode is None or not mode.valid:
        return torch.zeros(len(frame_times) - 1, dtype=torch.float64)
    points = torch.as_tensor(canonical_fg).double()
    reference = mode.reference_point.double()
    phi = mode.reference_velocity.double() + torch.cross(
        mode.angular_mode.double().expand_as(points), points - reference, dim=-1
    )
    weight = responsibilities_for_part.detach().cpu().double().clamp(0.0, 1.0).pow(float(gamma))
    residual = torch.as_tensor(residual_mean).detach().cpu().double()
    times = torch.as_tensor([float(v) for v in frame_times], dtype=torch.float64)
    deltas = torch.diff(times)
    corrections = []
    for interval in range(deltas.numel()):
        frame = interval + 1
        if frame >= residual.shape[0]:
            corrections.append(torch.zeros((), dtype=torch.float64))
            continue
        numerator = (weight * (residual[frame] * phi).sum(-1)).sum()
        denominator = (weight * (phi * phi).sum(-1)).sum().clamp_min(_EPS)
        # The residual is a displacement over the interval; the speed is the
        # displacement rate, hence the division by dt.
        corrections.append(numerator / denominator / deltas[interval].clamp_min(_EPS))
    return torch.stack(corrections)
