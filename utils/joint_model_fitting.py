"""Held-out multiview rendering competition for physical Stage-2 joints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.rigid_killing_motion import (
    RigidKillingMode,
    instantaneous_axis,
    integrate_rigid_mode_exact,
)
from utils.rigid_transform_utils import (
    direct_joint_transform,
    integrate_joint_transforms,
    transport_gaussian_state,
)
from utils.temporal_visibility_pretraining import _atomic_torch_save


def _unit(value):
    tensor = torch.as_tensor(value).detach().cpu().double().reshape(3)
    return tensor / torch.linalg.norm(tensor).clamp_min(1.0e-12)


def _finite_json(value):
    """Make a diagnostics record writable by a JSON dump that rejects NaN.

    The log-likelihood ratio is legitimately infinite when one model fits
    exactly, so the value is preserved as a string rather than replaced by a
    number that would read as a finite measurement.
    """
    if value is None:
        return None
    result = {}
    for key, item in value.items():
        if isinstance(item, float) and not math.isfinite(item):
            result[key] = "inf" if item > 0 else "-inf"
        elif torch.is_tensor(item):
            result[key] = item.tolist()
        else:
            result[key] = item
    return result


def _anchor_pivot(pivot, axis, part_points):
    """Slide a point along its axis to the place that describes the part.

    Every point on the line is the same joint, so the choice is free.  Anchoring
    at the moving part's centroid makes the reported pivot read as "the centre
    this part turns around" rather than "wherever the axis passes closest to the
    world origin", which is what it meant before and which moved with the scene.
    """
    anchor = (
        torch.zeros(3, dtype=torch.float64)
        if part_points is None
        else torch.as_tensor(part_points).detach().cpu().double().reshape(-1, 3).mean(dim=0)
    )
    return pivot - torch.dot(pivot - anchor, axis) * axis


def _fallback_rotation_axis(part_points):
    points = torch.as_tensor(part_points).detach().cpu().double()
    _, _, vh = torch.linalg.svd(points - points.mean(0), full_matrices=False)
    return _unit(vh[-1])


def fit_revolute_joint(mode, coarse_q, part_points=None):
    if mode.mode_type != "rotational":
        if part_points is None:
            raise ValueError("A translational initialization needs part_points")
        axis = _fallback_rotation_axis(part_points)
        points = torch.as_tensor(part_points).detach().cpu().double()
        pivot = points.mean(0)
        scale = torch.linalg.norm(points - pivot, dim=-1).median().clamp_min(1.0e-6)
        q = torch.as_tensor(coarse_q).detach().cpu().double()
        return {
            "joint_type": "revolute", "axis": axis, "pivot": pivot,
            "pitch": 0.0, "coarse_q": q / scale,
        }
    line = instantaneous_axis(mode)
    axis, omega_norm = line["axis"], line["angular_norm"]
    pivot = _anchor_pivot(line["pivot"], axis, part_points)
    return {
        "joint_type": "revolute",
        "axis": axis,
        "pivot": pivot,
        "pitch": 0.0,
        "coarse_q": torch.as_tensor(coarse_q).detach().cpu().double() * omega_norm,
    }


def fit_screw_joint(mode, coarse_q, part_points=None):
    if mode.mode_type != "rotational":
        revolute = fit_revolute_joint(mode, coarse_q, part_points)
        return {**revolute, "joint_type": "screw", "pitch": 0.0}
    line = instantaneous_axis(mode)
    axis, omega_norm, pitch = line["axis"], line["angular_norm"], line["pitch"]
    pivot = _anchor_pivot(line["pivot"], axis, part_points)
    return {
        "joint_type": "screw",
        "axis": axis,
        "pivot": pivot,
        "pitch": pitch,
        "coarse_q": torch.as_tensor(coarse_q).detach().cpu().double() * omega_norm,
    }


def fit_prismatic_joint(mode, coarse_q, part_points=None):
    if mode.mode_type == "translational":
        direction = _unit(mode.reference_velocity)
        scale = float(torch.linalg.norm(torch.as_tensor(mode.reference_velocity).double()))
        q = torch.as_tensor(coarse_q).detach().cpu().double() * scale
    else:
        if part_points is None:
            direction = _unit(mode.reference_velocity)
            q = torch.as_tensor(coarse_q).detach().cpu().double()
        else:
            points = torch.as_tensor(part_points).detach().cpu().double()
            center = points.mean(dim=0)
            centers = []
            for value in torch.as_tensor(coarse_q).detach().cpu().double():
                transformed = integrate_rigid_mode_exact(points, mode, value)["positions"]
                centers.append(transformed.mean(dim=0))
            displacement = torch.stack(centers) - center
            _, _, vh = torch.linalg.svd(displacement, full_matrices=False)
            direction = _unit(vh[0])
            q = displacement @ direction
    # A prismatic joint has no unique axis line -- every line parallel to the
    # motion describes the same joint -- so anchor it at the moving part's own
    # centroid instead of leaving it unlocated.
    pivot = (
        None
        if part_points is None
        else torch.as_tensor(part_points).detach().cpu().double().reshape(-1, 3).mean(dim=0)
    )
    return {
        "joint_type": "prismatic",
        "axis": direction,
        "pivot": pivot,
        "pitch": 0.0,
        "coarse_q": q,
    }


def joint_to_rigid_mode(joint):
    axis = torch.as_tensor(joint["axis"])
    if not axis.is_floating_point():
        axis = axis.float()
    axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-12)
    kind = str(joint["joint_type"])
    if kind == "prismatic":
        return RigidKillingMode(
            reference_point=torch.zeros(3, device=axis.device, dtype=axis.dtype),
            reference_velocity=axis,
            angular_mode=torch.zeros(3, device=axis.device, dtype=axis.dtype),
            mode_type="translational",
            normalization_scale=1.0,
            valid=True,
            diagnostics={"joint_type": kind},
        )
    pivot = torch.as_tensor(joint["pivot"], device=axis.device, dtype=axis.dtype)
    pitch = torch.as_tensor(
        joint.get("pitch", 0.0) if kind == "screw" else 0.0,
        device=axis.device,
        dtype=axis.dtype,
    )
    return RigidKillingMode(
        reference_point=pivot,
        reference_velocity=pitch * axis,
        angular_mode=axis,
        mode_type="rotational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={"joint_type": kind, "pitch": pitch},
    )


class JointModel(torch.nn.Module):
    """One trainable physical joint hypothesis and its delta-q trajectory."""

    joint_type = ""

    #: prismatic motion ignores the pivot entirely, so it is carried as a
    #: reported anchor rather than an optimizable parameter.
    pivot_is_geometry = True

    def __init__(self, initial, *, device, dtype, anchor=None):
        super().__init__()
        axis = _unit(initial["axis"]).to(device=device, dtype=dtype)
        self.axis_raw = torch.nn.Parameter(axis)
        pivot = initial.get("pivot")
        pivot_tensor = (
            None if pivot is None else torch.as_tensor(pivot, device=device, dtype=dtype).clone()
        )
        if pivot_tensor is None or not self.pivot_is_geometry:
            self.pivot_raw = None
            self.register_buffer(
                "pivot_fixed",
                torch.zeros(3, device=device, dtype=dtype) if pivot_tensor is None else pivot_tensor,
            )
            self.has_pivot = pivot_tensor is not None
        else:
            self.pivot_raw = torch.nn.Parameter(pivot_tensor)
            self.has_pivot = True
        # The point the reported pivot is slid to along the axis.  Any point on
        # the line describes the same joint, so the gauge is free; anchoring it
        # at the moving part makes the number mean "the centre this part turns
        # around" instead of "wherever the axis passes closest to the origin".
        self.register_buffer(
            "anchor",
            torch.zeros(3, device=device, dtype=dtype)
            if anchor is None
            else torch.as_tensor(anchor, device=device, dtype=dtype).clone(),
        )
        self.pitch_raw = (
            torch.nn.Parameter(torch.as_tensor(float(initial.get("pitch", 0.0)), device=device, dtype=dtype))
            if self.joint_type == "screw"
            else None
        )
        q = torch.as_tensor(initial["coarse_q"], device=device, dtype=dtype)
        self.delta_q = torch.nn.Parameter(torch.diff(q).clone())
        self.register_buffer("initial_axis", axis.clone())
        self.register_buffer(
            "initial_pivot",
            torch.zeros(3, device=device, dtype=dtype)
            if pivot is None
            else torch.as_tensor(pivot, device=device, dtype=dtype).clone(),
        )
        self.register_buffer(
            "initial_pitch",
            torch.as_tensor(float(initial.get("pitch", 0.0)), device=device, dtype=dtype),
        )

    @property
    def axis(self):
        return self.axis_raw / torch.linalg.norm(self.axis_raw).clamp_min(1.0e-12)

    @property
    def pivot(self):
        if not self.has_pivot:
            return None
        if self.pivot_raw is None:
            return self.pivot_fixed
        offset = self.pivot_raw - self.anchor
        return self.pivot_raw - torch.dot(offset, self.axis) * self.axis

    @property
    def pitch(self):
        return self.delta_q.new_zeros(()) if self.pitch_raw is None else self.pitch_raw

    @property
    def q(self):
        return torch.cat((self.delta_q.new_zeros(1), torch.cumsum(self.delta_q, dim=0)))

    def incremental_transform(self, delta_q):
        from utils.rigid_transform_utils import joint_incremental_transform
        return joint_incremental_transform(
            self.joint_type, self.axis, delta_q, pivot=self.pivot, pitch=self.pitch
        )

    def integrate(self, delta_q=None):
        return integrate_joint_transforms(
            self.joint_type,
            self.axis,
            self.delta_q if delta_q is None else delta_q,
            pivot=self.pivot,
            pitch=self.pitch,
        )

    def regularization(self):
        temporal = (
            torch.diff(self.delta_q).square().mean()
            if self.delta_q.numel() > 1
            else self.delta_q.new_zeros(())
        )
        return {"trajectory_acceleration": temporal}

    def set_trainable(self, block):
        geometry = block in ("geometry", "joint")
        trajectory = block in ("trajectory", "joint")
        self.axis_raw.requires_grad_(geometry)
        if self.pivot_raw is not None:
            self.pivot_raw.requires_grad_(geometry)
        if self.pitch_raw is not None:
            self.pitch_raw.requires_grad_(geometry)
        self.delta_q.requires_grad_(trajectory)

    def geometry_change(self):
        axis_change = torch.minimum(
            torch.linalg.norm(self.axis - self.initial_axis),
            torch.linalg.norm(self.axis + self.initial_axis),
        )
        pivot_change = self.axis.new_zeros(()) if self.pivot is None else torch.linalg.norm(self.pivot - self.initial_pivot)
        pitch_change = (self.pitch - self.initial_pitch).abs()
        return axis_change + pivot_change + pitch_change

    def joint_dict(self):
        return {
            "joint_type": self.joint_type,
            "axis": self.axis.detach().cpu().double(),
            "pivot": None if self.pivot is None else self.pivot.detach().cpu().double(),
            "pitch": float(self.pitch.detach()),
            "coarse_q": self.q.detach().cpu().double(),
        }


class RevoluteJointModel(JointModel):
    joint_type = "revolute"


class PrismaticJointModel(JointModel):
    joint_type = "prismatic"
    pivot_is_geometry = False


class ScrewJointModel(JointModel):
    joint_type = "screw"


def _make_joint_model(candidate, points, anchor=None):
    classes = {
        "revolute": RevoluteJointModel,
        "prismatic": PrismaticJointModel,
        "screw": ScrewJointModel,
    }
    return classes[str(candidate["joint_type"])](
        candidate, device=points.device, dtype=points.dtype, anchor=anchor
    )


def infer_joint_parent_child(num_moving_parts, topology):
    if str(topology) == "auto" and int(num_moving_parts) > 1:
        # ponytail: add moving-to-moving parent inference only with a validated
        # relative-transform benchmark; silently inventing a tree is worse.
        raise NotImplementedError(
            "Automatic moving-to-moving topology inference needs validated relative-transform evidence; use --joint_topology known for the current static-base graph"
        )
    return [
        {"joint_id": part - 1, "parent_part": 0, "child_part": part}
        for part in range(1, int(num_moving_parts) + 1)
    ]


def _candidate_scene_states(model, part_index, stage1, points, rotations, labels, frames):
    candidate = model.integrate()
    states = {}
    for frame in sorted(set(int(item) for item in frames)):
        xyz, orientation = points, F.normalize(rotations, dim=-1)
        for current_part, (mode, coordinate) in enumerate(
            zip(stage1["modes"], stage1["coordinates"]), start=1
        ):
            mask = labels == current_part
            if current_part == int(part_index):
                moved, rotated = transport_gaussian_state(
                    points,
                    rotations,
                    candidate["rotations"][frame],
                    candidate["translations"][frame],
                    candidate["rotation_quaternions"][frame],
                )
            else:
                result = integrate_rigid_mode_exact(
                    points,
                    mode.to(device=points.device, dtype=points.dtype),
                    torch.as_tensor(coordinate[frame]).to(points),
                    orientations=rotations,
                )
                moved, rotated = result["positions"], result["orientations"]
            xyz = torch.where(mask[:, None], moved, xyz)
            orientation = torch.where(mask[:, None], rotated, orientation)
        states[frame] = (xyz, F.normalize(orientation, dim=-1))
    return states, candidate


def _render_candidate_view(
    camera, gaussians, pipe, background, dataset, args, opt, positions, orientations
):
    from gaussian_renderer import render
    from utils.loss_utils import _load_mask_for_view, dino_feature_field_loss, ssim

    # The candidate geometry goes straight into the rasteriser.  The older route
    # through ``set_free_working_state`` needed the legacy dynamic-motion module
    # to be live -- and it mutated the model, so a caller that had not
    # initialised that module either crashed or, worse, silently rendered the
    # canonical geometry for every candidate, making every candidate score the
    # same.  Overriding the rasteriser inputs has neither problem.
    package = render(
        camera, gaussians, pipe, background,
        use_trained_exp=bool(dataset.train_test_exp),
        means3D_override=positions,
        rotations_override=orientations,
    )
    image = package["render"]
    target = camera.original_image.to(device=image.device, dtype=image.dtype)
    height, width = target.shape[-2:]
    mask = _load_mask_for_view(camera, dataset, args, (height, width), device=image.device)
    if mask is None:
        raise RuntimeError("Joint-model rendering requires every dynamic foreground mask")
    mask = (mask > 0.5).to(image.dtype)
    if camera.alpha_mask is not None:
        alpha = camera.alpha_mask.to(device=image.device, dtype=image.dtype)
        mask = mask * (alpha[0] if alpha.ndim == 3 else alpha).gt(0.5).to(mask.dtype)
    mask3 = mask[None].expand_as(image)
    rgb_l1 = (torch.abs(image - target) * mask3).sum() / mask3.sum().clamp_min(1.0)
    rgb_dssim = 1.0 - ssim(image * mask3, target * mask3)
    rgb = (1.0 - float(opt.lambda_dssim)) * rgb_l1 + float(opt.lambda_dssim) * rgb_dssim

    depth = rgb.new_zeros(())
    if bool(getattr(camera, "depth_reliable", False)):
        target_depth = camera.invdepthmap.to(device=image.device, dtype=image.dtype)
        depth_mask = camera.depth_mask.to(device=image.device, dtype=image.dtype)
        if depth_mask.ndim == 3:
            depth_mask = depth_mask[0]
        valid = depth_mask * mask
        depth = (torch.abs(package["depth"] - target_depth) * valid).sum() / valid.sum().clamp_min(1.0)

    dino = rgb.new_zeros(())
    dino_weight = float(getattr(args, "lambda_dino_feature_field_dynamic", 0.0))
    if bool(getattr(args, "use_dino_feature_field", False)) and dino_weight > 0.0:
        dino_loss, dino_stats = dino_feature_field_loss(
            camera, gaussians, pipe, background, dataset, args, stage="joint_compare"
        )
        if dino_loss is None:
            raise RuntimeError(
                f"Joint candidate DINO target is unavailable for {getattr(camera, 'image_name', '<unknown>')}: {dino_stats}"
            )
        dino = dino_loss
    depth_weight = float(getattr(opt, "depth_l1_weight_final", 0.0))
    total = rgb + depth_weight * depth + dino_weight * dino
    return {"total": total, "rgb": rgb, "depth": depth, "dino": dino}


def render_candidate_sequence(
    model,
    part_index,
    stage1,
    points,
    rotations,
    labels,
    pairs,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    *,
    detailed=False,
):
    states, trajectory = _candidate_scene_states(
        model, part_index, stage1, points, rotations, labels, [item[0] for item in pairs]
    )
    sums = {key: model.delta_q.new_zeros(()) for key in ("total", "rgb", "depth", "dino")}
    camera_rows, frame_rows = {}, {}
    for frame, camera_key, camera in pairs:
        losses = _render_candidate_view(
            camera, gaussians, pipe, background, dataset, args, opt, *states[int(frame)]
        )
        for key in sums:
            sums[key] = sums[key] + losses[key]
        if detailed:
            camera_rows.setdefault(str(camera_key), []).append({key: float(value.detach()) for key, value in losses.items()})
            frame_rows.setdefault(str(frame), []).append({key: float(value.detach()) for key, value in losses.items()})
    count = max(1, len(pairs))
    means = {key: value / count for key, value in sums.items()}
    if detailed:
        def average(rows):
            return {
                name: {key: sum(item[key] for item in values) / len(values) for key in sums}
                for name, values in rows.items()
            }
        means["per_camera"] = average(camera_rows)
        means["per_frame"] = average(frame_rows)
    return means, trajectory


def _fit_rendered_candidate(
    initial,
    part_index,
    stage1,
    points,
    rotations,
    labels,
    train_pairs,
    validation_pairs,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    *,
    log_fn=None,
):
    part_points = points[labels == int(part_index)]
    model = _make_joint_model(
        initial, points, anchor=part_points.mean(dim=0) if part_points.numel() else None
    )
    total_iterations = max(3, int(args.joint_compare_iters))
    if bool(getattr(args, "joint_compare_refit_geometry", False)):
        blocks = (
            ("trajectory", max(1, int(round(0.4 * total_iterations)))),
            ("geometry", max(1, int(round(0.3 * total_iterations)))),
        )
        blocks += (("trajectory", max(1, total_iterations - sum(item[1] for item in blocks))),)
    else:
        # Stage 2 labels; it does not re-fit geometry.  The axis, pivot and
        # pitch stay exactly as the Stage-1 rigid field left them, so the
        # comparison isolates which joint model explains the held-out renders.
        blocks = (("trajectory", total_iterations),)
    history, global_step = [], 0
    batch_size = min(len(train_pairs), max(1, int(args.joint_compare_batch_size)))
    batch_generator = torch.Generator().manual_seed(20_260_825 + int(part_index))
    batch_order = torch.randperm(len(train_pairs), generator=batch_generator).tolist()
    progress = tqdm(total=sum(item[1] for item in blocks), desc=f"Stage 2 part {part_index} {model.joint_type}", unit="iter")
    for block, iterations in blocks:
        model.set_trainable(block)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        learning_rate = (
            float(args.joint_compare_geometry_lr)
            if block == "geometry"
            else float(args.joint_compare_trajectory_lr)
        )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(iterations):
            start = (global_step * batch_size) % len(batch_order)
            batch = [
                train_pairs[batch_order[(start + index) % len(batch_order)]]
                for index in range(batch_size)
            ]
            optimizer.zero_grad(set_to_none=True)
            losses, _ = render_candidate_sequence(
                model, part_index, stage1, points, rotations, labels, batch,
                gaussians, pipe, background, dataset, args, opt,
            )
            temporal = model.regularization()["trajectory_acceleration"]
            objective = losses["total"] + 1.0e-4 * temporal
            if not bool(torch.isfinite(objective)):
                raise RuntimeError(f"Non-finite {model.joint_type} rendering objective")
            objective.backward()
            optimizer.step()
            global_step += 1
            row = {
                "iteration": global_step, "block": block,
                **{key: float(value.detach()) for key, value in losses.items()},
                "temporal": float(temporal.detach()),
            }
            history.append(row)
            progress.update(1)
            progress.set_postfix(total=f"{row['total']:.4f}")
            if log_fn is not None:
                prefix = f"joint_compare/{model.joint_type}"
                log_fn("joint_compare", global_step, {
                    f"{prefix}/part": part_index,
                    f"{prefix}/train_total": row["total"],
                    f"{prefix}/train_rgb": row["rgb"],
                    f"{prefix}/train_depth": row["depth"],
                    f"{prefix}/train_dino": row["dino"],
                })
    progress.close()
    model.set_trainable("none")
    with torch.no_grad():
        train, trajectory = render_candidate_sequence(
            model, part_index, stage1, points, rotations, labels, train_pairs,
            gaussians, pipe, background, dataset, args, opt, detailed=True,
        )
        validation, _ = render_candidate_sequence(
            model, part_index, stage1, points, rotations, labels, validation_pairs,
            gaussians, pipe, background, dataset, args, opt, detailed=True,
        )
        direct_rotations, direct_translations = [], []
        current_joint = {
            "joint_type": model.joint_type,
            "axis": model.axis,
            "pivot": model.pivot,
            "pitch": model.pitch,
        }
        for coordinate in model.q:
            direct_rotation, direct_translation, _ = direct_joint_transform(
                current_joint, coordinate
            )
            direct_rotations.append(direct_rotation)
            direct_translations.append(direct_translation)
        direct_rotation_error = torch.linalg.norm(
            trajectory["rotations"] - torch.stack(direct_rotations), dim=(-2, -1)
        ).max()
        direct_translation_error = torch.linalg.norm(
            trajectory["translations"] - torch.stack(direct_translations), dim=-1
        ).max()
        incremental_direct_error = float(torch.maximum(direct_rotation_error, direct_translation_error))
        if incremental_direct_error > 1.0e-5:
            raise RuntimeError(
                f"{model.joint_type} incremental/direct transform mismatch: {incremental_direct_error:.3e}"
            )
    payload = {
        "schema_version": 2,
        "part": int(part_index),
        "joint_parameters": model.joint_dict(),
        "delta_q": model.delta_q.detach().cpu().double(),
        "q": model.q.detach().cpu().double(),
        "integrated_transforms": {
            key: value.detach().cpu().double() for key, value in trajectory.items()
            if key in ("rotations", "translations", "rotation_quaternions")
        },
        "train": _losses_to_python(train),
        "validation": _losses_to_python(validation),
        "train_cameras": sorted({str(item[1]) for item in train_pairs}),
        "validation_cameras": sorted({str(item[1]) for item in validation_pairs}),
        "train_validation_camera_overlap": False,
        "optimization_batch_seed": 20_260_825 + int(part_index),
        "full_train_and_validation_scoring_after_fit": True,
        "frame_indices": sorted({int(item[0]) for item in train_pairs + validation_pairs}),
        "history": history,
        "loss_weights": {
            "rgb_l1": 1.0 - float(opt.lambda_dssim),
            "rgb_dssim": float(opt.lambda_dssim),
            "depth": float(getattr(opt, "depth_l1_weight_final", 0.0)),
            "dino": float(getattr(args, "lambda_dino_feature_field_dynamic", 0.0)),
        },
        "geometry_change": float(model.geometry_change().detach()),
        "geometry_changed_from_stage1": float(model.geometry_change().detach()) > 1.0e-8,
        "incremental_direct_transform_error_max": incremental_direct_error,
        "converged": len(history) > 1 and abs(history[-1]["total"] - history[-2]["total"]) < 1.0e-4,
        "ground_truth_used_as_algorithm_input": False,
    }
    return model, payload


def _losses_to_python(losses):
    result = {}
    for key, value in losses.items():
        if isinstance(value, dict):
            result[key] = value
        else:
            result[key] = float(value.detach())
    return result


def _select_candidate(payloads, screw_margin):
    ordered = sorted(payloads, key=lambda item: float(item["validation"]["total"]))
    selected = ordered[0]
    by_type = {item["joint_parameters"]["joint_type"]: item for item in payloads}
    tie_break = None
    if selected["joint_parameters"]["joint_type"] == "screw" and "revolute" in by_type:
        screw = float(selected["validation"]["total"])
        revolute = float(by_type["revolute"]["validation"]["total"])
        relative_gain = (revolute - screw) / max(abs(revolute), 1.0e-12)
        if relative_gain < float(screw_margin):
            selected = by_type["revolute"]
            tie_break = f"revolute preferred because screw held-out gain {relative_gain:.6g} < {float(screw_margin):.6g}"
    values = [float(item["validation"]["total"]) for item in payloads]
    return selected, tie_break, max(values) - min(values) < 1.0e-10


def _joint_candidates_for_args(args, mode, coarse_q, part_points):
    requested = str(getattr(args, "joint_types", "auto"))
    names = ("revolute", "prismatic") if requested == "auto" else (requested,)
    builders = {
        "revolute": fit_revolute_joint,
        "prismatic": fit_prismatic_joint,
        "screw": fit_screw_joint,
    }
    return tuple(builders[name](mode, coarse_q, part_points) for name in names)


def fit_joint_graph(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    opt,
    stage1,
    args,
    output_dir,
    *,
    frame_selection=None,
    log_fn=None,
):
    """Label each part's joint from its Stage-1 field plus held-out rendering."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from utils.unified_dynamic_training import (
        _camera_map,
        _complete_frame_groups,
        _ordered_dynamic_cameras,
        resolve_joint_frame_selection,
        subset_stage1_coordinates,
    )

    points = gaussians.get_canonical_xyz.detach()
    rotations = gaussians.get_rotation.detach().clone()
    labels = torch.as_tensor(stage1["labels"], device=points.device).long()
    graph = infer_joint_parent_child(len(stage1["modes"]), args.joint_topology)
    all_groups = _complete_frame_groups(_ordered_dynamic_cameras(scene.getTrainCameras()), args)
    if frame_selection is None:
        frame_selection = resolve_joint_frame_selection(all_groups, args)
    groups = frame_selection["groups"]
    # Every frame index below is a position in the selected subset, and the
    # Stage-1 profile is resampled to match, so a sparse selection stays
    # self-consistent through integration and rendering.
    coordinates = subset_stage1_coordinates(stage1, frame_selection["positions"])
    stage1 = {**stage1, "coordinates": coordinates}
    if frame_selection["is_sparse"]:
        print(
            f"[Stage 2] frame subset '{frame_selection['spec']}': "
            f"{frame_selection['selected_frames']}/{frame_selection['available_frames']} frames, "
            f"max gap {frame_selection['max_gap_seconds']:.3f}s"
        )
    camera_maps = [_camera_map(item[2]) for item in groups]
    camera_keys = sorted(set(camera_maps[0]).intersection(*[set(item) for item in camera_maps[1:]]))
    if len(camera_keys) < 2:
        raise RuntimeError("Stage-2 held-out rendering competition requires at least two physical cameras")
    validation_count = min(max(1, int(args.joint_compare_validation_cameras)), len(camera_keys) - 1)
    train_keys, validation_keys = camera_keys[:-validation_count], camera_keys[-validation_count:]
    if set(train_keys) & set(validation_keys):
        raise RuntimeError("Stage-2 train and held-out camera sets overlap")
    train_pairs = [(frame, key, camera_maps[frame][key]) for frame in range(1, len(groups)) for key in train_keys]
    validation_pairs = [(frame, key, camera_maps[frame][key]) for frame in range(1, len(groups)) for key in validation_keys]
    # Stage 1 fitted both the rigid and the translation-only field for every
    # part; keep them so the selection below can compare the two hypotheses on
    # the velocity they were fitted to.
    stage1_fields = list(stage1.get("velocity_fields") or [])
    joints, summaries, all_payloads = [], [], []
    for part_index, (mode, coarse_q) in enumerate(zip(stage1["modes"], coordinates), start=1):
        part_points = points[labels == part_index]
        if part_points.shape[0] < 3:
            raise RuntimeError(f"Stage 2 part {part_index} has fewer than three confident Gaussians")
        candidates = _joint_candidates_for_args(args, mode, coarse_q, part_points)
        fitted = []
        part_step_span = len(candidates) * int(args.joint_compare_iters) + 1
        for candidate_index, candidate in enumerate(candidates):
            candidate_log = (
                None
                if log_fn is None
                else lambda stage, local, stats, offset=(part_index - 1) * part_step_span + candidate_index * int(args.joint_compare_iters): log_fn(
                    stage, offset + local, stats
                )
            )
            model, payload = _fit_rendered_candidate(
                candidate, part_index, stage1, points, rotations, labels,
                train_pairs, validation_pairs, gaussians, pipe, background,
                dataset, args, opt, log_fn=candidate_log,
            )
            del model
            fitted.append(payload)
        revolute_validation = next(
            float(item["validation"]["total"])
            for item in fitted
            if item["joint_parameters"]["joint_type"] == "revolute"
        )
        for payload in fitted:
            parameter = payload["joint_parameters"]
            kind = parameter["joint_type"]
            total_q = abs(float(payload["q"][-1]))
            axis = torch.as_tensor(parameter["axis"], device=part_points.device, dtype=part_points.dtype)
            if parameter["pivot"] is None:
                radial_median = 0.0
                radial_consistency = None
            else:
                pivot = torch.as_tensor(parameter["pivot"], device=part_points.device, dtype=part_points.dtype)
                relative = part_points - pivot
                radial = torch.linalg.norm(relative - (relative @ axis)[:, None] * axis, dim=-1)
                radial_median = float(radial.median())
                radial_consistency = float(radial.std(unbiased=False) / radial.mean().clamp_min(1.0e-8))
            diagnostics = {
                "axis": parameter["axis"].tolist(),
                "total_displacement": total_q,
                "validation_improvement_over_revolute": (
                    revolute_validation - float(payload["validation"]["total"])
                ),
            }
            if kind == "revolute":
                diagnostics["radial_consistency_coefficient"] = radial_consistency
            elif kind == "prismatic":
                coarse_coordinate = torch.as_tensor(coarse_q).detach().cpu().double()
                angular_norm = float(torch.linalg.norm(torch.as_tensor(mode.angular_mode).double()))
                rotation_residual = float(
                    torch.sqrt((coarse_coordinate * angular_norm).square().mean())
                )
                diagnostics["rotation_residual"] = rotation_residual
                diagnostics["rotation_residual_radians_rms"] = rotation_residual
            else:
                axial = abs(float(parameter["pitch"])) * total_q
                diagnostics.update({
                    "pitch": float(parameter["pitch"]),
                    "total_axial_displacement": axial,
                    "axial_to_rotational_displacement_ratio": axial / max(radial_median * total_q, 1.0e-12),
                    "pitch_stability_across_camera_subsets": None,
                    "pitch_stability_note": "single deterministic 6/2-style camera split; cross-validation is optional",
                })
            payload["physical_diagnostics"] = diagnostics
            _atomic_torch_save(payload, output / f"phase2_part_{part_index}_{kind}_candidate.pt")
            if len(stage1["modes"]) == 1:
                _atomic_torch_save(payload, output / f"phase2_{kind}_candidate.pt")
        rendered, tie_break, degenerate = _select_candidate(
            fitted, args.joint_compare_screw_margin
        )
        scores = {item["joint_parameters"]["joint_type"]: float(item["validation"]["total"]) for item in fitted}
        ordered_scores = sorted(scores.values())
        gap = ordered_scores[1] - ordered_scores[0]
        render_margin = min(1.0, max(0.0, gap / max(abs(ordered_scores[1]), 1.0e-8)))
        by_type = {item["joint_parameters"]["joint_type"]: item for item in fitted}
        # Both joint types are fitted and the one the data supports is selected.
        # The decision comes from the velocity the fields were fitted to: Stage 1
        # fits the rigid field and the translation-only field on identical
        # observations and weights, and their residuals are compared with the
        # nested-model penalty for the rigid generator's three extra degrees of
        # freedom.  Nothing in it is a tuned constant -- see
        # compare_rigid_and_translation_evidence.  The held-out rendering
        # competition still fits both candidates and is reported beside it as an
        # independent check, with disagreement flagged.
        field = stage1_fields[part_index - 1] if part_index - 1 < len(stage1_fields) else None
        evidence = (field or {}).get("joint_type_evidence")
        render_preference = 0.0
        if "revolute" in by_type and "prismatic" in by_type:
            revolute_loss = float(by_type["revolute"]["validation"]["total"])
            prismatic_loss = float(by_type["prismatic"]["validation"]["total"])
            denominator = max(abs(revolute_loss), abs(prismatic_loss), 1.0e-12)
            render_preference = (prismatic_loss - revolute_loss) / denominator
        render_choice = "revolute" if render_preference > 0.0 else "prismatic"
        # ``joint_type`` is None only when there were no valid velocity
        # observations to compare, which is an absence of evidence rather than a
        # verdict; the fallback is named so the artefact says which happened.
        velocity_type = None if evidence is None else evidence.get("joint_type")
        if velocity_type is not None and velocity_type in by_type:
            selected = by_type[velocity_type]
            decided_by = "velocity_residual_model_comparison"
        else:
            selected = rendered
            decided_by = (
                "held_out_rendering_no_velocity_evidence"
                if velocity_type is None
                else "held_out_rendering"
            )
        sources_agree = selected["joint_parameters"]["joint_type"] == render_choice
        selected_parameters = selected["joint_parameters"]
        joint = {
            **graph[part_index - 1],
            **selected_parameters,
            "coarse_q": selected["q"],
            "selected_delta_q": selected["delta_q"],
            "selected_transform_sequence": selected["integrated_transforms"],
            # Posterior probability of the selected type under equal priors,
            # from the same log-evidence difference that made the decision.
            # None when no velocity evidence existed: there is no number to
            # report, and inventing one would hide that.
            "confidence": (
                None
                if evidence is None or evidence.get("revolute_probability") is None
                else float(
                    evidence["revolute_probability"]
                    if selected["joint_parameters"]["joint_type"] == "revolute"
                    else 1.0 - evidence["revolute_probability"]
                )
            ),
            "joint_type_evidence": evidence,
            "render_preference": float(render_preference),
            "render_choice": render_choice,
            "render_margin": float(render_margin),
            "sources_agree": bool(sources_agree),
            "decided_by": decided_by,
            "rendered_type": rendered["joint_parameters"]["joint_type"],
            "candidate_scores": scores,
        }
        joints.append(joint)
        summaries.append({
            "joint_id": part_index - 1,
            "child_part": part_index,
            "joint_type": joint["joint_type"],
            "confidence": (
                None if joint["confidence"] is None else float(joint["confidence"])
            ),
            "joint_type_evidence": _finite_json(joint["joint_type_evidence"]),
            "render_preference": float(joint["render_preference"]),
            "render_choice": joint["render_choice"],
            "render_margin": float(joint["render_margin"]),
            "sources_agree": bool(joint["sources_agree"]),
            "decided_by": joint["decided_by"],
            "rendered_type": joint["rendered_type"],
            "axis": joint["axis"].tolist(),
            "pivot": None if joint["pivot"] is None else joint["pivot"].tolist(),
            "pitch": float(joint["pitch"]),
            "candidate_scores": joint["candidate_scores"],
            "tie_break": tie_break,
            "all_candidate_scores_degenerate": degenerate,
            "geometry_changed": {item["joint_parameters"]["joint_type"]: item["geometry_changed_from_stage1"] for item in fitted},
            "failure_flags": (
                (["all_candidate_losses_numerically_identical"] if degenerate else [])
                + ([] if bool(getattr(args, "joint_compare_refit_geometry", False)) or all(
                    item["geometry_changed_from_stage1"] for item in fitted
                ) else ["one_or_more_candidate_geometries_never_changed"])
                + ([] if sources_agree else ["velocity_and_render_evidence_disagree"])
            ),
        })
        all_payloads.append({"part": part_index, "candidates": fitted, "selected": joint["joint_type"], "tie_break": tie_break, "degenerate": degenerate})
        if log_fn is not None:
            type_id = {"revolute": 0, "prismatic": 1, "screw": 2}[joint["joint_type"]]
            selected_stats = {
                "joint_selected/part": part_index,
                "joint_selected/type": type_id,
                "joint_selected/type_id": type_id,
                "joint_selected/confidence": float(joint["confidence"] or 0.0),
                "joint_selected/pitch": float(joint["pitch"]),
            }
            for payload in fitted:
                kind = payload["joint_parameters"]["joint_type"]
                prefix = f"joint_compare/{kind}"
                selected_stats.update({
                    f"{prefix}/val_total": payload["validation"]["total"],
                    f"{prefix}/val_rgb": payload["validation"]["rgb"],
                    f"{prefix}/val_depth": payload["validation"]["depth"],
                    f"{prefix}/val_dino": payload["validation"]["dino"],
                    f"{prefix}/geometry_change": payload["geometry_change"],
                    f"{prefix}/pitch": payload["joint_parameters"]["pitch"],
                })
            log_fn(
                "joint_selected",
                part_index * part_step_span,
                selected_stats,
            )

    serializable_graph = {
        "schema_version": 1,
        "topology_policy": str(args.joint_topology),
        "edges": [
            {key: value for key, value in item.items() if key in ("joint_id", "parent_part", "child_part")}
            for item in joints
        ],
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "phase2_joint_graph.json").write_text(json.dumps(serializable_graph, indent=2), encoding="utf-8")
    selected_payload = {
        "schema_version": 2, "joints": joints,
        "selected_joint_type": [item["joint_type"] for item in joints],
        "selected_joint_parameters": joints,
        "selected_delta_q": [item["selected_delta_q"] for item in joints],
        "selected_q": [item["coarse_q"] for item in joints],
        "selected_transform_sequence": [item["selected_transform_sequence"] for item in joints],
        "ground_truth_used_as_algorithm_input": False,
    }
    _atomic_torch_save({"schema_version": 2, "joints": joints, "ground_truth_used_as_algorithm_input": False}, output / "phase2_joint_models.pt")
    _atomic_torch_save(selected_payload, output / "phase2_selected_joint.pt")
    _atomic_torch_save(selected_payload, output / "phase2_selected_joint_model.pt")
    summary = {
        "schema_version": 2,
        "joint_count": len(joints),
        "selection_evidence": "both_joint_types_fitted_then_selected_by_nested_model_velocity_evidence_with_held_out_rendering_reported",
        "geometry_refit_enabled": bool(getattr(args, "joint_compare_refit_geometry", False)),
        "train_cameras": train_keys,
        "validation_cameras": validation_keys,
        "camera_sets_overlap": False,
        "frames": len(groups),
        "frame_selection": {
            key: value
            for key, value in frame_selection.items()
            if key not in ("groups",)
        },
        "loss_weights": {
            "rgb_l1": 1.0 - float(opt.lambda_dssim),
            "rgb_dssim": float(opt.lambda_dssim),
            "depth": float(getattr(opt, "depth_l1_weight_final", 0.0)),
            "dino": float(getattr(args, "lambda_dino_feature_field_dynamic", 0.0)),
        },
        "joints": summaries,
        "ground_truth_used_as_algorithm_input": False,
    }
    (output / "phase2_joint_fit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    comparison = {**summary, "parts": [_comparison_json(item) for item in all_payloads]}
    for name in ("phase2_joint_comparison.json", "phase2_joint_model_comparison.json"):
        (output / name).write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    _write_comparison_html(all_payloads, output / "phase2_joint_comparison.html")
    _write_loss_curves(all_payloads, output / "phase2_candidate_loss_curves.png")
    _write_joint_debug(points, labels, joints, output / "phase2_joint_debug.html")
    return {"graph": serializable_graph, "joints": joints, "summary": summary, "comparison": all_payloads}


def _comparison_json(item):
    return {
        "part": item["part"], "selected": item["selected"],
        "tie_break": item["tie_break"], "all_candidate_scores_degenerate": item["degenerate"],
        "candidates": [{
            "joint_type": candidate["joint_parameters"]["joint_type"],
            "axis": candidate["joint_parameters"]["axis"].tolist(),
            "pivot": None if candidate["joint_parameters"]["pivot"] is None else candidate["joint_parameters"]["pivot"].tolist(),
            "pitch": candidate["joint_parameters"]["pitch"],
            "total_displacement": float(candidate["q"][-1]),
            "train": candidate["train"], "validation": candidate["validation"],
            "geometry_change": candidate["geometry_change"],
            "geometry_changed_from_stage1": candidate["geometry_changed_from_stage1"],
            "physical_diagnostics": candidate["physical_diagnostics"],
            "converged": candidate["converged"],
        } for candidate in item["candidates"]],
    }


def _write_comparison_html(parts, output):
    rows = []
    for part in parts:
        for candidate in part["candidates"]:
            train, validation = candidate["train"], candidate["validation"]
            kind = candidate["joint_parameters"]["joint_type"]
            selected = "yes" if kind == part["selected"] else ""
            values = [part["part"], kind, train["rgb"], train["depth"], train["dino"], validation["rgb"], validation["depth"], validation["dino"], validation["total"], selected]
            rows.append("<tr>" + "".join(f"<td>{value:.7g}</td>" if isinstance(value, float) else f"<td>{value}</td>" for value in values) + "</tr>")
    document = """<!doctype html><html><head><meta charset='utf-8'><style>body{font-family:sans-serif}table{border-collapse:collapse}th,td{padding:6px 10px;border:1px solid #aaa}</style></head><body><h2>Stage 2 held-out rendering competition</h2><table><thead><tr><th>Part</th><th>Candidate</th><th>Train RGB</th><th>Train depth</th><th>Train DINO</th><th>Val RGB</th><th>Val depth</th><th>Val DINO</th><th>Val total</th><th>Selected</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table></body></html>"
    Path(output).write_text(document, encoding="utf-8")


def _write_loss_curves(parts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for part in parts:
        for candidate in part["candidates"]:
            history = candidate["history"]
            axis.plot([item["iteration"] for item in history], [item["total"] for item in history], label=f"part {part['part']} {candidate['joint_parameters']['joint_type']}")
    axis.set_xlabel("candidate optimization iteration")
    axis.set_ylabel("RGB + depth + DINO")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _write_joint_debug(points, labels, joints, output):
    xyz = torch.as_tensor(points).cpu().float()
    label = torch.as_tensor(labels).cpu().long()
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"]
    traces = []
    for part in sorted(int(item) for item in label.unique().tolist() if int(item) >= 0):
        selected = xyz[label == part]
        traces.append({
            "type": "scatter3d", "mode": "markers", "name": f"part {part}",
            "x": selected[:, 0].tolist(), "y": selected[:, 1].tolist(), "z": selected[:, 2].tolist(),
            "marker": {"size": 2.2, "color": colors[part % len(colors)]},
        })
    extent = float(torch.linalg.norm(xyz.amax(dim=0) - xyz.amin(dim=0)).clamp_min(1.0e-6))
    for joint in joints:
        origin = torch.zeros(3) if joint["pivot"] is None else torch.as_tensor(joint["pivot"]).float()
        axis = torch.as_tensor(joint["axis"]).float()
        start, end = origin - 0.6 * extent * axis, origin + 0.6 * extent * axis
        traces.append({
            "type": "scatter3d", "mode": "lines", "name": f"joint {joint['joint_id']} {joint['joint_type']}",
            "x": [float(start[0]), float(end[0])], "y": [float(start[1]), float(end[1])], "z": [float(start[2]), float(end[2])],
            "line": {"width": 8, "color": "#111111"},
        })
    document = f"""<!doctype html><html><head><meta charset='utf-8'><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><div id='plot' style='width:100vw;height:100vh'></div><script>Plotly.newPlot('plot',{json.dumps(traces)},{json.dumps({'title': 'Stage 2 fitted joint graph', 'scene': {'aspectmode': 'data'}, 'margin': {'l': 0, 'r': 0, 't': 45, 'b': 0}})},{{responsive:true}});</script></body></html>"""
    Path(output).write_text(document, encoding="utf-8")
