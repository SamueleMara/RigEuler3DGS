"""Exact semantic-foreground joint initialization and trainable part field.

Contract
--------
* The global Eulerian proposal velocity field remains defined for every Gaussian.
* This module sees only the semantic foreground selected from the frozen static
  semantic field.
* Every selected foreground Gaussian contributes its canonical XYZ and its own
  frozen statically trained DINO descriptor to the complete pairwise matrix.
* The trainable joint field contains exactly one vector per semantic-foreground
  Gaussian and exactly two slots for the current problem: foreground-static and
  one foreground-moving rigid part.
* Full-scene helper tensors map non-foreground Gaussians to dynamic probability 0
  for bookkeeping only. They must still use proposal velocity in transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class JointInitializationResult:
    foreground_ids: torch.Tensor
    combined_distance: torch.Tensor
    initial_cluster_id_fg: torch.Tensor
    medoid_index_fg: torch.Tensor
    xyz_distance_scale: float
    static_initial_cluster: int
    cluster_sizes: torch.Tensor
    num_gaussians_full: int
    foreground_mask_full: torch.Tensor

    @property
    def gaussian_ids(self) -> torch.Tensor:
        """Compatibility alias used by an earlier all-Gaussian branch."""
        return self.foreground_ids


def _robust_xyz_scale(xyz: torch.Tensor, max_samples: int = 32768) -> float:
    n = int(xyz.shape[0])
    if n < 2:
        return 1.0
    count = min(int(max_samples), max(n * 2, 1024))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    ia = torch.randint(0, n, (count,), generator=gen)
    ib = torch.randint(0, n, (count,), generator=gen)
    d = torch.linalg.norm(xyz[ia] - xyz[ib], dim=-1)
    d = d[torch.isfinite(d) & (d > 1e-8)]
    if d.numel() == 0:
        bbox = xyz.max(dim=0).values - xyz.min(dim=0).values
        return max(float(torch.linalg.norm(bbox).item()), 1e-4)
    return max(float(torch.quantile(d, 0.5).item()), 1e-4)


@torch.no_grad()
def build_combined_distance_matrix(
    xyz: torch.Tensor,
    dino: torch.Tensor,
    xyz_weight: float,
    dino_weight: float,
    xyz_scale: float = -1.0,
    block_size: int = 512,
    compute_device: str = "cpu",
    storage_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, float]:
    """Compute and retain the complete foreground XYZ+DINO distance matrix.

    Computation is intentionally CPU-only. ``compute_device`` is retained for
    CLI compatibility, but requesting CUDA prints a warning and still uses CPU.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {tuple(xyz.shape)}")
    if dino.ndim != 2 or dino.shape[0] != xyz.shape[0] or dino.shape[1] <= 0:
        raise ValueError(
            "The frozen static per-Gaussian DINO field is mandatory and must "
            f"be [N,F]; got xyz={tuple(xyz.shape)}, dino={tuple(dino.shape)}"
        )
    if not torch.isfinite(xyz).all():
        raise ValueError("Canonical foreground centers contain NaN/Inf values")
    if not torch.isfinite(dino).all():
        raise ValueError("Static foreground DINO descriptors contain NaN/Inf values")
    if str(compute_device).lower().startswith("cuda"):
        print(
            "[Joint init] CUDA was requested for the full N_fg^2 matrix; "
            "executing the exact same computation on CPU for GPU safety."
        )

    xyz_work = xyz.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    dino_work = F.normalize(
        dino.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous(),
        dim=-1,
        eps=1e-8,
    )
    if float(xyz_scale) <= 0:
        xyz_scale = _robust_xyz_scale(xyz_work)
    xyz_scale = max(float(xyz_scale), 1e-8)

    wx = float(xyz_weight)
    wf = float(dino_weight)
    if wx < 0 or wf < 0 or wx + wf <= 0:
        raise ValueError("XYZ/DINO weights must be non-negative and not both zero")

    n = int(xyz_work.shape[0])
    bs = max(1, int(block_size))
    distance = torch.empty((n, n), dtype=storage_dtype, device="cpu")
    dino_t = dino_work.t().contiguous()
    total_blocks = (n + bs - 1) // bs
    print(
        f"[Joint init] Exact semantic-foreground matrix on CPU: N_fg={n}, "
        f"pairs={n*n:,}, feature_dim={dino_work.shape[1]}, blocks={total_blocks}, "
        f"storage={str(storage_dtype).replace('torch.', '')}"
    )
    for bi, start in enumerate(range(0, n, bs), start=1):
        end = min(n, start + bs)
        d_xyz = torch.cdist(xyz_work[start:end], xyz_work, p=2) / xyz_scale
        d_cos = (1.0 - dino_work[start:end].matmul(dino_t)).clamp_(0.0, 2.0)
        distance[start:end].copy_((wx * d_xyz + wf * d_cos).to(storage_dtype))
        if bi == 1 or bi == total_blocks or bi % 4 == 0:
            print(f"[Joint init] Distance rows {end}/{n} ({100.0*end/max(n,1):.1f}%)")
        del d_xyz, d_cos
    distance.fill_diagonal_(0)
    return distance.contiguous(), xyz_scale


@torch.no_grad()
def _cluster_medoid(distance: torch.Tensor, ids: torch.Tensor, block_size: int = 1024) -> int:
    ids = ids.detach().cpu().long()
    m = int(ids.numel())
    if m == 1:
        return int(ids.item())
    costs = torch.zeros(m, dtype=torch.float64)
    for start in range(0, m, max(1, int(block_size))):
        end = min(m, start + max(1, int(block_size)))
        rows = distance.index_select(0, ids[start:end])
        sub = rows.index_select(1, ids).float()
        costs[start:end] = sub.sum(dim=1, dtype=torch.float64)
    return int(ids[int(torch.argmin(costs).item())].item())


@torch.no_grad()
def kmedoids_precomputed(
    distance: torch.Tensor,
    num_clusters: int,
    iterations: int = 12,
    medoid_block_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square precomputed matrix")
    n = int(distance.shape[0])
    k = min(max(2, int(num_clusters)), n)
    d32 = distance.float()

    first = int(torch.argmin(d32.mean(dim=1)).item())
    medoids = [first]
    nearest = d32[:, first].clone()
    for _ in range(1, k):
        nxt = int(torch.argmax(nearest).item())
        medoids.append(nxt)
        nearest = torch.minimum(nearest, d32[:, nxt])
    medoids_t = torch.tensor(medoids, dtype=torch.long)

    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iterations))):
        labels = d32.index_select(1, medoids_t).argmin(dim=1)
        new_medoids = medoids_t.clone()
        used = set()
        for c in range(k):
            ids = torch.where(labels == c)[0]
            if ids.numel() == 0:
                nearest = d32.index_select(1, medoids_t).min(dim=1).values
                for u in used:
                    nearest[u] = -1
                new_medoids[c] = int(torch.argmax(nearest).item())
            else:
                new_medoids[c] = _cluster_medoid(distance, ids, medoid_block_size)
            used.add(int(new_medoids[c].item()))
        if torch.equal(new_medoids, medoids_t):
            medoids_t = new_medoids
            break
        medoids_t = new_medoids
    labels = d32.index_select(1, medoids_t).argmin(dim=1)
    return labels.long(), medoids_t.long()


@torch.no_grad()
def initialize_joint_parts_from_static_dino(
    canonical_xyz: torch.Tensor,
    static_dino_field: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    foreground_ids: Optional[torch.Tensor] = None,
    num_moving_parts: int = 1,
    xyz_weight: float = 1.0,
    dino_weight: float = 1.0,
    xyz_scale: float = -1.0,
    distance_block_size: int = 512,
    distance_device: str = "cpu",
    distance_storage: str = "float16",
    kmedoids_iterations: int = 12,
) -> JointInitializationResult:
    """Exact initialization over every and only semantic-foreground Gaussian."""
    n_full = int(canonical_xyz.shape[0])
    if static_dino_field.ndim != 2 or static_dino_field.shape[0] != n_full:
        raise ValueError(
            "Joint initialization requires one frozen static DINO descriptor for "
            "every canonical Gaussian before semantic sub-selection"
        )
    if foreground_ids is None:
        if foreground_mask is None:
            raise ValueError(
                "Semantic foreground selection is mandatory. Refusing the old "
                "all-Gaussian joint initialization fallback."
            )
        mask = foreground_mask.detach().reshape(-1).bool().cpu()
        if mask.numel() != n_full:
            raise ValueError("foreground_mask length does not match Gaussian count")
        fg_ids = torch.where(mask)[0]
    else:
        fg_ids = foreground_ids.detach().reshape(-1).long().cpu()
        if fg_ids.numel() == 0:
            raise ValueError("semantic foreground is empty")
        if int(fg_ids.min()) < 0 or int(fg_ids.max()) >= n_full:
            raise ValueError("foreground_ids contain out-of-range indices")
        if torch.unique(fg_ids).numel() != fg_ids.numel():
            raise ValueError("foreground_ids contain duplicates")
        mask = torch.zeros(n_full, dtype=torch.bool)
        mask[fg_ids] = True

    if fg_ids.numel() < int(num_moving_parts) + 1:
        raise ValueError(
            f"Only {fg_ids.numel()} semantic-foreground Gaussians are available "
            "for foreground-static plus moving-part initialization"
        )

    xyz_fg = canonical_xyz.detach().cpu().index_select(0, fg_ids)
    dino_fg = static_dino_field.detach().cpu().index_select(0, fg_ids)
    storage_dtype = torch.float32 if str(distance_storage).lower() == "float32" else torch.float16
    distance, used_scale = build_combined_distance_matrix(
        xyz=xyz_fg,
        dino=dino_fg,
        xyz_weight=xyz_weight,
        dino_weight=dino_weight,
        xyz_scale=xyz_scale,
        block_size=distance_block_size,
        compute_device=distance_device,
        storage_dtype=storage_dtype,
    )
    labels_raw, medoids_raw = kmedoids_precomputed(
        distance,
        num_clusters=int(num_moving_parts) + 1,
        iterations=kmedoids_iterations,
    )
    counts_raw = torch.bincount(labels_raw, minlength=int(num_moving_parts) + 1)

    # Initialization convention only: largest foreground cluster -> static slot.
    order = torch.argsort(counts_raw, descending=True)
    mapping = {int(order[0]): 0}
    for slot, raw in enumerate(order[1:].tolist(), start=1):
        mapping[int(raw)] = int(slot)
    labels = torch.empty_like(labels_raw)
    medoids = torch.empty_like(medoids_raw)
    for raw, slot in mapping.items():
        labels[labels_raw == raw] = slot
        medoids[slot] = medoids_raw[raw]
    counts = torch.bincount(labels, minlength=int(num_moving_parts) + 1)

    return JointInitializationResult(
        foreground_ids=fg_ids,
        combined_distance=distance,
        initial_cluster_id_fg=labels,
        medoid_index_fg=medoids,
        xyz_distance_scale=float(used_scale),
        static_initial_cluster=0,
        cluster_sizes=counts,
        num_gaussians_full=n_full,
        foreground_mask_full=mask,
    )


class JointPartField(nn.Module):
    """Trainable cosine field defined only on semantic-foreground Gaussians."""

    def __init__(
        self,
        num_gaussians: int,
        foreground_ids: Optional[torch.Tensor] = None,
        initial_cluster_id_fg: Optional[torch.Tensor] = None,
        num_moving_parts: int = 1,
        embedding_dim: int = 8,
        assignment_temperature: float = 0.15,
        init_noise: float = 0.02,
        gaussian_ids: Optional[torch.Tensor] = None,
        **_: object,
    ):
        super().__init__()
        # gaussian_ids is accepted only as a compatibility alias from the old
        # constructor; it is interpreted as foreground IDs, never as all-scene.
        if foreground_ids is None:
            foreground_ids = gaussian_ids
        if foreground_ids is None or initial_cluster_id_fg is None:
            raise ValueError("foreground_ids and initial_cluster_id_fg are required")

        self.num_gaussians = int(num_gaussians)
        self.num_moving_parts = int(num_moving_parts)
        self.num_parts = self.num_moving_parts + 1
        self.embedding_dim = int(embedding_dim)
        self.assignment_temperature = float(assignment_temperature)
        if self.num_moving_parts != 1:
            raise ValueError("Current physical model requires exactly one moving rigid part")
        if self.embedding_dim < self.num_parts:
            raise ValueError("embedding_dim must be >= number of part slots")

        fg = foreground_ids.detach().reshape(-1).long()
        init = initial_cluster_id_fg.detach().reshape(-1).long()
        if fg.numel() != init.numel():
            raise ValueError("foreground_ids and initial labels have different lengths")
        if fg.numel() == 0:
            raise ValueError("JointPartField cannot be empty")
        if int(fg.min()) < 0 or int(fg.max()) >= self.num_gaussians:
            raise ValueError("foreground_ids contain invalid full-scene indices")
        if torch.unique(fg).numel() != fg.numel():
            raise ValueError("foreground_ids contain duplicates")
        if int(init.min()) < 0 or int(init.max()) >= self.num_parts:
            raise ValueError("initial labels are outside [0,K]")

        self.register_buffer("foreground_ids", fg, persistent=True)
        self.register_buffer("initial_cluster_id_fg", init, persistent=True)
        mask = torch.zeros(self.num_gaussians, dtype=torch.bool, device=fg.device)
        mask[fg] = True
        self.register_buffer("foreground_mask_full", mask, persistent=True)

        basis = torch.zeros(self.num_parts, self.embedding_dim, dtype=torch.float32, device=fg.device)
        for c in range(self.num_parts):
            basis[c, c] = 1.0
        self.prototypes = nn.Parameter(basis.clone())
        point = basis[init].clone()
        if float(init_noise) > 0:
            point = point + float(init_noise) * torch.randn_like(point)
        self.point_field = nn.Parameter(F.normalize(point, dim=-1))

    @property
    def gaussian_ids(self) -> torch.Tensor:
        return self.foreground_ids

    def normalized_point_field(self) -> torch.Tensor:
        return F.normalize(self.point_field, dim=-1, eps=1e-8)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-8)

    def assignments_fg(self, temperature: Optional[float] = None) -> torch.Tensor:
        temp = max(float(self.assignment_temperature if temperature is None else temperature), 1e-4)
        similarity = self.normalized_point_field().matmul(self.normalized_prototypes().t())
        return torch.softmax(similarity / temp, dim=-1)

    def full_assignments(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        fg_assign = self.assignments_fg()
        dtype = fg_assign.dtype if dtype is None else dtype
        full = torch.zeros(self.num_gaussians, self.num_parts, device=fg_assign.device, dtype=dtype)
        # Non-foreground is represented as slot-0 only for bookkeeping. This
        # tensor must never be used to force background proposal velocity to zero.
        full[:, 0] = 1.0
        full[self.foreground_ids] = fg_assign.to(dtype=dtype)
        return full

    def dynamic_probability_full(self) -> torch.Tensor:
        full = torch.zeros(self.num_gaussians, 1, device=self.point_field.device, dtype=self.point_field.dtype)
        full[self.foreground_ids] = self.assignments_fg()[:, 1:].sum(dim=-1, keepdim=True)
        return full

    @torch.no_grad()
    def hard_cluster_id_full(self) -> torch.Tensor:
        full = torch.zeros(self.num_gaussians, dtype=torch.long, device=self.point_field.device)
        full[self.foreground_ids] = self.assignments_fg().argmax(dim=-1)
        return full

    @torch.no_grad()
    def initial_cluster_id_full(self) -> torch.Tensor:
        full = torch.zeros(self.num_gaussians, dtype=torch.long, device=self.point_field.device)
        full[self.foreground_ids] = self.initial_cluster_id_fg
        return full

    def confidence_fg(self) -> torch.Tensor:
        a = self.assignments_fg()
        uniform = 1.0 / float(self.num_parts)
        return ((a.max(dim=-1, keepdim=True).values - uniform) / max(1.0 - uniform, 1e-8)).clamp(0, 1)

    def clustering_losses(
        self,
        pair_samples: int = 4096,
        different_margin: float = 0.20,
        minimum_part_fraction: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        z = self.normalized_point_field()
        proto = self.normalized_prototypes()
        a = self.assignments_fg()
        similarity = z.matmul(proto.t())
        compact = (a * (1.0 - similarity)).sum(dim=-1).mean()
        entropy = -(a.clamp_min(1e-8) * a.clamp_min(1e-8).log()).sum(dim=-1).mean()

        pp = proto.matmul(proto.t())
        offdiag = ~torch.eye(self.num_parts, dtype=torch.bool, device=pp.device)
        separation = F.relu(pp[offdiag] - float(different_margin)).pow(2).mean()
        mass = a.mean(dim=0)
        occupancy = F.relu(float(minimum_part_fraction) - mass).pow(2).sum()

        hard = a.detach().argmax(dim=-1)
        n = int(z.shape[0])
        if n < 2:
            pairwise = z.new_zeros(())
        else:
            m = min(max(2, int(pair_samples)), n)
            ids = torch.randperm(n, device=z.device)[:m]
            perm = torch.randperm(m, device=z.device)
            cos = (z[ids] * z[ids][perm]).sum(dim=-1)
            same = hard[ids] == hard[ids][perm]
            terms = []
            if same.any():
                terms.append((1.0 - cos[same]).mean())
            if (~same).any():
                terms.append(F.relu(cos[~same] - float(different_margin)).mean())
            pairwise = torch.stack(terms).mean() if terms else z.new_zeros(())

        return {
            "field_compact": compact,
            "field_pairwise": pairwise,
            "prototype_separation": separation,
            "assignment_entropy": entropy,
            "part_occupancy": occupancy,
        }

    @torch.no_grad()
    def enforce_nonempty_parts(self, minimum_part_fraction: float = 0.02) -> Dict[str, int]:
        assignment = self.assignments_fg()
        hard = assignment.argmax(dim=-1)
        n = int(hard.numel())
        target = max(1, int(math.ceil(float(minimum_part_fraction) * n)))
        before = torch.bincount(hard, minlength=self.num_parts)
        changed = 0
        z = self.normalized_point_field()
        proto = self.normalized_prototypes()
        for slot in range(self.num_parts):
            deficit = target - int((hard == slot).sum())
            if deficit <= 0:
                continue
            candidates = torch.where(hard != slot)[0]
            if candidates.numel() == 0:
                continue
            take = candidates[torch.topk(assignment[candidates, slot], k=min(deficit, candidates.numel())).indices]
            self.point_field.data[take] = F.normalize(0.1 * z[take] + 0.9 * proto[slot][None], dim=-1)
            hard[take] = slot
            changed += int(take.numel())
        after = torch.bincount(self.assignments_fg().argmax(dim=-1), minlength=self.num_parts)
        return {
            "changed": changed,
            "minimum_points_per_part": target,
            "before_min": int(before.min()),
            "after_min": int(after.min()),
        }

    @torch.no_grad()
    def summary(self) -> Dict[str, object]:
        a = self.assignments_fg()
        hard = a.argmax(dim=-1)
        return {
            "num_gaussians_full": self.num_gaussians,
            "num_semantic_foreground_gaussians": int(self.foreground_ids.numel()),
            "num_semantic_background_gaussians": int(self.num_gaussians - self.foreground_ids.numel()),
            "num_moving_parts": self.num_moving_parts,
            "num_parts": self.num_parts,
            "embedding_dim": self.embedding_dim,
            "assignment_temperature": self.assignment_temperature,
            "initial_cluster_sizes": torch.bincount(self.initial_cluster_id_fg.cpu(), minlength=self.num_parts).tolist(),
            "current_cluster_sizes": torch.bincount(hard.cpu(), minlength=self.num_parts).tolist(),
            "current_part_mass": a.mean(dim=0).cpu().tolist(),
            "prototype_cosine": self.normalized_prototypes().matmul(self.normalized_prototypes().t()).cpu().tolist(),
            "semantic_isolation": True,
        }


def save_joint_initialization(
    output_dir: str,
    result: JointInitializationResult,
    xyz_weight: float,
    dino_weight: float,
    save_distance_matrix: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "foreground_ids": result.foreground_ids,
        "gaussian_ids": result.foreground_ids,  # compatibility alias
        "foreground_mask_full": result.foreground_mask_full,
        "initial_cluster_id_fg": result.initial_cluster_id_fg,
        "medoid_index_fg": result.medoid_index_fg,
        "cluster_sizes": result.cluster_sizes,
        "xyz_distance_scale": float(result.xyz_distance_scale),
        "xyz_weight": float(xyz_weight),
        "dino_weight": float(dino_weight),
        "num_gaussians_full": int(result.num_gaussians_full),
    }
    torch.save(payload, out / "joint_initial_clusters.pt")
    if save_distance_matrix:
        torch.save(
            {
                "combined_distance": result.combined_distance,
                "foreground_ids": result.foreground_ids,
                "foreground_mask_full": result.foreground_mask_full,
                "xyz_distance_scale": float(result.xyz_distance_scale),
                "xyz_weight": float(xyz_weight),
                "dino_weight": float(dino_weight),
                "definition": "lambda_xyz*euclidean/scale + lambda_dino*(1-cosine(static_DINO_i,static_DINO_j))",
            },
            out / "combined_xyz_dino_distance_matrix.pt",
        )
    summary = {
        "num_gaussians_full": int(result.num_gaussians_full),
        "num_foreground_gaussians": int(result.foreground_ids.numel()),
        "num_background_gaussians": int(result.num_gaussians_full - result.foreground_ids.numel()),
        "distance_matrix_shape": list(result.combined_distance.shape),
        "distance_matrix_dtype": str(result.combined_distance.dtype),
        "xyz_distance_scale": float(result.xyz_distance_scale),
        "xyz_weight": float(xyz_weight),
        "dino_weight": float(dino_weight),
        "static_slot": 0,
        "moving_slots": list(range(1, int(result.cluster_sizes.numel()))),
        "cluster_sizes": result.cluster_sizes.tolist(),
        "medoid_indices_in_foreground_array": result.medoid_index_fg.tolist(),
        "uses_every_semantic_foreground_gaussian_center": True,
        "uses_static_dino_for_every_semantic_foreground_gaussian": True,
        "non_foreground_used_for_joint_initialization": False,
    }
    with open(out / "joint_initialization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
