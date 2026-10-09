"""Canonical supernode partition and the grouped observation operator.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 3.

The partition is built ONCE from the canonical frame-0 geometry and is then
immutable (section 3.1). It must not use future GT motion labels -- grouping on
motion would make the support/joint results circular.

Section 3.2's tangent, which is what makes the state low-dimensional:

    delta_mu_i  = delta_t_c + delta_theta_c x (mu_i^0 - c_c^0)
    delta_phi_i = delta_theta_c

so a supernode has 6 DoF no matter how many Gaussians it owns, and neighbouring
Gaussians can no longer trade arbitrary motion against one another -- which is
exactly the ill-conditioning section 1.2 measured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from diff_gaussian_rasterization import _C

EPS = 1e-12


@dataclass
class SupernodePartition:
    """An immutable canonical grouping (section 3.1)."""

    supernode_id: torch.Tensor        # [N] int32, -1 = unassigned
    centroids: torch.Tensor           # [C,3] canonical c_c^0
    member_counts: torch.Tensor       # [C]
    radius: torch.Tensor              # [C] spatial extent
    adjacency: torch.Tensor           # [E,2] int64 undirected supernode graph
    canonical_means: torch.Tensor     # [N,3] the mu^0 the grouping was built on
    config: dict = field(default_factory=dict)

    @property
    def num_supernodes(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def num_gaussians(self) -> int:
        return int(self.supernode_id.shape[0])

    def canonical_hash(self) -> str:
        """Section 14's canonical-reference hash: this must not change for the
        whole of motion estimation."""
        h = hashlib.sha256()
        h.update(self.canonical_means.detach().cpu().numpy().tobytes())
        h.update(self.supernode_id.detach().cpu().numpy().tobytes())
        h.update(self.centroids.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return {"num_supernodes": self.num_supernodes,
                "num_gaussians": self.num_gaussians,
                "member_counts": self.member_counts.cpu().tolist(),
                "radius": self.radius.cpu().tolist(),
                "centroids": self.centroids.cpu().tolist(),
                "num_edges": int(self.adjacency.shape[0]),
                "canonical_hash": self.canonical_hash(),
                "config": self.config}


@torch.no_grad()
def build_supernodes(means: torch.Tensor, *, target_size: int = 64,
                     num_supernodes: int | None = None, knn: int = 8,
                     iterations: int = 25, seed: int = 0) -> SupernodePartition:
    """Deterministic canonical grouping (section 3.1).

    Farthest-point seeding followed by Lloyd iterations on canonical geometry.
    Deterministic for a fixed seed and config, as section 3.1 requires: the
    seeding is a farthest-point walk from one seeded index, not a random draw
    per iteration.
    """
    x = means.detach()
    n = int(x.shape[0])
    c = int(num_supernodes) if num_supernodes else max(1, n // max(target_size, 1))
    c = max(1, min(c, n))

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=g).item())
    seeds = [first]
    d = (x - x[first]).norm(dim=1)
    for _ in range(c - 1):
        nxt = int(d.argmax())
        seeds.append(nxt)
        d = torch.minimum(d, (x - x[nxt]).norm(dim=1))
    centroids = x[torch.tensor(seeds, device=x.device)].clone()

    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(int(iterations)):
        # chunked nearest-centroid to keep memory bounded on the full scene
        parts = []
        for s in range(0, n, 8192):
            parts.append(torch.cdist(x[s:s + 8192], centroids).argmin(dim=1))
        new_assign = torch.cat(parts)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for k in range(c):
            m = assign == k
            if bool(m.any()):
                centroids[k] = x[m].mean(dim=0)

    counts = torch.bincount(assign, minlength=c)
    # An empty supernode is a degenerate group, not a silent one: drop it and
    # renumber, so `num_supernodes` always equals the number of real groups.
    keep = counts > 0
    if not bool(keep.all()):
        remap = -torch.ones(c, dtype=torch.long, device=x.device)
        remap[keep] = torch.arange(int(keep.sum()), device=x.device)
        assign = remap[assign]
        centroids = centroids[keep]
        counts = counts[keep]
        c = int(keep.sum())

    radius = torch.zeros(c, device=x.device, dtype=x.dtype)
    for k in range(c):
        m = assign == k
        if bool(m.any()):
            radius[k] = (x[m] - centroids[k]).norm(dim=1).max()

    # supernode adjacency from centroid kNN (section 3.1), used by the Potts
    # smoothness term in support inference (section 6.3).
    kk = min(int(knn), max(c - 1, 1))
    edges = []
    if c > 1:
        dist = torch.cdist(centroids, centroids)
        dist.fill_diagonal_(float("inf"))
        nbr = dist.topk(kk, dim=1, largest=False).indices
        for a in range(c):
            for b in nbr[a].tolist():
                if a < b:
                    edges.append((a, b))
                elif b < a:
                    edges.append((b, a))
        edges = sorted(set(edges))
    adjacency = (torch.tensor(edges, dtype=torch.long, device=x.device)
                 if edges else torch.zeros((0, 2), dtype=torch.long, device=x.device))

    return SupernodePartition(
        supernode_id=assign.to(torch.int32), centroids=centroids,
        member_counts=counts, radius=radius, adjacency=adjacency,
        canonical_means=x.clone(),
        config={"target_size": target_size, "num_supernodes": c, "knn": knn,
                "iterations": iterations, "seed": seed})


# ---------------------------------------------------------------------------
# Grouped observation operator (section 3.3)
# ---------------------------------------------------------------------------
class GroupedObservation:
    """H_G G, matrix-free, on top of the validated per-Gaussian operators."""

    def __init__(self, linearization, partition: SupernodePartition,
                 canonical_means: torch.Tensor | None = None):
        self.lin = linearization
        self.part = partition
        # The tangent is defined against CANONICAL geometry (section 3.2's
        # mu_i^0 - c_c^0), not the current deformed state, so the group basis
        # cannot drift as the estimate moves.
        self.means = (canonical_means if canonical_means is not None
                      else partition.canonical_means).contiguous().float()
        self.sid = partition.supernode_id.contiguous().to(torch.int32)
        self.cent = partition.centroids.contiguous().float()

    @property
    def num_supernodes(self) -> int:
        return self.part.num_supernodes

    def expand(self, delta_zeta: torch.Tensor) -> torch.Tensor:
        """G delta_zeta -> [N,6]."""
        return _C.iekf_expand_group_tangent(
            self.sid, self.means, self.cent, delta_zeta.contiguous().float(), False)

    def reduce(self, pose_adjoint: torch.Tensor) -> torch.Tensor:
        """G^T applied to a [N,6] adjoint -> [C,6]."""
        return _C.iekf_reduce_group_adjoint(
            self.sid, self.means, self.cent, pose_adjoint.contiguous().float(),
            self.num_supernodes, False)

    def apply_H(self, delta_zeta: torch.Tensor):
        return self.lin.apply_H(self.expand(delta_zeta))

    def apply_HT(self, pixel_adjoints) -> torch.Tensor:
        return self.reduce(self.lin.apply_HT(pixel_adjoints))

    def block_fisher(self, precisions) -> torch.Tensor:
        """[C,6,6]. Preconditioner only -- the exact operator keeps every
        cross-supernode coupling through apply_H / apply_HT (section 3.3)."""
        from diff_gaussian_rasterization import iekf_build_fisher

        packed = None
        for view, (pc, pd) in zip(self.lin.views, precisions):
            f = iekf_build_fisher(view.cache, pc, pd, view.transmittance,
                                  view.alpha, view.depth, view.n_contrib)
            packed = f if packed is None else packed + f
        if packed is None:
            return torch.zeros((self.num_supernodes, 6, 6), device=self.means.device)
        grouped = _C.iekf_reduce_group_fisher(
            self.sid, self.means, self.cent, packed.contiguous(),
            self.num_supernodes, False)
        out = grouped.new_zeros((self.num_supernodes, 6, 6))
        t = 0
        for a in range(6):
            for b in range(a, 6):
                out[:, a, b] = grouped[:, t]
                out[:, b, a] = grouped[:, t]
                t += 1
        return out
