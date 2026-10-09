"""Supernode part-boundary refinement by splitting.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md section 15.

Canonical supernodes are built by geometry; parts are defined by motion, so a
supernode can straddle a real part boundary. Section 15: "This cannot remain
only mitigated by hysteresis."

The canonical GEOMETRY stays immutable -- mu^0 and R^0 are never touched. What
is refined is the partition TOPOLOGY: a supernode whose members disagree about
which part they belong to is SPLIT, deterministically, with parent-child
provenance recorded so any downstream label can be traced to the original
grouping.

Split-only, as section 15 requires initially: merging would make the partition
non-monotone and could undo a split that a later window still needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS = 1e-12


@dataclass
class SplitRecord:
    parent: int
    children: list
    disagreement: float
    num_members: int
    method: str = "kmeans2_on_part_evidence"


@dataclass
class RefinedPartition:
    partition: object                       # the new SupernodePartition
    splits: list = field(default_factory=list)
    parent_of: dict = field(default_factory=dict)      # child -> parent
    disagreement: list = field(default_factory=list)   # per ORIGINAL supernode
    initial_num_supernodes: int = 0

    def to_dict(self) -> dict:
        return {"initial_num_supernodes": self.initial_num_supernodes,
                "final_num_supernodes": self.partition.num_supernodes,
                "num_splits": len(self.splits),
                "splits": [{"parent": s.parent, "children": s.children,
                            "disagreement": s.disagreement,
                            "num_members": s.num_members, "method": s.method}
                           for s in self.splits],
                "parent_of": {str(k): v for k, v in self.parent_of.items()},
                "disagreement_per_initial_supernode": self.disagreement,
                "canonical_geometry_mutated": False}


@torch.no_grad()
def per_gaussian_part_evidence(linearization, scene, mu: torch.Tensor,
                               labels: torch.Tensor, precisions, residuals,
                               K: int) -> torch.Tensor:
    """s_{i,k}: how much each GAUSSIAN would explain under each part model.

    This is the per-Gaussian analogue of the supernode unary, so disagreement
    inside a supernode can be measured at the granularity the split needs.
    """
    weighted = [(p * r, q * d) for (p, q), (r, d) in zip(precisions, residuals)]
    g = linearization.apply_HT(weighted)                   # [N,6]
    s = torch.zeros((mu.shape[0], K), device=mu.device, dtype=mu.dtype)
    for k in range(1, K + 1):
        model = scene.parts[k - 1]
        if model is None:
            continue
        # The tangent of part k's model evaluated at EVERY Gaussian, not only
        # its current members -- the question is whether a member of some other
        # label would be better explained by this one.
        t = model.tangent(mu)
        s[:, k - 1] = (g * t).sum(dim=1)
    return s


@torch.no_grad()
def supernode_disagreement(evidence: torch.Tensor, supernode_id: torch.Tensor,
                           num_supernodes: int) -> torch.Tensor:
    """Section 15's max_k Var_{i in c}(s_{i,k}), normalised.

    Normalised by the supernode's own evidence scale, so a strongly-moving
    supernode is not flagged merely for having large numbers.
    """
    sid = supernode_id.long().clamp_min(0)
    C, K = num_supernodes, evidence.shape[1]
    out = torch.zeros(C, device=evidence.device, dtype=evidence.dtype)
    cnt = torch.zeros(C, device=evidence.device, dtype=evidence.dtype)
    cnt.index_add_(0, sid, torch.ones(sid.shape[0], device=evidence.device,
                                      dtype=evidence.dtype))
    for k in range(K):
        v = evidence[:, k]
        s1 = torch.zeros(C, device=v.device, dtype=v.dtype).index_add_(0, sid, v)
        s2 = torch.zeros(C, device=v.device, dtype=v.dtype).index_add_(0, sid, v * v)
        n = cnt.clamp_min(1.0)
        var = (s2 / n - (s1 / n) ** 2).clamp_min(0.0)
        scale = (s2 / n).clamp_min(EPS)
        out = torch.maximum(out, var / scale)
    return out


@torch.no_grad()
def split_supernodes(partition, evidence: torch.Tensor, *,
                     tau_split: float = 0.25, min_members: int = 8,
                     max_splits: int = 64) -> RefinedPartition:
    """Deterministic split-only topology refinement (section 15).

    A flagged supernode is split by 2-means on its members' part-evidence
    vectors -- the quantity they disagree about -- seeded deterministically by
    the two extreme members, so the result depends only on the data.
    """
    from estimator.supernodes import SupernodePartition

    sid = partition.supernode_id.long().clamp_min(0)
    C = partition.num_supernodes
    dis = supernode_disagreement(evidence, sid, C)
    order = torch.argsort(dis, descending=True)

    new_id = sid.clone()
    next_label = C
    splits, parent_of = [], {}
    for c in order.tolist():
        if len(splits) >= max_splits:
            break
        if float(dis[c]) <= tau_split:
            break
        members = torch.nonzero(sid == c, as_tuple=False).reshape(-1)
        if members.numel() < 2 * min_members:
            continue
        x = evidence[members]
        # deterministic seeding: the two members furthest apart in evidence
        d = torch.cdist(x, x)
        i, j = divmod(int(d.argmax()), d.shape[1])
        ca, cb = x[i].clone(), x[j].clone()
        assign = torch.zeros(members.numel(), dtype=torch.long, device=x.device)
        for _ in range(25):
            da = (x - ca).norm(dim=1)
            db = (x - cb).norm(dim=1)
            new_assign = (db < da).long()
            if torch.equal(new_assign, assign):
                break
            assign = new_assign
            if bool((assign == 0).any()):
                ca = x[assign == 0].mean(dim=0)
            if bool((assign == 1).any()):
                cb = x[assign == 1].mean(dim=0)
        if int((assign == 1).sum()) < min_members or int((assign == 0).sum()) < min_members:
            continue
        child = next_label
        next_label += 1
        new_id[members[assign == 1]] = child
        parent_of[child] = c
        splits.append(SplitRecord(parent=c, children=[c, child],
                                  disagreement=float(dis[c]),
                                  num_members=int(members.numel())))

    if not splits:
        return RefinedPartition(partition=partition, splits=[], parent_of={},
                                disagreement=dis.cpu().tolist(),
                                initial_num_supernodes=C)

    # rebuild the partition on the SAME canonical means -- geometry untouched
    x = partition.canonical_means
    n_new = int(new_id.max()) + 1
    centroids = torch.zeros((n_new, 3), device=x.device, dtype=x.dtype)
    counts = torch.zeros(n_new, device=x.device, dtype=x.dtype)
    centroids.index_add_(0, new_id, x)
    counts.index_add_(0, new_id, torch.ones(x.shape[0], device=x.device, dtype=x.dtype))
    keep = counts > 0
    remap = -torch.ones(n_new, dtype=torch.long, device=x.device)
    remap[keep] = torch.arange(int(keep.sum()), device=x.device)
    new_id = remap[new_id]
    centroids = (centroids[keep] / counts[keep].unsqueeze(1))
    counts = counts[keep]
    parent_of = {int(remap[c]): p for c, p in parent_of.items()
                 if int(remap[c]) >= 0}

    c_new = int(keep.sum())
    radius = torch.zeros(c_new, device=x.device, dtype=x.dtype)
    for k in range(c_new):
        m = new_id == k
        if bool(m.any()):
            radius[k] = (x[m] - centroids[k]).norm(dim=1).max()
    kk = min(int(partition.config.get("knn", 8)), max(c_new - 1, 1))
    edges = []
    if c_new > 1:
        dmat = torch.cdist(centroids, centroids)
        dmat.fill_diagonal_(float("inf"))
        nbr = dmat.topk(kk, dim=1, largest=False).indices
        for a in range(c_new):
            for b in nbr[a].tolist():
                edges.append((min(a, b), max(a, b)))
        edges = sorted(set(edges))
    adjacency = (torch.tensor(edges, dtype=torch.long, device=x.device)
                 if edges else torch.zeros((0, 2), dtype=torch.long, device=x.device))

    refined = SupernodePartition(
        supernode_id=new_id.to(torch.int32), centroids=centroids,
        member_counts=counts.long(), radius=radius, adjacency=adjacency,
        canonical_means=partition.canonical_means,
        config={**partition.config, "refined_by_split": True,
                "num_supernodes": c_new, "tau_split": tau_split})
    return RefinedPartition(partition=refined, splits=splits,
                            parent_of=parent_of, disagreement=dis.cpu().tolist(),
                            initial_num_supernodes=C)
