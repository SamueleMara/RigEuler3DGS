"""Free-space two-hypothesis refinement of the rigid Phase-2 support (opt-in).

The coherence selector picks the moving support from Phase-1 trajectories, so it
inherits two Phase-1 failure modes: a static surface that the moving part HIDES
(the inside of a fridge as the door closes) keeps the velocity it picked up and
travels with the occluder, and a moving surface whose track was torn is left out.
Neither depends on the observations once the rigid motion T_t is known, so after
the first rigid solve every Gaussian centre x (canonical) is tested against the
OBSERVED depth d of every camera and processing frame under both hypotheses:

    static  x          moving  T_t x

    violated    z < d - tau     the camera sees a surface behind the point: it is not there
    consistent  |z - d| <= tau
    hidden      z > d + tau     occluded -- no evidence. An occluder (the arm, the
                                closing part itself) can only hide a point, never
                                violate it, so a disappearing static surface stays static.

    L = 2 (V_static - V_moving) + (C_moving - C_static)

moving if L > margin; every other Gaussian is STATIC, the null hypothesis: the
moving label has to be earned by evidence, so a Gaussian seen fewer than
`min_obs` times while the hypotheses differ (a surface the moving part hides for
most of the motion, a back face) is static too. Only Gaussians whose two
hypotheses never separate by more than 2 tau (on the joint axis) carry no
decision and keep their previous label. Counts come only
from frames where the hypotheses are more than 2 tau apart, and only from cameras
that saw the Gaussian on the observed surface at the canonical frame (a
transparent or floating Gaussian sits in front of the sensed depth from the start,
which is not motion evidence). Observations and the estimated rigid motion only:
no GT, no Phase-1 trajectory.
"""
from __future__ import annotations

import torch

CONSISTENT, VIOLATED, HIDDEN = 0, 1, 2
LABELS = ("moving", "static", "small_motion", "unobserved")


@torch.no_grad()
def classify(points: torch.Tensor, camera, depth: torch.Tensor, valid: torch.Tensor, tau: float) -> torch.Tensor:
    """[N] code per point in one camera: 0 consistent, 1 violated, 2 hidden, -1 no pixel/depth.

    Pinhole of the 3DGS camera spec (row-vector view matrix, p_cam = [p_w, 1] @ V; pixel
    centres at ((x/z)/tan_fov_x + 1) W/2 - 1/2)."""
    dev = depth.device
    V = camera.viewmatrix.to(dev, torch.float64)
    P = torch.cat([points.to(dev, torch.float64), torch.ones_like(points[:, :1], device=dev, dtype=torch.float64)], 1) @ V
    z = P[:, 2]
    H, W = depth.shape[-2:]
    zs = z.clamp_min(1e-9)
    u = torch.round((P[:, 0] / zs / camera.tanfovx + 1.0) * W / 2.0 - 0.5).long()
    v = torch.round((P[:, 1] / zs / camera.tanfovy + 1.0) * H / 2.0 - 0.5).long()
    inside = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = torch.full((points.shape[0],), -1, dtype=torch.int8, device=dev)
    ui, vi = u[inside], v[inside]
    d = depth[vi, ui].double()
    ok = valid[vi, ui].bool() & (d > 0)
    r = z[inside] - d
    code = torch.where(r.abs() <= tau, CONSISTENT, torch.where(r < -tau, VIOLATED, HIDDEN)).to(torch.int8)
    code[~ok] = -1
    out[inside] = code
    return out


@torch.no_grad()
def evidence(mu0: torch.Tensor, R: torch.Tensor, p: torch.Tensor, observations, cameras, tau: float = 0.02,
             stride: int = 1, device=None) -> dict:
    """Counts [N, 2 (static, moving), 3 (consistent, violated, hidden)] over frames x cameras.

    R [T,3,3], p [T,3]: the rigid pose of each processing frame (x_t = R_t x + p_t);
    observations[t][c] = (rgb, depth, valid) or None."""
    dev = device or (cameras[0].viewmatrix.device if cameras else mu0.device)
    x0 = mu0.to(dev, torch.float64)
    n = x0.shape[0]
    counts = torch.zeros((n, 2, 3), dtype=torch.int32, device=dev)
    disp = torch.zeros(n, dtype=torch.float64, device=dev)
    # A camera speaks for a Gaussian only if it saw it ON the observed surface at the
    # canonical frame (processing frame 0, where both hypotheses coincide). A Gaussian
    # already in front of the sensed depth there -- a transparent bottle the depth
    # sees through, a floater -- would otherwise read as "static violated" in every
    # frame, which is not motion evidence.
    gate = [None if ob is None else classify(x0, cameras[c], ob[1].to(dev), ob[2].to(dev), tau) == CONSISTENT
            for c, ob in enumerate(observations[0])]
    frames = list(range(0, len(observations), max(int(stride), 1)))
    for t in frames:
        xm = x0 @ R[t].to(dev, torch.float64).T + p[t].to(dev, torch.float64)
        d = (xm - x0).norm(dim=1)
        disp = torch.maximum(disp, d)
        apart = d > 2 * tau                                  # frames where the hypotheses can disagree
        for c, ob in enumerate(observations[t]):
            if ob is None or gate[c] is None:
                continue
            use = gate[c] & apart
            depth, valid = ob[1].to(dev), ob[2].to(dev)
            for h, X in enumerate((x0, xm)):
                code = classify(X, cameras[c], depth, valid, tau)
                for k in (CONSISTENT, VIOLATED, HIDDEN):
                    counts[:, h, k] += ((code == k) & use).int()
    return {"counts": counts.cpu(), "displacement_max": disp.cpu(), "frames": frames, "tau": float(tau),
            "canonical_surface_cameras": torch.stack([g for g in gate if g is not None]).sum(0).cpu()
            if any(g is not None for g in gate) else torch.zeros(n, dtype=torch.long)}


def assign(ev: dict, previous: torch.Tensor, margin: float = 4.0, min_obs: int = 6, min_members: int = 200) -> dict:
    """Membership from the evidence. previous: bool [N] (the coherence support)."""
    c = ev["counts"].long()
    cs, vs, cm, vm = c[:, 0, CONSISTENT], c[:, 0, VIOLATED], c[:, 1, CONSISTENT], c[:, 1, VIOLATED]
    score = 2 * (vs - vm) + (cm - cs)
    label = torch.where(score > margin, 0, 1)                                 # moving / static
    small = ev["displacement_max"] < 2 * ev["tau"]
    unobserved = (cs + vs + cm + vm) < min_obs
    label[unobserved] = 3
    label[small] = 2
    prev = previous.bool().cpu()
    members = torch.where(label == 2, prev, label == 0)
    accepted = bool((label == 0).sum() >= min_members)
    if not accepted:
        members = prev.clone()
    names = {i: nm for i, nm in enumerate(LABELS)}
    table = {nm: {"total": int((label == i).sum()), "in_previous_support": int(((label == i) & prev).sum())}
             for i, nm in names.items()}
    return {
        "members": members, "label": label.to(torch.int8), "score": score.to(torch.int32),
        "accepted": accepted,
        "summary": {
            "accepted": accepted,
            "reason": None if accepted else f"fewer than {min_members} Gaussians carry moving evidence under this "
                                            "rigid motion; the coherence support is kept",
            "previous_members": int(prev.sum()), "members": int(members.sum()),
            "added": int((members & ~prev).sum()), "removed": int((prev & ~members).sum()),
            "labels": table,
            "rule": f"L = 2 (V_static - V_moving) + (C_moving - C_static) > {margin}: moving; else static "
                    f"(null hypothesis), including fewer than {min_obs} observations; max displacement < 2 tau: "
                    "keep the previous label",
            "tau_m": ev["tau"], "frames_used": len(ev["frames"])}}
