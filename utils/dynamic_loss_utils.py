#
# Losses and detached diagnostics for the adaptive Eulerian 3DGS extension.
#

import re
import torch
import torch.nn.functional as F


def get_camera_timestamp(camera, fps=30.0, time_scale=1.0):
    if hasattr(camera, "timestamp"):
        return float(camera.timestamp) * float(time_scale)
    if hasattr(camera, "frame_id"):
        return float(camera.frame_id) / max(float(fps), 1e-6) * float(time_scale)
    name = getattr(camera, "image_name", "")
    nums = re.findall(r"\d+", str(name))
    if nums:
        return float(nums[-1]) / max(float(fps), 1e-6) * float(time_scale)
    uid = float(getattr(camera, "uid", 0))
    return uid / max(float(fps), 1e-6) * float(time_scale)


def normalize_camera_timestamps(cameras, fps=30.0, time_scale=1.0):
    raw = []
    for cam in cameras:
        raw.append(get_camera_timestamp(cam, fps=fps, time_scale=1.0))
    if not raw:
        return
    min_t = min(raw)
    max_t = max(raw)
    denom = max(max_t - min_t, 1e-6)
    for cam, t in zip(cameras, raw):
        cam.raw_timestamp = float(t)
        cam.timestamp = ((float(t) - min_t) / denom) * float(time_scale)
        cam.frame_id = int(round(float(t) * float(fps)))


def speed_loss(velocity, reduction="mean"):
    if velocity is None or velocity.numel() == 0:
        return torch.tensor(0.0, device="cuda")
    val = (velocity ** 2).sum(dim=-1)
    return val.mean() if reduction == "mean" else val


def static_velocity_loss(velocity, dynamic_prob=None):
    if velocity is None or velocity.numel() == 0:
        return torch.tensor(0.0, device="cuda")
    per = (velocity ** 2).sum(dim=-1, keepdim=True)
    if dynamic_prob is not None and dynamic_prob.numel() == per.numel():
        static_weight = (1.0 - dynamic_prob.detach()).clamp(0.0, 1.0)
        per = per * static_weight
    return per.mean()


def dynamic_sparsity_loss(dynamic_prob):
    if dynamic_prob is None or dynamic_prob.numel() == 0:
        return torch.tensor(0.0, device="cuda")
    return dynamic_prob.mean()


def dynamic_binarization_loss(dynamic_prob):
    if dynamic_prob is None or dynamic_prob.numel() == 0:
        return torch.tensor(0.0, device="cuda")
    p = dynamic_prob.clamp(1e-6, 1.0 - 1e-6)
    return (p * (1.0 - p)).mean()


def cycle_consistency_loss(x_target_detached, x_back):
    if x_back is None or x_back.numel() == 0:
        return torch.tensor(0.0, device="cuda")
    return ((x_back - x_target_detached.detach()) ** 2).sum(dim=-1).mean()


def sample_pairs_by_cell(cell_id, max_cells=512, pairs_per_cell=32, min_points=2):
    """Sample local pairs from current Gaussian-to-motion-cell assignments."""
    if cell_id is None or cell_id.numel() == 0:
        return None
    device = cell_id.device
    with torch.no_grad():
        active = torch.unique(cell_id)
        if active.numel() > int(max_cells):
            perm = torch.randperm(active.numel(), device=device)[:int(max_cells)]
            active = active[perm]
        src = []
        dst = []
        for cid in active.tolist():
            idx = torch.where(cell_id == cid)[0]
            if idx.numel() < int(min_points):
                continue
            k = min(int(pairs_per_cell), int(idx.numel()))
            a = idx[torch.randint(0, idx.numel(), (k,), device=device)]
            b = idx[torch.randint(0, idx.numel(), (k,), device=device)]
            valid = a != b
            if valid.any():
                src.append(a[valid])
                dst.append(b[valid])
        if not src:
            return None
        return torch.cat(src, dim=0), torch.cat(dst, dim=0)


def dino_pair_weights(dino_feat, src, dst, sigma_f=0.25):
    if dino_feat is None or dino_feat.numel() == 0:
        return torch.ones(src.shape[0], 1, device=src.device)
    f1 = F.normalize(dino_feat[src].detach(), dim=-1)
    f2 = F.normalize(dino_feat[dst].detach(), dim=-1)
    cosine = (f1 * f2).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    dist = 1.0 - cosine
    return torch.exp(-dist / max(float(sigma_f), 1e-6)).detach()


def dino_weighted_rigidity_loss(x0, xt, cell_id=None, dino_feat=None,
                                pairs=None, max_cells=512, pairs_per_cell=32,
                                sigma_f=0.25):
    if xt is None or xt.numel() == 0:
        return torch.tensor(0.0, device=x0.device)
    if pairs is None:
        pairs = sample_pairs_by_cell(cell_id, max_cells=max_cells, pairs_per_cell=pairs_per_cell)
    if pairs is None:
        return torch.tensor(0.0, device=x0.device)
    src, dst = pairs
    if src.numel() == 0:
        return torch.tensor(0.0, device=x0.device)
    d0 = torch.linalg.norm(x0[src].detach() - x0[dst].detach(), dim=-1, keepdim=True)
    dt = torch.linalg.norm(xt[src] - xt[dst], dim=-1, keepdim=True)
    w = dino_pair_weights(dino_feat, src, dst, sigma_f=sigma_f)
    return (w * (dt - d0) ** 2).mean()


def _kabsch_transform(src, dst):
    """Return R,t fitting dst ~= src @ R.T + t. src/dst: [N,3]."""
    src_mean = src.mean(dim=0, keepdim=True)
    dst_mean = dst.mean(dim=0, keepdim=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.t().matmul(dst_c)
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t().matmul(U.t())
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1, :] *= -1
        R = Vh.t().matmul(U.t())
    t = dst_mean.squeeze(0) - src_mean.squeeze(0).matmul(R.t())
    return R, t


def se3_local_coherence_loss(x0, xt, cell_id, max_cells=256, min_points=6, max_points_per_cell=64):
    if cell_id is None or cell_id.numel() == 0 or xt is None or xt.numel() == 0:
        return torch.tensor(0.0, device=x0.device)
    losses = []
    with torch.no_grad():
        unique = torch.unique(cell_id)
        if unique.numel() > int(max_cells):
            unique = unique[torch.randperm(unique.numel(), device=unique.device)[:int(max_cells)]]
    for cid in unique.tolist():
        idx = torch.where(cell_id == cid)[0]
        if idx.numel() < int(min_points):
            continue
        if idx.numel() > int(max_points_per_cell):
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:int(max_points_per_cell)]]
        src = x0[idx].detach()
        dst = xt[idx]
        try:
            R, t = _kabsch_transform(src.detach(), dst.detach())
        except RuntimeError:
            continue
        pred = src.matmul(R.detach().t()) + t.detach()
        losses.append(((dst - pred) ** 2).sum(dim=-1).mean())
    if not losses:
        return torch.tensor(0.0, device=x0.device)
    return torch.stack(losses).mean()


def compute_cell_incoherence_scores(x0, xt, velocity, cell_id, dino_feat=None,
                                     render_residual_per_gaussian=None,
                                     dynamic_prob=None,
                                     twist_residual_per_gaussian=None,
                                     num_cells=None,
                                     lambda_vel=1.0,
                                     lambda_disp=0.25,
                                     lambda_dino=0.1,
                                     lambda_img=0.0,
                                     lambda_dyn_entropy=0.0,
                                     lambda_dyn_var=0.0,
                                     lambda_twist=0.0,
                                     lambda_strain=0.0,
                                     strain_max_points=64):
    """Detached per-cell ambiguity scores for adaptive motion-grid split/merge.

    This is the motion-grid analogue of static 3DGS densification.  Static
    densification uses high image-space position gradients to clone/split
    Gaussians.  Here we keep Gaussian identity fixed and instead split the
    local velocity-field cell when a cell mixes incompatible evidence:

      * velocity variance: static + dynamic Gaussians in one cell;
      * displacement variance: different transported motion inside one cell;
      * DINO variance: semantic/part boundary inside one cell;
      * render gradients: RGB/SSIM/depth losses still push visible Gaussians;
      * dynamic-probability entropy/variance: ambiguous static/dynamic labels;
      * global twist residual: dynamic cells whose velocity disagrees with the
        fitted rigid-body Eulerian twist;
      * local strain/radial-rate: cells whose internal velocity field stretches
        local distances, a signature of static/dynamic mixing near hinges.

    The returned scores are detached diagnostics only; topology changes are not
    differentiated through.
    """
    device = x0.device
    if cell_id is None or cell_id.numel() == 0:
        return torch.zeros(0, device=device)
    if num_cells is None:
        num_cells = int(cell_id.max().item()) + 1
    scores = torch.zeros(num_cells, device=device)
    counts = torch.zeros(num_cells, device=device)
    with torch.no_grad():
        cell_id = cell_id.detach().long()
        disp = xt.detach() - x0.detach() if xt is not None else torch.zeros_like(x0)
        vel = velocity.detach() if velocity is not None else torch.zeros_like(x0)
        dyn = None
        if dynamic_prob is not None and dynamic_prob.numel() > 0:
            dyn = dynamic_prob.detach().reshape(-1)
        img = None
        if render_residual_per_gaussian is not None and render_residual_per_gaussian.numel() > 0:
            img = render_residual_per_gaussian.detach().reshape(-1)
        twist_res = None
        if twist_residual_per_gaussian is not None and twist_residual_per_gaussian.numel() > 0:
            twist_res = twist_residual_per_gaussian.detach().reshape(-1)
        pos = xt.detach() if xt is not None else x0.detach()

        for cid in torch.unique(cell_id).tolist():
            if cid < 0 or cid >= num_cells:
                continue
            idx = torch.where(cell_id == cid)[0]
            if idx.numel() < 2:
                continue
            v = vel[idx]
            d = disp[idx]
            score = torch.tensor(0.0, device=device)

            if float(lambda_vel) > 0:
                score = score + float(lambda_vel) * v.var(dim=0, unbiased=False).sum()
            if float(lambda_disp) > 0:
                score = score + float(lambda_disp) * d.var(dim=0, unbiased=False).sum()

            if dino_feat is not None and dino_feat.numel() > 0 and float(lambda_dino) > 0:
                f = F.normalize(dino_feat[idx].detach(), dim=-1)
                fmean = F.normalize(f.mean(dim=0, keepdim=True), dim=-1)
                score = score + float(lambda_dino) * (1.0 - (f * fmean).sum(dim=-1)).mean()

            if img is not None and float(lambda_img) > 0:
                # Actual screen/viewspace gradient magnitude ||dL/dxy||, exactly
                # like static 3DGS densification uses xyz_gradient_accum.
                score = score + float(lambda_img) * img[idx].mean()

            if twist_res is not None and float(lambda_twist) > 0:
                score = score + float(lambda_twist) * twist_res[idx].mean()

            if float(lambda_strain) > 0 and idx.numel() >= 2:
                sub = idx
                maxp = int(max(2, strain_max_points))
                if sub.numel() > maxp:
                    perm = torch.randperm(sub.numel(), device=device)[:maxp]
                    sub = sub[perm]
                px = pos[sub]
                pv = vel[sub]
                r = px[:, None, :] - px[None, :, :]
                dv = pv[:, None, :] - pv[None, :, :]
                dist = torch.linalg.norm(r, dim=-1).clamp_min(1e-8)
                radial_rate = (r * dv).sum(dim=-1) / dist
                mask = torch.triu(torch.ones_like(radial_rate, dtype=torch.bool), diagonal=1)
                if mask.any():
                    score = score + float(lambda_strain) * radial_rate[mask].pow(2).mean()

            if dyn is not None:
                p = dyn[idx].clamp(1e-6, 1.0 - 1e-6)
                if float(lambda_dyn_entropy) > 0:
                    score = score + float(lambda_dyn_entropy) * (p * (1.0 - p)).mean()
                if float(lambda_dyn_var) > 0:
                    score = score + float(lambda_dyn_var) * p.var(unbiased=False)

            scores[cid] = score
            counts[cid] = float(idx.numel())
    return scores.detach()


def velocity_temporal_consistency_loss(
    motion_field,
    x,
    t,
    dt,
    grid=None,
    gaussian_cell_id=None,
    time_min=0.0,
    time_max=1.0,
    max_points=8192,
):
    """Smooth v_theta(x,t) in time using first and second finite differences.

    The sampled positions x are detached by the caller/inside this function so
    the loss routes gradients to the velocity field and motion-grid features,
    not to the canonical Gaussian positions.  This is the direct regularizer
    that was missing when renders flickered: photometric supervision alone can
    fit each frame with temporally inconsistent velocities.

    Returns:
        (L_first_order, L_second_order)
    """
    if motion_field is None or x is None or x.numel() == 0:
        dev = x.device if x is not None else torch.device("cuda")
        z = torch.tensor(0.0, device=dev)
        return z, z

    xq = x.detach()
    n = xq.shape[0]
    if int(max_points) > 0 and n > int(max_points):
        idx = torch.randperm(n, device=xq.device)[:int(max_points)]
        xq = xq[idx]
        if gaussian_cell_id is not None and gaussian_cell_id.numel() == n:
            gaussian_cell_id = gaussian_cell_id[idx]

    t = float(t)
    dt = abs(float(dt))
    tmin = float(time_min)
    tmax = float(time_max)
    tm = max(tmin, t - dt)
    tp = min(tmax, t + dt)

    # If one side collapses at the boundary, still keep a valid first-order term.
    v0 = motion_field.query_velocity(xq, t, grid=grid, gaussian_cell_id=gaussian_cell_id)
    losses_first = []
    if tp != t:
        vp = motion_field.query_velocity(xq, tp, grid=grid, gaussian_cell_id=gaussian_cell_id)
        losses_first.append(((vp - v0) ** 2).sum(dim=-1).mean())
    else:
        vp = None
    if tm != t:
        vm = motion_field.query_velocity(xq, tm, grid=grid, gaussian_cell_id=gaussian_cell_id)
        losses_first.append(((v0 - vm) ** 2).sum(dim=-1).mean())
    else:
        vm = None

    if losses_first:
        l_first = torch.stack(losses_first).mean()
    else:
        l_first = torch.tensor(0.0, device=xq.device)

    if vp is not None and vm is not None:
        l_second = ((vp - 2.0 * v0 + vm) ** 2).sum(dim=-1).mean()
    else:
        l_second = torch.tensor(0.0, device=xq.device)
    return l_first, l_second


def dynamic_motion_target_loss(dynamic_prob, velocity, quantile=0.75, temperature=0.15):
    """Provide a positive, detached speed-based target for dynamic logits.

    In the first implementation the dynamic logits were trained only by
    sparsity/binarization terms.  That has a trivial optimum: everything is
    static.  This loss keeps labels interpretable by pushing high-speed
    Gaussians toward dynamic while leaving the velocity field gradients
    detached from the target construction.
    """
    if dynamic_prob is None or velocity is None or dynamic_prob.numel() == 0 or velocity.numel() == 0:
        dev = dynamic_prob.device if dynamic_prob is not None else (velocity.device if velocity is not None else torch.device("cuda"))
        return torch.tensor(0.0, device=dev)
    p = dynamic_prob.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    with torch.no_grad():
        speed = torch.linalg.norm(velocity.detach(), dim=-1).reshape(-1)
        if speed.numel() != p.numel() or float(speed.max().item()) <= 1e-12:
            target = torch.zeros_like(p)
        else:
            q = float(quantile)
            q = q / 100.0 if q > 1.0 else q
            q = min(max(q, 0.0), 1.0)
            thr = torch.quantile(speed, q)
            scale = torch.clamp(speed.std(unbiased=False) * float(temperature), min=1e-8)
            target = torch.sigmoid((speed - thr) / scale).clamp(0.0, 1.0)
    return F.binary_cross_entropy(p, target)


def velocity_rigidity_loss(x, velocity, cell_id=None, dino_feat=None,
                           pairs=None, max_cells=512, pairs_per_cell=32,
                           sigma_f=0.25, eps=1e-8):
    """Instantaneous rigid-motion coherence for the Eulerian velocity field.

    For a rigid group, pairwise distances should not change instantaneously:
        d/dt ||x_i - x_j||^2 = 2 (x_i - x_j)^T (v_i - v_j) = 0.

    This is complementary to the displacement/SE(3) losses.  It directly acts on
    the current velocity field, so neighboring Gaussians cannot start/stop at
    very different times while still being locally labelled as a rigid group.
    """
    if x is None or velocity is None or x.numel() == 0 or velocity.numel() == 0:
        dev = x.device if x is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev)
    if pairs is None:
        pairs = sample_pairs_by_cell(cell_id, max_cells=max_cells, pairs_per_cell=pairs_per_cell)
    if pairs is None:
        return torch.tensor(0.0, device=x.device)
    src, dst = pairs
    if src.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    r = (x[src].detach() - x[dst].detach())
    dv = velocity[src] - velocity[dst]
    r2 = (r ** 2).sum(dim=-1, keepdim=True).clamp_min(float(eps))
    radial_rate = (r * dv).sum(dim=-1, keepdim=True) / torch.sqrt(r2)
    w = dino_pair_weights(dino_feat, src, dst, sigma_f=sigma_f)
    return (w * radial_rate ** 2).mean()


def cell_dynamic_label_coherence_loss(dynamic_prob, cell_id, max_cells=512, min_points=2):
    """Encourage local cells not to contain arbitrary mixtures of static/dynamic logits.

    This does not force all cells dynamic.  Sparsity + the speed target decide
    which cells move, while this term prevents the checkerboard/late-follower
    behaviour where adjacent Gaussians of the same local cell receive very
    different dynamic probabilities.
    """
    if dynamic_prob is None or cell_id is None or dynamic_prob.numel() == 0 or cell_id.numel() == 0:
        dev = dynamic_prob.device if dynamic_prob is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev)
    p = dynamic_prob.reshape(-1, 1)
    device = p.device
    losses = []
    with torch.no_grad():
        active = torch.unique(cell_id)
        if active.numel() > int(max_cells):
            active = active[torch.randperm(active.numel(), device=device)[:int(max_cells)]]
    for cid in active.tolist():
        idx = torch.where(cell_id == cid)[0]
        if idx.numel() < int(min_points):
            continue
        vals = p[idx]
        losses.append(((vals - vals.mean().detach()) ** 2).mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _weighted_kabsch_transform(src, dst, weights=None, eps=1e-8):
    """Weighted rigid transform fitting dst ~= src @ R.T + t.

    Args:
        src: [N,3] canonical/source points.
        dst: [N,3] transported/target points.
        weights: optional [N] or [N,1] non-negative weights.

    Returns:
        R: [3,3], t: [3]
    """
    if weights is None:
        weights = torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype)
    else:
        weights = weights.reshape(-1, 1).to(device=src.device, dtype=src.dtype).clamp_min(0.0)
    wsum = weights.sum().clamp_min(float(eps))
    src_mean = (weights * src).sum(dim=0, keepdim=True) / wsum
    dst_mean = (weights * dst).sum(dim=0, keepdim=True) / wsum
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = (weights * src_c).t().matmul(dst_c)
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t().matmul(U.t())
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1, :] *= -1
        R = Vh.t().matmul(U.t())
    t = dst_mean.squeeze(0) - src_mean.squeeze(0).matmul(R.t())
    return R, t


def _standardize_feature(x, eps=1e-6):
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(eps))
    return (x - mu) / sd


def _tiny_kmeans(features, k, iters=8):
    """Small torch k-means used only under no_grad for part assignment.

    Deterministic-ish initialization: choose points uniformly along the first
    principal-ish dimension after sorting by feature norm. This avoids importing
    sklearn and keeps the training script self-contained.
    """
    n = features.shape[0]
    k = int(max(1, min(int(k), int(n))))
    if k <= 1:
        return torch.zeros(n, dtype=torch.long, device=features.device)
    score = features[:, 0]
    order = torch.argsort(score)
    picks = torch.linspace(0, n - 1, steps=k, device=features.device).long().clamp(0, n - 1)
    centers = features[order[picks]].clone()
    labels = torch.zeros(n, dtype=torch.long, device=features.device)
    for _ in range(int(max(1, iters))):
        dist2 = torch.cdist(features, centers).pow(2)
        labels = torch.argmin(dist2, dim=1)
        new_centers = []
        for c in range(k):
            idx = torch.where(labels == c)[0]
            if idx.numel() == 0:
                new_centers.append(centers[c])
            else:
                new_centers.append(features[idx].mean(dim=0))
        centers = torch.stack(new_centers, dim=0)
    return labels



def _skew_matrix(x):
    """Return [N,3,3] skew matrices so skew(x) @ y = x cross y."""
    z = torch.zeros_like(x[..., 0])
    X = torch.stack(
        [
            z, -x[..., 2], x[..., 1],
            x[..., 2], z, -x[..., 0],
            -x[..., 1], x[..., 0], z,
        ],
        dim=-1,
    )
    return X.reshape(*x.shape[:-1], 3, 3)


def _select_dynamic_indices_from_prob(p, min_points=128, max_points=8192, threshold=0.5):
    """Select dynamic points with a robust top-probability fallback."""
    n = p.numel()
    device = p.device
    with torch.no_grad():
        keep = p.detach() >= float(threshold)
        if int(keep.sum().item()) < int(min_points):
            k_keep = min(n, max(int(min_points), int(0.1 * n)))
            order = torch.argsort(p.detach(), descending=True)
            keep = torch.zeros(n, dtype=torch.bool, device=device)
            keep[order[:k_keep]] = True
        idx = torch.where(keep)[0]
        if int(max_points) > 0 and idx.numel() > int(max_points):
            score = p[idx].detach()
            order = torch.argsort(score, descending=True)
            idx = idx[order[: int(max_points)]]
    return idx


def _fit_weighted_twist(x, velocity, weights=None, ridge=1e-6, detach_fit=True):
    """Fit the Eulerian rigid-body twist v(x) = omega x x + u.

    Unknown xi = [omega_x, omega_y, omega_z, u_x, u_y, u_z].  For each point:

        v_i = -skew(x_i) omega + u

    The fit is weighted least squares on the current velocity field.  By default
    the fit itself is detached, and gradients flow only through the residual
    evaluated with live x/velocity tensors.
    """
    xf = x.detach() if detach_fit else x
    vf = velocity.detach() if detach_fit else velocity
    n = xf.shape[0]
    if weights is None:
        w = torch.ones(n, device=xf.device, dtype=xf.dtype)
    else:
        w = weights.reshape(-1).to(device=xf.device, dtype=xf.dtype).clamp_min(0.0)
    sqrt_w = torch.sqrt(w.clamp_min(1e-8))

    A = torch.zeros(n, 3, 6, device=xf.device, dtype=xf.dtype)
    A[:, :, :3] = -_skew_matrix(xf)
    eye = torch.eye(3, device=xf.device, dtype=xf.dtype).view(1, 3, 3).expand(n, 3, 3)
    A[:, :, 3:] = eye
    A_flat = A.reshape(-1, 6)
    b_flat = vf.reshape(-1)
    ww = sqrt_w.repeat_interleave(3)
    Aw = A_flat * ww[:, None]
    bw = b_flat * ww
    AtA = Aw.t().matmul(Aw)
    Atb = Aw.t().matmul(bw)
    reg = float(ridge) * torch.eye(6, device=xf.device, dtype=xf.dtype)
    try:
        xi = torch.linalg.solve(AtA + reg, Atb)
    except RuntimeError:
        xi = torch.linalg.pinv(AtA + reg).matmul(Atb)
    omega = xi[:3]
    trans = xi[3:]
    if detach_fit:
        omega = omega.detach()
        trans = trans.detach()
    return omega, trans


def _twist_predict_velocity(x, omega, trans):
    return torch.cross(omega.reshape(1, 3).expand_as(x), x, dim=-1) + trans.reshape(1, 3)


def global_twist_velocity_loss(
    x,
    velocity,
    dynamic_prob=None,
    num_parts=1,
    dynamic_threshold=0.5,
    min_points=128,
    max_points=8192,
    cluster_iters=8,
    robust_delta=0.03,
    detach_fit=True,
    ridge=1e-6,
):
    """Eulerian rigid-motion prior: fit one twist per moving part.

    This is the velocity-field equivalent of a global SE(3) constraint.  It does
    not ask positions to match a rigid transform; it asks the learned Eulerian
    velocity field itself to be a rigid-body field:

        v_i(t) ~= omega_k(t) x x_i(t) + u_k(t)

    The fitted twist is detached by default, so gradients update the velocity
    field that produced x/v rather than differentiating through the linear solve.
    """
    if x is None or velocity is None or x.numel() == 0 or velocity.numel() == 0:
        dev = x.device if x is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev), {"parts_used": 0, "points_used": 0, "omega_norm_mean": 0.0, "residual_mean": 0.0}
    n = x.shape[0]
    device = x.device
    dtype = x.dtype
    if dynamic_prob is None or dynamic_prob.numel() == 0 or dynamic_prob.reshape(-1).numel() != n:
        p = torch.ones(n, device=device, dtype=dtype)
    else:
        p = dynamic_prob.reshape(-1).to(device=device, dtype=dtype).clamp(0.0, 1.0)

    idx = _select_dynamic_indices_from_prob(p, min_points=min_points, max_points=max_points, threshold=dynamic_threshold)
    if idx.numel() < int(min_points):
        return torch.tensor(0.0, device=device), {"parts_used": 0, "points_used": int(idx.numel()), "omega_norm_mean": 0.0, "residual_mean": 0.0}

    x_all = x[idx]
    v_all = velocity[idx]
    w_all = p[idx].detach().clamp_min(1e-4)

    with torch.no_grad():
        k = max(1, int(num_parts))
        if k <= 1:
            labels = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        else:
            v_dir = v_all.detach() / torch.linalg.norm(v_all.detach(), dim=-1, keepdim=True).clamp_min(1e-8)
            feat = torch.cat([_standardize_feature(x_all.detach()), _standardize_feature(v_dir)], dim=-1)
            labels = _tiny_kmeans(feat, k=k, iters=int(cluster_iters))

    losses = []
    omega_norms = []
    residual_means = []
    used_parts = 0
    used_points = 0
    delta = float(robust_delta)
    for part_id in range(max(1, int(num_parts))):
        mask = labels == int(part_id)
        if int(mask.sum().item()) < int(min_points):
            continue
        xp = x_all[mask]
        vp = v_all[mask]
        wp = w_all[mask]
        omega, trans = _fit_weighted_twist(xp, vp, weights=wp, ridge=ridge, detach_fit=detach_fit)
        pred = _twist_predict_velocity(xp, omega, trans)
        residual = torch.linalg.norm(vp - pred, dim=-1)
        if delta > 0.0:
            per = torch.where(
                residual < delta,
                0.5 * residual.pow(2) / max(delta, 1e-8),
                residual - 0.5 * delta,
            )
        else:
            per = residual.pow(2)
        losses.append((wp * per).sum() / wp.sum().clamp_min(1e-8))
        omega_norms.append(torch.linalg.norm(omega.detach()))
        residual_means.append(residual.detach().mean())
        used_parts += 1
        used_points += int(mask.sum().item())

    if not losses:
        return torch.tensor(0.0, device=device), {"parts_used": 0, "points_used": int(idx.numel()), "omega_norm_mean": 0.0, "residual_mean": 0.0}
    return torch.stack(losses).mean(), {
        "parts_used": int(used_parts),
        "points_used": int(used_points),
        "omega_norm_mean": float(torch.stack(omega_norms).mean().item()) if omega_norms else 0.0,
        "residual_mean": float(torch.stack(residual_means).mean().item()) if residual_means else 0.0,
    }


def global_twist_residual_scores(
    x,
    velocity,
    dynamic_prob=None,
    num_parts=1,
    dynamic_threshold=0.5,
    min_points=128,
    max_points=8192,
    cluster_iters=8,
    ridge=1e-6,
):
    """Detached per-Gaussian residual to the fitted global Eulerian twist.

    Returns [N] with nonzero scores only on selected dynamic candidates.  This is
    used by adaptive-grid splitting to refine cells whose velocities disagree
    with the global rigid-body field, especially mixed static/dynamic hinge cells.
    """
    if x is None or velocity is None or x.numel() == 0 or velocity.numel() == 0:
        dev = x.device if x is not None else torch.device("cuda")
        return torch.zeros(0, device=dev)
    with torch.no_grad():
        n = x.shape[0]
        device = x.device
        dtype = x.dtype
        if dynamic_prob is None or dynamic_prob.numel() == 0 or dynamic_prob.reshape(-1).numel() != n:
            p = torch.ones(n, device=device, dtype=dtype)
        else:
            p = dynamic_prob.reshape(-1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        out = torch.zeros(n, device=device, dtype=dtype)
        idx = _select_dynamic_indices_from_prob(p, min_points=min_points, max_points=max_points, threshold=dynamic_threshold)
        if idx.numel() < int(min_points):
            return out
        xs = x.detach()[idx]
        vs = velocity.detach()[idx]
        ws = p.detach()[idx].clamp_min(1e-4)
        k = max(1, int(num_parts))
        if k <= 1:
            labels = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        else:
            v_dir = vs / torch.linalg.norm(vs, dim=-1, keepdim=True).clamp_min(1e-8)
            feat = torch.cat([_standardize_feature(xs), _standardize_feature(v_dir)], dim=-1)
            labels = _tiny_kmeans(feat, k=k, iters=int(cluster_iters))
        for part_id in range(k):
            mask = labels == int(part_id)
            if int(mask.sum().item()) < int(min_points):
                continue
            xp = xs[mask]
            vp = vs[mask]
            wp = ws[mask]
            omega, trans = _fit_weighted_twist(xp, vp, weights=wp, ridge=ridge, detach_fit=True)
            pred = _twist_predict_velocity(xp, omega, trans)
            residual = torch.linalg.norm(vp - pred, dim=-1)
            # Weight by dynamic confidence: static cells far away should not split
            # merely because their position would have nonzero rigid-link velocity.
            out[idx[mask]] = residual * wp
        if out.numel() > 8 and (out > 0).any():
            nz = out[out > 0]
            # Perfect/near-perfect rigid fields should not become high scores
            # just because of normalization by numerical noise.
            if float(nz.max().item()) < 1e-6:
                return torch.zeros_like(out)
            scale = torch.quantile(nz, 0.95).clamp_min(1e-6)
            out = (out / scale).clamp(0.0, 10.0)
        return out.detach()



def _parse_float_sequence(values, default=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if values is None:
        return list(default)
    if isinstance(values, str):
        vals = []
        for tok in values.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
        return vals if vals else list(default)
    try:
        vals = [float(v) for v in values]
        return vals if vals else list(default)
    except Exception:
        return list(default)


def _robust_unit_score(x, q=0.95, eps=1e-8):
    """Map nonnegative scores to roughly [0,1] with a detached robust scale."""
    x = x.detach().reshape(-1)
    if x.numel() == 0:
        return x
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x)).clamp_min(0.0)
    if float(x.max().item()) <= float(eps):
        return torch.zeros_like(x)
    qq = min(max(float(q), 0.5), 0.999)
    scale = torch.quantile(x, qq).clamp_min(float(eps))
    return (x / scale).clamp(0.0, 1.0)


@torch.no_grad()
def _sample_indices(n, max_points, device):
    if int(max_points) > 0 and int(n) > int(max_points):
        return torch.randperm(int(n), device=device)[: int(max_points)]
    return torch.arange(int(n), device=device)


@torch.no_grad()
def compute_windowed_motion_scores(
    x0,
    motion_field,
    grid=None,
    gaussian_cell_id=None,
    current_velocity=None,
    times=(0.0, 0.25, 0.5, 0.75, 1.0),
    integration_steps=4,
    max_points=4096,
    speed_weight=0.25,
    displacement_weight=0.75,
):
    """Detached motion score for dynamic-label supervision.

    This is not instantaneous speed only. It measures motion over a temporal
    window, so revolute hinge-region Gaussians can become dynamic even when
    their velocity at the currently sampled frame is small.
    """
    if x0 is None or x0.numel() == 0:
        dev = x0.device if x0 is not None else torch.device("cuda")
        z = torch.zeros(0, device=dev)
        return torch.zeros(0, dtype=torch.long, device=dev), z, z, z, torch.zeros(0, 3, device=dev)

    n = int(x0.shape[0])
    device = x0.device
    idx = _sample_indices(n, max_points=max_points, device=device)
    xs = x0.detach()[idx]

    if gaussian_cell_id is not None and torch.is_tensor(gaussian_cell_id) and gaussian_cell_id.numel() == n:
        cid = gaussian_cell_id.detach()[idx]
    else:
        cid = None

    times = _parse_float_sequence(times)
    times = sorted(set([min(max(float(t), 0.0), 1.0) for t in times]))
    use_times = [t for t in times if abs(float(t)) > 1e-8]
    if not use_times:
        use_times = [1.0]

    disp_norms = []
    disp_vecs = []
    if motion_field is None:
        for _t in use_times:
            dv = torch.zeros_like(xs)
            disp_vecs.append(dv)
            disp_norms.append(torch.zeros(xs.shape[0], device=device, dtype=xs.dtype))
    else:
        for t in use_times:
            try:
                xt, _vt = motion_field.integrate(
                    x0=xs,
                    t0=0.0,
                    t1=float(t),
                    steps=int(integration_steps),
                    grid=grid,
                    gaussian_cell_id=cid,
                )
                dv = (xt.detach() - xs)
            except Exception:
                dv = torch.zeros_like(xs)
            disp_vecs.append(dv)
            disp_norms.append(torch.linalg.norm(dv, dim=-1))

    disp_stack = torch.stack(disp_norms, dim=0)
    max_disp, argmax = disp_stack.max(dim=0)
    disp_vec_stack = torch.stack(disp_vecs, dim=0)
    gather = argmax.view(1, -1, 1).expand(1, xs.shape[0], 3)
    max_disp_vec = torch.gather(disp_vec_stack, dim=0, index=gather).squeeze(0)

    disp_score = _robust_unit_score(max_disp, q=0.95)

    if current_velocity is not None and current_velocity.numel() > 0 and current_velocity.shape[0] == n:
        speed = torch.linalg.norm(current_velocity.detach()[idx], dim=-1)
        speed_score = _robust_unit_score(speed, q=0.95)
    else:
        speed_score = torch.zeros_like(disp_score)

    sw = max(float(speed_weight), 0.0)
    dw = max(float(displacement_weight), 0.0)
    norm = max(sw + dw, 1e-8)
    motion_score = (sw * speed_score + dw * disp_score) / norm
    motion_score = torch.where(torch.isfinite(motion_score), motion_score, torch.zeros_like(motion_score)).clamp(0.0, 1.0)

    return idx, motion_score.detach(), disp_score.detach(), speed_score.detach(), max_disp_vec.detach()


def windowed_dynamic_motion_target_loss(
    dynamic_prob,
    x0,
    motion_field,
    grid=None,
    gaussian_cell_id=None,
    current_velocity=None,
    times=(0.0, 0.25, 0.5, 0.75, 1.0),
    integration_steps=4,
    max_points=4096,
    quantile=0.75,
    temperature=0.20,
    speed_weight=0.25,
    displacement_weight=0.75,
):
    """BCE target for dynamic logits from temporal motion, not speed only.

    The target is detached, so gradients update dynamic logits only.  The motion
    field is used as a teacher signal, not optimized through this target.
    """
    if dynamic_prob is None or dynamic_prob.numel() == 0:
        dev = x0.device if x0 is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev), {"motion_target_points": 0}

    p_all = dynamic_prob.reshape(-1)
    idx, motion_score, disp_score, speed_score, _disp_vec = compute_windowed_motion_scores(
        x0=x0,
        motion_field=motion_field,
        grid=grid,
        gaussian_cell_id=gaussian_cell_id,
        current_velocity=current_velocity,
        times=times,
        integration_steps=integration_steps,
        max_points=max_points,
        speed_weight=speed_weight,
        displacement_weight=displacement_weight,
    )
    if idx.numel() == 0:
        return torch.tensor(0.0, device=p_all.device), {"motion_target_points": 0}

    p = p_all[idx].clamp(1e-6, 1.0 - 1e-6)
    with torch.no_grad():
        if float(motion_score.max().item()) <= 1e-8:
            target = torch.zeros_like(motion_score)
            thr = torch.tensor(0.0, device=motion_score.device)
            scale = torch.tensor(1.0, device=motion_score.device)
        else:
            q = float(quantile)
            q = q / 100.0 if q > 1.0 else q
            q = min(max(q, 0.0), 1.0)
            thr = torch.quantile(motion_score, q)
            scale = torch.clamp(motion_score.std(unbiased=False) * float(temperature), min=1e-4)
            target = torch.sigmoid((motion_score - thr) / scale).clamp(0.0, 1.0)

    loss = F.binary_cross_entropy(p, target)
    info = {
        "motion_target_points": int(idx.numel()),
        "motion_target_mean": float(target.mean().detach().item()),
        "motion_target_max": float(target.max().detach().item()) if target.numel() else 0.0,
        "motion_score_mean": float(motion_score.mean().detach().item()) if motion_score.numel() else 0.0,
        "motion_score_threshold": float(thr.detach().item()),
        "motion_disp_score_mean": float(disp_score.mean().detach().item()) if disp_score.numel() else 0.0,
        "motion_speed_score_mean": float(speed_score.mean().detach().item()) if speed_score.numel() else 0.0,
    }
    return loss, info


def motion_aware_dynamic_label_coherence_loss(
    dynamic_prob,
    x0,
    motion_field,
    grid=None,
    gaussian_cell_id=None,
    current_velocity=None,
    dino_feat=None,
    times=(0.0, 0.25, 0.5, 0.75, 1.0),
    integration_steps=4,
    max_points=2048,
    k=8,
    sigma_x=-1.0,
    sigma_motion=0.25,
    sigma_f=0.25,
    boundary_weight=0.25,
    boundary_margin=0.50,
):
    """Boundary-preserving dynamic-label coherence.

    This is the dynamic analogue of static 3D semantic/DINO coherence:
      - close points with similar motion/appearance are encouraged to share a label;
      - close points with different temporal motion are encouraged to have a sharp
        static/dynamic boundary.

    Smooth term:
        w_same(i,j) * ||p_i - p_j||^2

    Boundary term:
        w_boundary(i,j) * max(0, margin - |p_i - p_j|)^2

    The motion/DINO/spatial weights are detached.  Gradients flow only to
    dynamic logits.
    """
    if dynamic_prob is None or dynamic_prob.numel() == 0 or x0 is None or x0.numel() == 0:
        dev = dynamic_prob.device if dynamic_prob is not None else (x0.device if x0 is not None else torch.device("cuda"))
        return torch.tensor(0.0, device=dev), {"motion_label_pairs": 0}

    n = int(x0.shape[0])
    idx, _motion_score, _disp_score, _speed_score, disp_vec = compute_windowed_motion_scores(
        x0=x0,
        motion_field=motion_field,
        grid=grid,
        gaussian_cell_id=gaussian_cell_id,
        current_velocity=current_velocity,
        times=times,
        integration_steps=integration_steps,
        max_points=max_points,
        speed_weight=0.0,
        displacement_weight=1.0,
    )
    if idx.numel() < 4:
        return torch.tensor(0.0, device=x0.device), {"motion_label_pairs": 0}

    xs = x0.detach()[idx]
    p = dynamic_prob.reshape(-1, 1)[idx].clamp(1e-6, 1.0 - 1e-6)
    m = disp_vec.detach()
    M = int(idx.numel())
    kk = min(max(int(k), 1), max(M - 1, 1))

    with torch.no_grad():
        dist_x2 = torch.cdist(xs, xs).pow(2)
        dist_x2.fill_diagonal_(float("inf"))
        nn = torch.topk(dist_x2, k=kk, largest=False).indices
        src = torch.arange(M, device=x0.device).view(-1, 1).expand(M, kk).reshape(-1)
        dst = nn.reshape(-1)

        dx2 = dist_x2[src, dst]
        finite = torch.isfinite(dx2)
        if not finite.any():
            return torch.tensor(0.0, device=x0.device), {"motion_label_pairs": 0}

        if float(sigma_x) <= 0:
            sx2 = torch.median(dx2[finite]).clamp_min(1e-8)
        else:
            sx2 = torch.tensor(float(sigma_x) ** 2, device=x0.device, dtype=x0.dtype).clamp_min(1e-8)

        dm2 = ((m[src] - m[dst]) ** 2).sum(dim=-1)
        sm = float(sigma_motion)
        if sm <= 0:
            sm2 = torch.median(dm2[finite]).clamp_min(1e-8)
        else:
            sm2 = torch.tensor(sm ** 2, device=x0.device, dtype=x0.dtype).clamp_min(1e-8)

        w_spatial = torch.exp(-dx2 / sx2)
        w_motion_same = torch.exp(-dm2 / sm2)
        w_motion_diff = (1.0 - w_motion_same).clamp(0.0, 1.0)

        w_app = torch.ones_like(w_spatial)
        if dino_feat is not None and torch.is_tensor(dino_feat) and dino_feat.numel() > 0 and dino_feat.shape[0] == n:
            f = F.normalize(dino_feat.detach()[idx], dim=-1)
            sim = (f[src] * f[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            fd = (1.0 - sim).clamp_min(0.0)
            w_app = torch.exp(-fd / max(float(sigma_f), 1e-6))

        # Same-motion neighbors behave like static 3D coherence.
        w_same = w_spatial * w_motion_same * w_app

        # Different-motion but spatially close neighbors define a boundary.
        # The appearance factor keeps this focused on plausible same-object/local
        # contacts rather than unrelated far/visually distinct points.
        w_boundary = w_spatial * w_motion_diff * w_app

        w_same = torch.where(torch.isfinite(w_same), w_same, torch.zeros_like(w_same)).detach()
        w_boundary = torch.where(torch.isfinite(w_boundary), w_boundary, torch.zeros_like(w_boundary)).detach()

    diff = (p[src] - p[dst]).abs().reshape(-1)
    smooth = (w_same * diff.pow(2)).sum() / w_same.sum().clamp_min(1e-8)

    bweight = max(float(boundary_weight), 0.0)
    if bweight > 0.0:
        margin = max(float(boundary_margin), 0.0)
        boundary = (w_boundary * F.relu(margin - diff).pow(2)).sum() / w_boundary.sum().clamp_min(1e-8)
        loss = smooth + bweight * boundary
    else:
        boundary = torch.tensor(0.0, device=x0.device, dtype=x0.dtype)
        loss = smooth

    return loss, {
        "motion_label_pairs": int(src.numel()),
        "motion_label_weight_mean": float(w_same.mean().detach().item()) if w_same.numel() else 0.0,
        "motion_boundary_weight_mean": float(w_boundary.mean().detach().item()) if w_boundary.numel() else 0.0,
        "L_motion_label_smooth_inner": float(smooth.detach().item()),
        "L_motion_label_boundary_inner": float(boundary.detach().item()),
    }


def dynamic_pairwise_position_rigidity_loss(
    x0,
    motion_field,
    grid=None,
    gaussian_cell_id=None,
    dynamic_prob=None,
    dino_feat=None,
    times=(0.0, 0.25, 0.5, 0.75, 1.0),
    integration_steps=4,
    max_points=4096,
    k=8,
    dynamic_threshold=0.5,
    min_points=128,
    robust_delta=0.01,
    sigma_f=0.25,
):
    """Preserve local pairwise distances of neighboring dynamic points.

    For a rigid object, the distance between neighboring material points should
    remain constant over time.  This loss samples dynamic/foreground Gaussians,
    builds KNN pairs in canonical space, transports the same points to several
    times, and penalizes distance changes:

        ||x_i(t) - x_j(t)|| ~= ||x_i(0) - x_j(0)||

    The pair graph and weights are detached.  Gradients flow through the
    transported positions to motion_field/motion_grid.
    """
    if x0 is None or x0.numel() == 0 or motion_field is None:
        dev = x0.device if x0 is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev), {"rigidity_pairs": 0, "rigidity_times": 0}

    n = int(x0.shape[0])
    device = x0.device
    dtype = x0.dtype

    if dynamic_prob is None or dynamic_prob.numel() == 0 or dynamic_prob.reshape(-1).numel() != n:
        p = torch.ones(n, device=device, dtype=dtype)
    else:
        p = dynamic_prob.reshape(-1).to(device=device, dtype=dtype).clamp(0.0, 1.0)

    with torch.no_grad():
        idx = _select_dynamic_indices_from_prob(
            p,
            min_points=int(min_points),
            max_points=int(max_points),
            threshold=float(dynamic_threshold),
        )
        if idx.numel() < int(min_points):
            return torch.tensor(0.0, device=device), {"rigidity_pairs": 0, "rigidity_times": 0, "rigidity_points": int(idx.numel())}

        xs0 = x0.detach()[idx]
        M = int(xs0.shape[0])
        kk = min(max(int(k), 1), max(M - 1, 1))

        dist2 = torch.cdist(xs0, xs0).pow(2)
        dist2.fill_diagonal_(float("inf"))
        nn = torch.topk(dist2, k=kk, largest=False).indices
        src = torch.arange(M, device=device).view(-1, 1).expand(M, kk).reshape(-1)
        dst = nn.reshape(-1)

        d0 = torch.linalg.norm(xs0[src] - xs0[dst], dim=-1).clamp_min(1e-8)
        pair_w = torch.sqrt((p[idx][src] * p[idx][dst]).clamp_min(1e-6)).detach()

        if dino_feat is not None and torch.is_tensor(dino_feat) and dino_feat.numel() > 0 and dino_feat.shape[0] == n:
            f = F.normalize(dino_feat.detach()[idx], dim=-1)
            sim = (f[src] * f[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            fd = (1.0 - sim).clamp_min(0.0)
            pair_w = pair_w * torch.exp(-fd / max(float(sigma_f), 1e-6))

        pair_w = torch.where(torch.isfinite(pair_w), pair_w, torch.zeros_like(pair_w)).detach()
        d0 = d0.detach()

        if gaussian_cell_id is not None and torch.is_tensor(gaussian_cell_id) and gaussian_cell_id.numel() == n:
            cid = gaussian_cell_id.detach()[idx]
        else:
            cid = None

    ts = _parse_float_sequence(times)
    ts = sorted(set([min(max(float(t), 0.0), 1.0) for t in ts]))
    ts = [t for t in ts if abs(t) > 1e-8]
    if not ts:
        return torch.tensor(0.0, device=device), {"rigidity_pairs": int(src.numel()), "rigidity_times": 0, "rigidity_points": int(idx.numel())}

    xs = x0[idx]  # canonical positions are frozen, but keep tensor device/dtype.
    losses = []
    mean_abs = []
    delta = float(robust_delta)
    for t in ts:
        xt, _ = motion_field.integrate(
            x0=xs,
            t0=0.0,
            t1=float(t),
            steps=int(integration_steps),
            grid=grid,
            gaussian_cell_id=cid,
        )
        dt = torch.linalg.norm(xt[src] - xt[dst], dim=-1).clamp_min(1e-8)
        residual = dt - d0
        abs_r = residual.abs()
        if delta > 0.0:
            per = torch.where(
                abs_r < delta,
                0.5 * residual.pow(2) / max(delta, 1e-8),
                abs_r - 0.5 * delta,
            )
        else:
            per = residual.pow(2)
        losses.append((pair_w * per).sum() / pair_w.sum().clamp_min(1e-8))
        mean_abs.append(abs_r.detach().mean())

    if not losses:
        return torch.tensor(0.0, device=device), {"rigidity_pairs": int(src.numel()), "rigidity_times": 0, "rigidity_points": int(idx.numel())}

    return torch.stack(losses).mean(), {
        "rigidity_pairs": int(src.numel()),
        "rigidity_times": int(len(ts)),
        "rigidity_points": int(idx.numel()),
        "rigidity_mean_abs_dist_error": float(torch.stack(mean_abs).mean().item()) if mean_abs else 0.0,
    }


def motion_contrastive_dynamic_label_loss(
    dynamic_prob,
    x0,
    motion_field,
    grid=None,
    gaussian_cell_id=None,
    times=(0.0, 0.25, 0.5, 0.75, 1.0),
    integration_steps=4,
    max_points=2048,
    num_pairs=32768,
    target_quantile=0.95,
    target_gamma=1.0,
    loss_type="smooth_l1",
    huber_beta=0.10,
):
    """Pairwise contrastive supervision for dynamic labels from temporal motion.

    For sampled pairs (i,j), compute a detached temporal-motion difference:

        d_motion(i,j) =
            normalized RMS_t ||(x_i(t)-x_i(0)) - (x_j(t)-x_j(0))||

    and match it with the dynamic-label difference:

        d_label(i,j) = |p_dyn_i - p_dyn_j|.

    The target d_motion is detached, so this loss updates only _dynamic_logit.
    It does NOT push the motion field toward zero.  The motion field is still
    trained by RGB/depth/DINO transport/twist/cycle; this loss only says that
    points with different temporal behavior should have different labels, while
    points with similar temporal behavior should have similar labels.

    Unlike the previous local KNN position-rigidity loss, pairs are sampled
    globally from the dynamic candidate set, not only from neighboring points.
    """
    if dynamic_prob is None or dynamic_prob.numel() == 0 or x0 is None or x0.numel() == 0:
        dev = dynamic_prob.device if dynamic_prob is not None else (x0.device if x0 is not None else torch.device("cuda"))
        return torch.tensor(0.0, device=dev), {"contrastive_pairs": 0}

    n = int(x0.shape[0])
    device = x0.device
    dtype = x0.dtype
    if n < 4:
        return torch.tensor(0.0, device=device), {"contrastive_pairs": 0}

    # Sample points first, then sample arbitrary pairs among them.
    idx = _sample_indices(n, max_points=max_points, device=device)
    if idx.numel() < 4:
        return torch.tensor(0.0, device=device), {"contrastive_pairs": 0}

    p = dynamic_prob.reshape(-1)[idx].clamp(1e-6, 1.0 - 1e-6)
    xs = x0.detach()[idx]
    M = int(idx.numel())

    if gaussian_cell_id is not None and torch.is_tensor(gaussian_cell_id) and gaussian_cell_id.numel() == n:
        cid = gaussian_cell_id.detach()[idx]
    else:
        cid = None

    ts = _parse_float_sequence(times)
    ts = sorted(set([min(max(float(t), 0.0), 1.0) for t in ts]))
    ts = [t for t in ts if abs(t) > 1e-8]
    if not ts:
        ts = [1.0]

    # Build detached temporal signatures.  No gradient to the motion field.
    with torch.no_grad():
        disp_list = []
        if motion_field is None:
            for _t in ts:
                disp_list.append(torch.zeros_like(xs))
        else:
            for t in ts:
                try:
                    xt, _ = motion_field.integrate(
                        x0=xs,
                        t0=0.0,
                        t1=float(t),
                        steps=int(integration_steps),
                        grid=grid,
                        gaussian_cell_id=cid,
                    )
                    disp_list.append((xt.detach() - xs).detach())
                except Exception:
                    disp_list.append(torch.zeros_like(xs))

        signature = torch.cat(disp_list, dim=-1)  # [M, 3*T]
        signature = torch.where(torch.isfinite(signature), signature, torch.zeros_like(signature))

        P = int(num_pairs)
        if P <= 0:
            P = min(M * 8, 32768)
        src = torch.randint(0, M, (P,), device=device)
        dst = torch.randint(0, M, (P,), device=device)
        neq = src != dst
        src = src[neq]
        dst = dst[neq]
        if src.numel() < 4:
            return torch.tensor(0.0, device=device), {"contrastive_pairs": int(src.numel())}

        # RMS temporal signature distance.  This is the normalized "difference in time".
        raw = torch.linalg.norm(signature[src] - signature[dst], dim=-1)
        raw = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw)).clamp_min(0.0)

        if float(raw.max().item()) <= 1e-12:
            target = torch.zeros_like(raw)
            scale = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            q = min(max(float(target_quantile), 0.50), 0.999)
            scale = torch.quantile(raw, q).clamp_min(1e-8)
            target = (raw / scale).clamp(0.0, 1.0)
            gamma = max(float(target_gamma), 1e-6)
            if abs(gamma - 1.0) > 1e-8:
                target = target.pow(gamma)

    # Dynamic probabilities are normalized labels in [0,1].
    # Their pairwise absolute difference is already a normalized label distance.
    label_diff = (p[src] - p[dst]).abs().clamp(0.0, 1.0)

    if str(loss_type).lower() in ("l2", "mse"):
        per = (label_diff - target).pow(2)
        loss = per.mean()
    elif str(loss_type).lower() in ("l1", "mae"):
        per = (label_diff - target).abs()
        loss = per.mean()
    else:
        # SmoothL1 is stable when targets are noisy early in training.
        loss = F.smooth_l1_loss(label_diff, target, beta=float(huber_beta), reduction="mean")

    with torch.no_grad():
        pos = target > 0.7
        neg = target < 0.3
        return_stats = {
            "contrastive_pairs": int(src.numel()),
            "contrastive_target_mean": float(target.mean().item()) if target.numel() else 0.0,
            "contrastive_target_std": float(target.std().item()) if target.numel() > 1 else 0.0,
            "contrastive_label_diff_mean": float(label_diff.detach().mean().item()) if label_diff.numel() else 0.0,
            "contrastive_motion_scale": float(scale.item()) if torch.is_tensor(scale) else float(scale),
            "contrastive_pos_pairs": float(pos.float().mean().item()) if target.numel() else 0.0,
            "contrastive_neg_pairs": float(neg.float().mean().item()) if target.numel() else 0.0,
        }
    return loss, return_stats


def model_based_dynamic_motion_target_loss(
    dynamic_prob,
    x,
    velocity,
    num_parts=1,
    dynamic_threshold=0.5,
    min_fit_points=128,
    max_fit_points=8192,
    max_points=4096,
    cluster_iters=8,
    ridge=1e-6,
    temperature=-1.0,
    target_power=1.0,
    loss_type="bce",
    huber_beta=0.10,
    fit_score_mode="prob_or_speed",
):
    """Model-based dynamic-logit target from identity-vs-rigid residuals.

    This replaces the old windowed displacement target:

        disp_i = max_t ||x_i(t)-x_i(0)||

    which incorrectly labels only high-displacement points as dynamic and can
    mark hinge-near points as static.

    New target:

        e_static_i = ||v_i||
        e_rigid_i  = min_k ||v_i - (omega_k x x_i + u_k)||
        q_dyn_i    = sigmoid((e_static_i - e_rigid_i) / temperature)

    So a point is dynamic when a fitted moving rigid velocity field explains its
    velocity better than the null/identity/static field, independent of its
    displacement amplitude.

    The target is detached.  Gradients flow only to dynamic_prob/_dynamic_logit,
    not to the velocity field or the twist fit.
    """
    if dynamic_prob is None or x is None or velocity is None:
        dev = dynamic_prob.device if dynamic_prob is not None else (x.device if x is not None else torch.device("cuda"))
        return torch.tensor(0.0, device=dev), {"model_target_points": 0, "model_target_parts_used": 0}
    if dynamic_prob.numel() == 0 or x.numel() == 0 or velocity.numel() == 0:
        return torch.tensor(0.0, device=x.device), {"model_target_points": 0, "model_target_parts_used": 0}

    n = int(x.shape[0])
    device = x.device
    dtype = x.dtype
    p_all = dynamic_prob.reshape(-1).to(device=device, dtype=dtype).clamp(1e-6, 1.0 - 1e-6)
    if p_all.numel() != n or n < 4:
        return torch.tensor(0.0, device=device), {"model_target_points": 0, "model_target_parts_used": 0}

    with torch.no_grad():
        x_det = x.detach()
        v_det = velocity.detach()
        speed = torch.linalg.norm(v_det, dim=-1)
        speed_score = _robust_unit_score(speed, q=0.95)

        mode = str(fit_score_mode).lower()
        if mode == "prob":
            fit_score = p_all.detach()
        elif mode == "speed":
            fit_score = speed_score.detach()
        else:
            # More robust than using current labels alone: early labels may be
            # wrong, but high-speed points still help estimate the moving twist.
            fit_score = torch.maximum(p_all.detach(), speed_score.detach()).clamp(0.0, 1.0)

        fit_idx = _select_dynamic_indices_from_prob(
            fit_score,
            min_points=int(min_fit_points),
            max_points=int(max_fit_points),
            threshold=float(dynamic_threshold),
        )
        if fit_idx.numel() < int(min_fit_points):
            return torch.tensor(0.0, device=device), {
                "model_target_points": 0,
                "model_target_fit_points": int(fit_idx.numel()),
                "model_target_parts_used": 0,
            }

        xf = x_det[fit_idx]
        vf = v_det[fit_idx]
        wf = fit_score[fit_idx].clamp_min(1e-4)

        k = max(1, int(num_parts))
        if k <= 1:
            fit_labels = torch.zeros(fit_idx.numel(), dtype=torch.long, device=device)
        else:
            v_dir = vf / torch.linalg.norm(vf, dim=-1, keepdim=True).clamp_min(1e-8)
            feat = torch.cat([_standardize_feature(xf), _standardize_feature(v_dir)], dim=-1)
            fit_labels = _tiny_kmeans(feat, k=k, iters=int(cluster_iters))

        twists = []
        for part_id in range(k):
            m = fit_labels == int(part_id)
            if int(m.sum().item()) < int(min_fit_points):
                continue
            omega, trans = _fit_weighted_twist(
                xf[m],
                vf[m],
                weights=wf[m],
                ridge=float(ridge),
                detach_fit=True,
            )
            twists.append((omega.detach(), trans.detach()))

        if not twists:
            return torch.tensor(0.0, device=device), {
                "model_target_points": 0,
                "model_target_fit_points": int(fit_idx.numel()),
                "model_target_parts_used": 0,
            }

        idx = _sample_indices(n, max_points=max_points, device=device)
        if idx.numel() < 4:
            return torch.tensor(0.0, device=device), {"model_target_points": int(idx.numel()), "model_target_parts_used": int(len(twists))}

        xs = x_det[idx]
        vs = v_det[idx]

        e_static = torch.linalg.norm(vs, dim=-1)
        rigid_residuals = []
        for omega, trans in twists:
            pred = _twist_predict_velocity(xs, omega, trans)
            rigid_residuals.append(torch.linalg.norm(vs - pred, dim=-1))
        e_rigid = torch.stack(rigid_residuals, dim=0).min(dim=0).values

        residual_diff = e_static - e_rigid
        if float(temperature) > 0:
            temp = torch.tensor(float(temperature), device=device, dtype=dtype)
        else:
            # Robust automatic temperature.  Too small -> hard/unstable target.
            temp = torch.quantile(residual_diff.abs(), 0.75).clamp_min(1e-4)

        target = torch.sigmoid(residual_diff / temp).clamp(0.0, 1.0)
        power = max(float(target_power), 1e-6)
        if abs(power - 1.0) > 1e-8:
            target = target.pow(power)
        target = target.detach()

    # IMPORTANT: take ps outside torch.no_grad(), otherwise this target loss
    # logs a value but cannot update _dynamic_logit.
    ps = p_all[idx]

    if str(loss_type).lower() in ("smooth_l1", "huber"):
        loss = F.smooth_l1_loss(ps, target, beta=float(huber_beta), reduction="mean")
    elif str(loss_type).lower() in ("l1", "mae"):
        loss = (ps - target).abs().mean()
    elif str(loss_type).lower() in ("l2", "mse"):
        loss = (ps - target).pow(2).mean()
    else:
        # p is already sigmoid(dynamic_logit), so use BCE on probabilities.
        loss = F.binary_cross_entropy(ps, target, reduction="mean")

    with torch.no_grad():
        stats = {
            "model_target_points": int(idx.numel()),
            "model_target_fit_points": int(fit_idx.numel()),
            "model_target_parts_used": int(len(twists)),
            "model_target_mean": float(target.mean().item()) if target.numel() else 0.0,
            "model_target_std": float(target.std().item()) if target.numel() > 1 else 0.0,
            "model_target_prob_mean": float(ps.detach().mean().item()) if ps.numel() else 0.0,
            "model_target_e_static_mean": float(e_static.mean().item()) if e_static.numel() else 0.0,
            "model_target_e_rigid_mean": float(e_rigid.mean().item()) if e_rigid.numel() else 0.0,
            "model_target_residual_diff_mean": float(residual_diff.mean().item()) if residual_diff.numel() else 0.0,
            "model_target_temperature": float(temp.item()) if torch.is_tensor(temp) else float(temp),
        }
    return loss, stats


def rigid_model_contrastive_dynamic_label_loss(
    x,
    velocity,
    dynamic_prob,
    num_parts=1,
    dynamic_threshold=0.5,
    min_fit_points=128,
    max_fit_points=8192,
    max_points=2048,
    num_pairs=32768,
    cluster_iters=8,
    ridge=1e-6,
    temperature=-1.0,
    loss_type="smooth_l1",
    huber_beta=0.10,
):
    """Late static-vs-rigid contrastive sharpening for dynamic labels.

    This should run only after rigid motion fitting is already active.

    It fits:
      - identity/static model: v_static(x) = 0
      - k moving rigid velocity models: v_k(x) = omega_k x x + u_k

    Then for each sampled point it computes detached residuals:

        e_static_i = ||v_i||
        e_rigid_i  = min_k ||v_i - (omega_k x x_i + u_k)||

    A continuous dynamic target is:

        q_i = sigmoid((e_static_i - e_rigid_i) / temperature)

    so:
      - q_i -> 0 when identity/null velocity explains point better
      - q_i -> 1 when one fitted rigid moving field explains point better

    Finally it uses a contrastive pair target:

        target_ij = |q_i - q_j|
        label_ij  = |p_dyn_i - p_dyn_j|

    and minimizes SmoothL1(label_ij, stopgrad(target_ij)).

    The fitted twists, residual targets, and pair targets are detached.
    Gradients flow only to _dynamic_logit, not to the motion field.
    """
    if x is None or velocity is None or dynamic_prob is None:
        dev = x.device if x is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev), {"rigid_contrastive_pairs": 0}
    if x.numel() == 0 or velocity.numel() == 0 or dynamic_prob.numel() == 0:
        return torch.tensor(0.0, device=x.device), {"rigid_contrastive_pairs": 0}

    n = int(x.shape[0])
    device = x.device
    dtype = x.dtype
    if n < 4:
        return torch.tensor(0.0, device=device), {"rigid_contrastive_pairs": 0}

    p_all = dynamic_prob.reshape(-1).to(device=device, dtype=dtype).clamp(1e-6, 1.0 - 1e-6)
    if p_all.numel() != n:
        return torch.tensor(0.0, device=device), {"rigid_contrastive_pairs": 0}

    with torch.no_grad():
        # Fit moving rigid fields only from likely-dynamic candidates, with
        # robust top-prob fallback.
        fit_idx = _select_dynamic_indices_from_prob(
            p_all,
            min_points=int(min_fit_points),
            max_points=int(max_fit_points),
            threshold=float(dynamic_threshold),
        )
        if fit_idx.numel() < int(min_fit_points):
            return torch.tensor(0.0, device=device), {
                "rigid_contrastive_pairs": 0,
                "rigid_contrastive_fit_points": int(fit_idx.numel()),
                "rigid_contrastive_parts_used": 0,
            }

        xf = x.detach()[fit_idx]
        vf = velocity.detach()[fit_idx]
        wf = p_all.detach()[fit_idx].clamp_min(1e-4)

        k = max(1, int(num_parts))
        if k <= 1:
            fit_labels = torch.zeros(fit_idx.numel(), dtype=torch.long, device=device)
        else:
            v_dir = vf / torch.linalg.norm(vf, dim=-1, keepdim=True).clamp_min(1e-8)
            feat = torch.cat([_standardize_feature(xf), _standardize_feature(v_dir)], dim=-1)
            fit_labels = _tiny_kmeans(feat, k=k, iters=int(cluster_iters))

        twists = []
        for part_id in range(k):
            m = fit_labels == int(part_id)
            if int(m.sum().item()) < int(min_fit_points):
                continue
            omega, trans = _fit_weighted_twist(
                xf[m],
                vf[m],
                weights=wf[m],
                ridge=float(ridge),
                detach_fit=True,
            )
            twists.append((omega.detach(), trans.detach()))

        if not twists:
            return torch.tensor(0.0, device=device), {
                "rigid_contrastive_pairs": 0,
                "rigid_contrastive_fit_points": int(fit_idx.numel()),
                "rigid_contrastive_parts_used": 0,
            }

        idx = _sample_indices(n, max_points=max_points, device=device)
        if idx.numel() < 4:
            return torch.tensor(0.0, device=device), {"rigid_contrastive_pairs": 0}

        xs = x.detach()[idx]
        vs = velocity.detach()[idx]

        e_static = torch.linalg.norm(vs, dim=-1)
        rigid_residuals = []
        for omega, trans in twists:
            pred = _twist_predict_velocity(xs, omega, trans)
            rigid_residuals.append(torch.linalg.norm(vs - pred, dim=-1))
        e_rigid = torch.stack(rigid_residuals, dim=0).min(dim=0).values

        diff = e_static - e_rigid
        if float(temperature) > 0:
            temp = torch.tensor(float(temperature), device=device, dtype=dtype)
        else:
            temp = torch.quantile(diff.abs(), 0.75).clamp_min(1e-4)
        q_dyn = torch.sigmoid(diff / temp).clamp(0.0, 1.0)

        M = int(idx.numel())
        P = int(num_pairs)
        if P <= 0:
            P = min(M * 8, 32768)
        src = torch.randint(0, M, (P,), device=device)
        dst = torch.randint(0, M, (P,), device=device)
        neq = src != dst
        src = src[neq]
        dst = dst[neq]
        if src.numel() < 4:
            return torch.tensor(0.0, device=device), {"rigid_contrastive_pairs": int(src.numel())}

        target = (q_dyn[src] - q_dyn[dst]).abs().detach().clamp(0.0, 1.0)

    # IMPORTANT: take ps outside torch.no_grad(), otherwise label sharpening
    # cannot update _dynamic_logit.
    ps = p_all[idx]
    label_diff = (ps[src] - ps[dst]).abs().clamp(0.0, 1.0)
    if str(loss_type).lower() in ("l2", "mse"):
        loss = (label_diff - target).pow(2).mean()
    elif str(loss_type).lower() in ("l1", "mae"):
        loss = (label_diff - target).abs().mean()
    else:
        loss = F.smooth_l1_loss(label_diff, target, beta=float(huber_beta), reduction="mean")

    with torch.no_grad():
        return_stats = {
            "rigid_contrastive_pairs": int(src.numel()),
            "rigid_contrastive_fit_points": int(fit_idx.numel()),
            "rigid_contrastive_sample_points": int(idx.numel()),
            "rigid_contrastive_parts_used": int(len(twists)),
            "rigid_contrastive_q_dyn_mean": float(q_dyn.mean().item()) if q_dyn.numel() else 0.0,
            "rigid_contrastive_q_dyn_std": float(q_dyn.std().item()) if q_dyn.numel() > 1 else 0.0,
            "rigid_contrastive_target_mean": float(target.mean().item()) if target.numel() else 0.0,
            "rigid_contrastive_label_diff_mean": float(label_diff.detach().mean().item()) if label_diff.numel() else 0.0,
            "rigid_contrastive_e_static_mean": float(e_static.mean().item()) if e_static.numel() else 0.0,
            "rigid_contrastive_e_rigid_mean": float(e_rigid.mean().item()) if e_rigid.numel() else 0.0,
            "rigid_contrastive_temperature": float(temp.item()) if torch.is_tensor(temp) else float(temp),
        }
    return loss, return_stats


def global_se3_motion_loss(
    x0,
    xt,
    dynamic_prob=None,
    num_parts=1,
    dynamic_threshold=0.5,
    min_points=128,
    max_points=8192,
    cluster_iters=8,
    robust_delta=0.03,
    detach_fit=True,
):
    """Fit one/global SE(3) transform per moving part and penalize residuals.

    This is the rigid-motion prior needed after the velocity-field warmup has
    already discovered a usable static/dynamic separation.  At every dynamic
    iteration and for the current timestamp, it:

      1. selects high-dynamic-probability Gaussians;
      2. optionally splits them into K moving parts;
      3. fits R,t for each part using weighted Kabsch;
      4. penalizes transported positions that deviate from that rigid transform.

    The fitted transforms are detached by default, so gradients route to the
    Eulerian velocity field that generated xt, not to the SVD fitting procedure.
    This prevents the hinge-near cells from learning their own cyclic phase
    while the rest of the dynamic part follows a different coherent motion.
    """
    if x0 is None or xt is None or x0.numel() == 0 or xt.numel() == 0:
        dev = x0.device if x0 is not None else torch.device("cuda")
        return torch.tensor(0.0, device=dev), {"parts_used": 0, "points_used": 0}
    n = x0.shape[0]
    device = x0.device
    if dynamic_prob is None or dynamic_prob.numel() == 0:
        p = torch.ones(n, device=device, dtype=x0.dtype)
    else:
        p = dynamic_prob.reshape(-1).to(device=device, dtype=x0.dtype).clamp(0.0, 1.0)
        if p.numel() != n:
            p = torch.ones(n, device=device, dtype=x0.dtype)

    with torch.no_grad():
        thr = float(dynamic_threshold)
        keep = p.detach() >= thr
        # Robust fallback for early warmup if probabilities are not yet sharp.
        if int(keep.sum().item()) < int(min_points):
            k_keep = min(n, max(int(min_points), int(0.1 * n)))
            order = torch.argsort(p.detach(), descending=True)
            keep = torch.zeros(n, dtype=torch.bool, device=device)
            keep[order[:k_keep]] = True
        idx = torch.where(keep)[0]
        if idx.numel() > int(max_points) > 0:
            # Prefer high-confidence dynamic points, but keep sampling cheap.
            score = p[idx].detach()
            order = torch.argsort(score, descending=True)
            top = idx[order[: int(max_points)]]
            idx = top

    if idx.numel() < int(min_points):
        return torch.tensor(0.0, device=device), {"parts_used": 0, "points_used": int(idx.numel())}

    src_all = x0[idx]
    dst_all = xt[idx]
    w_all = p[idx].detach().clamp_min(1e-4)

    with torch.no_grad():
        k = max(1, int(num_parts))
        if k <= 1:
            labels = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        else:
            disp = (dst_all.detach() - src_all.detach())
            feat = torch.cat([_standardize_feature(src_all.detach()), _standardize_feature(disp)], dim=-1)
            labels = _tiny_kmeans(feat, k=k, iters=int(cluster_iters))

    losses = []
    used_parts = 0
    used_points = 0
    delta = float(robust_delta)
    for part_id in range(max(1, int(num_parts))):
        part_mask = labels == int(part_id)
        if int(part_mask.sum().item()) < int(min_points):
            continue
        src = src_all[part_mask]
        dst = dst_all[part_mask]
        w = w_all[part_mask]
        fit_src = src.detach() if detach_fit else src
        fit_dst = dst.detach() if detach_fit else dst
        try:
            R, t = _weighted_kabsch_transform(fit_src, fit_dst, weights=w.detach())
        except RuntimeError:
            continue
        if detach_fit:
            R = R.detach()
            t = t.detach()
        pred = src.matmul(R.t()) + t
        residual = torch.linalg.norm(dst - pred, dim=-1)
        if delta > 0.0:
            # Smooth-L1/Huber-like robust residual in metric units.
            per = torch.where(
                residual < delta,
                0.5 * residual.pow(2) / max(delta, 1e-8),
                residual - 0.5 * delta,
            )
        else:
            per = residual.pow(2)
        losses.append((w * per).sum() / w.sum().clamp_min(1e-8))
        used_parts += 1
        used_points += int(part_mask.sum().item())

    if not losses:
        return torch.tensor(0.0, device=device), {"parts_used": 0, "points_used": int(idx.numel())}
    return torch.stack(losses).mean(), {"parts_used": int(used_parts), "points_used": int(used_points)}

# -----------------------------------------------------------------------------
# Joint single-rigid-part optimization
# -----------------------------------------------------------------------------

def _joint_zero_like(reference):
    if reference is not None and torch.is_tensor(reference):
        return reference.new_zeros(())
    return torch.tensor(0.0, device="cuda")


def _joint_robust_norm(vector, delta=0.05):
    """Per-point pseudo-Huber vector magnitude."""
    sq = (vector ** 2).sum(dim=-1)
    delta = float(delta)
    if delta <= 0:
        return sq
    d2 = delta * delta
    return d2 * (torch.sqrt(1.0 + sq / d2) - 1.0)


@torch.no_grad()
def _joint_adaptive_temperature(
    residual_difference,
    requested_temperature=-1.0,
    quantile=0.50,
    minimum=1e-3,
    maximum=5e-2,
):
    """Scale identity-vs-rigid targets to the actual residual distribution."""
    if float(requested_temperature) > 0:
        return residual_difference.new_tensor(float(requested_temperature))
    finite = residual_difference.detach().abs()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        value = float(minimum)
    else:
        value = float(torch.quantile(finite, float(quantile)).item())
        value = max(float(minimum), min(float(maximum), value))
    return residual_difference.new_tensor(value)


def joint_single_rigid_losses(
    total_velocity,
    rigid_velocity,
    proposal_velocity,
    dynamic_prob,
    part_field,
    rigid_twist_field,
    current_time,
    robust_delta=0.05,
    label_temperature=-1.0,
    label_temperature_quantile=0.50,
    label_temperature_min=1e-3,
    label_temperature_max=5e-2,
    confident_static_threshold=0.20,
    confident_dynamic_threshold=0.80,
    twist_dt=0.02,
    field_pair_samples=1024,
    field_different_margin=0.20,
    minimum_part_fraction=0.02,
):
    """Correct joint static/dynamic + one-rigid-part objective.

    Gradient routing is deliberately separated:

    * ``twist_fit``: proposal is detached. Only omega(t),u(t) learn to fit the
      motion proposal selected by the current moving assignment.
    * ``proposal_rigid``: rigid velocity is detached. Only the MLP/grid proposal
      learns to agree with the rigid expert for confident moving Gaussians.
    * ``proposal_static``: only confident static Gaussians are driven to zero.
    * ``label``: both velocity experts are detached. Only the per-Gaussian
      cosine part field and its prototypes receive assignment gradients.
    * field clustering losses use only cosine geometry of the learned
      dynamic/static field. XYZ and DINO are initialization-only.
    * ``twist_smooth`` trains only the explicit temporal twist network.
    """
    zero = _joint_zero_like(total_velocity)
    losses = {
        "twist_fit": zero,
        "proposal_rigid": zero,
        "proposal_static": zero,
        "label": zero,
        "field_compact": zero,
        "field_pairwise": zero,
        "prototype_separation": zero,
        "assignment_entropy": zero,
        "part_occupancy": zero,
        "twist_smooth": zero,
    }
    info = {
        "joint_points": 0,
        "joint_q_mean": 0.0,
        "joint_p_mean": 0.0,
        "joint_label_temperature_used": 0.0,
        "joint_static_residual_mean": 0.0,
        "joint_rigid_residual_mean": 0.0,
        "joint_omega_norm": 0.0,
        "joint_translation_norm": 0.0,
    }
    if (
        total_velocity is None
        or rigid_velocity is None
        or proposal_velocity is None
        or dynamic_prob is None
        or total_velocity.numel() == 0
    ):
        return losses, info

    p = dynamic_prob.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    if p.shape[0] != proposal_velocity.shape[0]:
        raise RuntimeError(
            "Joint rigid loss received inconsistent foreground tensor lengths: "
            f"p={p.shape[0]}, proposal={proposal_velocity.shape[0]}"
        )
    if part_field is None:
        raise RuntimeError(
            "Exact joint rigid fitting requires JointPartField over semantic foreground; "
            "refusing probability-only fallback."
        )
    assignment_fg = part_field.assignments_fg()
    if assignment_fg.shape[0] != p.shape[0]:
        raise RuntimeError(
            "JointPartField size does not match semantic-foreground rigid tensors: "
            f"field={assignment_fg.shape[0]}, rigid_points={p.shape[0]}"
        )

    p_det = p.detach()
    proposal_speed = torch.linalg.norm(proposal_velocity.detach(), dim=-1)
    with torch.no_grad():
        finite_speed = proposal_speed[torch.isfinite(proposal_speed)]
        speed_scale = (
            torch.quantile(finite_speed, 0.80).clamp_min(1e-4)
            if finite_speed.numel() > 0
            else proposal_speed.new_tensor(1.0)
        )
        motion_evidence = (proposal_speed / speed_scale).clamp(0.0, 1.0)
        # The optimized cosine clustering, not speed, decides membership of the
        # moving part. Motion evidence only weights the quality of points inside
        # that selected moving slot. Static-slot points never fit omega,u.
        moving_cluster_mask = assignment_fg.detach().argmax(dim=-1) > 0
        twist_weight = moving_cluster_mask.float() * p_det * (0.10 + 0.90 * motion_evidence)
        dynamic_mask = p_det >= float(confident_dynamic_threshold)
        static_mask = p_det <= float(confident_static_threshold)

    # 1) Explicit omega,u fit a detached proposal. This direction is critical:
    # the free proposal cannot collapse toward the zero-initialized twist.
    twist_per = _joint_robust_norm(
        rigid_velocity - proposal_velocity.detach(), robust_delta
    )
    losses["twist_fit"] = (
        (twist_weight * twist_per).sum() / twist_weight.sum().clamp_min(1.0)
    )

    # 2) Proposal field learns the detached rigid model only for confident
    # moving points. Uncertain points stay free to receive rendering evidence.
    if dynamic_mask.any():
        losses["proposal_rigid"] = _joint_robust_norm(
            proposal_velocity[dynamic_mask] - rigid_velocity.detach()[dynamic_mask],
            robust_delta,
        ).mean()

    # 3) Zero-velocity prior only for confident static points.
    if static_mask.any():
        losses["proposal_static"] = _joint_robust_norm(
            proposal_velocity[static_mask], robust_delta
        ).mean()

    # 4) Static-vs-rigid assignment with a temperature matched to the actual
    # residual scale. Both experts are detached: this term trains only labels.
    with torch.no_grad():
        e_static = torch.linalg.norm(proposal_velocity.detach(), dim=-1)
        e_rigid = torch.linalg.norm(
            proposal_velocity.detach() - rigid_velocity.detach(), dim=-1
        )
        residual_difference = e_static - e_rigid
        temp = _joint_adaptive_temperature(
            residual_difference,
            requested_temperature=label_temperature,
            quantile=label_temperature_quantile,
            minimum=label_temperature_min,
            maximum=label_temperature_max,
        )
        q = torch.sigmoid(residual_difference / temp).clamp(1e-4, 1.0 - 1e-4)
    losses["label"] = F.binary_cross_entropy(p, q)

    # 5) Trainable clustering is exclusively the cosine geometry of the
    # dynamic/static field. Geometry and DINO never enter these losses.
    cluster_losses = part_field.clustering_losses(
        pair_samples=int(field_pair_samples),
        different_margin=float(field_different_margin),
        minimum_part_fraction=float(minimum_part_fraction),
    )
    for name, value in cluster_losses.items():
        losses[name] = value

    # 6) Smooth explicit omega(t),u(t).
    if rigid_twist_field is not None:
        dt = max(float(twist_dt), 1e-4)
        t = float(current_time)
        omega_prev, u_prev = rigid_twist_field(max(0.0, t - dt))
        omega_now, u_now = rigid_twist_field(t)
        omega_next, u_next = rigid_twist_field(min(1.0, t + dt))
        first = (omega_next - omega_prev).pow(2).mean()
        first = first + (u_next - u_prev).pow(2).mean()
        second = (omega_next - 2.0 * omega_now + omega_prev).pow(2).mean()
        second = second + (u_next - 2.0 * u_now + u_prev).pow(2).mean()
        losses["twist_smooth"] = first + second
        info["joint_omega_norm"] = float(omega_now.detach().norm().item())
        info["joint_translation_norm"] = float(u_now.detach().norm().item())

    info.update({
        "joint_points": int(total_velocity.shape[0]),
        "joint_q_mean": float(q.mean().item()),
        "joint_q_ge_05": int((q >= 0.5).sum().item()),
        "joint_q_p05": float(torch.quantile(q, 0.05).item()),
        "joint_q_p50": float(torch.quantile(q, 0.50).item()),
        "joint_q_p95": float(torch.quantile(q, 0.95).item()),
        "joint_p_mean": float(p_det.mean().item()),
        "joint_p_ge_05": int((p_det >= 0.5).sum().item()),
        "joint_p_confident_static": int(static_mask.sum().item()),
        "joint_p_confident_dynamic": int(dynamic_mask.sum().item()),
        "joint_p_uncertain": int((~static_mask & ~dynamic_mask).sum().item()),
        "joint_label_temperature_used": float(temp.item()),
        "joint_static_residual_mean": float(e_static.mean().item()),
        "joint_rigid_residual_mean": float(e_rigid.mean().item()),
        "joint_proposal_velocity_mean": float(proposal_speed.mean().item()),
        "joint_rigid_velocity_mean": float(torch.linalg.norm(rigid_velocity.detach(), dim=-1).mean().item()),
        "joint_total_velocity_mean": float(torch.linalg.norm(total_velocity.detach(), dim=-1).mean().item()),
        "joint_motion_evidence_mean": float(motion_evidence.mean().item()),
        "joint_twist_weight_mean": float(twist_weight.mean().item()),
        "joint_twist_fit_moving_cluster_points": int(moving_cluster_mask.sum().item()),
        "joint_speed_scale_p80": float(speed_scale.item()),
    })
    if part_field is not None:
        with torch.no_grad():
            assignment = part_field.assignments_fg()
            mass = assignment.mean(dim=0)
            for slot, value in enumerate(mass.tolist()):
                info[f"joint_part_mass_{slot}"] = float(value)
            info["joint_part_confidence_mean"] = float(part_field.confidence_fg().mean().item())
    return losses, info
