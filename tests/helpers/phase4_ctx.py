"""In-memory Phase-4 contexts for the procedural gate tests (same stage functions as the CLI)."""
import hashlib

import numpy as np
import torch

from recovery.material_completion import Phase4Scene
from recovery.phase4_stages import Context
from recovery.rgbd_loss import LossConfig
from recovery.view_cache import PreparedViews, ViewKey

QUIET = lambda *a, **k: None


def prepared(views):
    keys = [ViewKey(v.cam.name, int(v.frame), v.cam.W, v.cam.H) for v in views]
    return PreparedViews.from_memory(keys, [v.get() for v in views], log=QUIET)


def ctx_from_scene(s, model=None, core=None, stride=1):
    """stride > 1: the 4B / 4C recovery context over every stride-th frame (same model, same joint)."""
    m = model if model is not None else s.model
    views = s.views
    if stride > 1:
        keep = set(sorted({int(v.frame) for v in s.views})[::stride])
        views = [v for v in s.views if int(v.frame) in keep]
    sc = Phase4Scene(m, s.joint, views, LossConfig(), sh_degree=0)
    pv = prepared(views)
    proof = {"procedural": True, "upstream_sha256": {"procedural": True}, "joint": s.joint.to_dict(),
             "views_combined_sha256": hashlib.sha256("".join(pv.checksums).encode()).hexdigest(), "views": len(pv)}
    return Context(scene=sc, pv=pv, core=np.asarray(s.core if core is None else core, np.int64),
                   setup={"procedural": True}, input_proof=proof, recovery_stride=stride)


def surface_layers(s, cam, frame, names_of=None):
    """GT per-surface visible weights [S,H,W] and depths [S,H,W] of the FULL ground-truth scene at (cam, frame)."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from recovery.articulated_scene import JointTransport, canonical_tensors, posed
    tr = JointTransport(s.joint)
    t = canonical_tensors(s.full)
    gtm = torch.as_tensor(s.full.lineage["material"].astype(bool), device="cuda")
    mu, q = posed(t["xyz"], t["rot"], gtm, tr.at(frame))
    zc = (mu @ cam.V.cuda()[:3, :3] + cam.V.cuda()[3, :3])[:, 2]
    R = GaussianRasterizer(GaussianRasterizationSettings(
        image_height=cam.H, image_width=cam.W, tanfovx=cam.tanfovx, tanfovy=cam.tanfovy, bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0, viewmatrix=cam.V.cuda(), projmatrix=cam.P.cuda(), sh_degree=0, campos=cam.campos.cuda(),
        prefiltered=False, debug=False, antialiasing=False))
    names = sorted(set(s.surface_full)) if names_of is None else names_of
    Wt, Dd = [], []
    for nm in names:
        grp = torch.as_tensor(s.surface_full == nm, device="cuda").float()
        c, *_ = R.forward_rgbd(mu, torch.zeros_like(mu), t["opacity"], colors_precomp=torch.stack([grp, grp * zc, grp * 0], 1),
                               scales=t["scale"], rotations=q)
        Wt.append(c[0])
        Dd.append(c[1] / c[0].clamp_min(1e-9))
    return names, torch.stack(Wt), torch.stack(Dd)
