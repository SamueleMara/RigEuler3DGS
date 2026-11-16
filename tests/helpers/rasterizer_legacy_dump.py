"""Legacy entry points only (rasterize_gaussians fwd+bwd, features, positions), saved for cross-build compare."""
import sys, math, torch
out_path = sys.argv[1]
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
import diff_gaussian_rasterization as D
print("rasterizer:", D.__file__, "has rgbd:", hasattr(D._C, "rasterize_gaussians_rgbd"))
import importlib.util
spec = importlib.util.spec_from_file_location("T", str(HERE / "test_rgbd_cuda_forward_backward.py"))
src = open(spec.origin).read().replace("import recovery  # noqa: E402,F401  (pins this checkout's rasterizer)", "")
src = src.replace("from diff_gaussian_rasterization import (GaussianRasterizationSettings, GaussianRasterizer,  # noqa: E402\n                                         _C, metric_depth_from)", "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer")
src = src.replace("from recovery import raster_oracle as O  # noqa: E402", "")
T = type(sys)("T"); T.__file__ = spec.origin; exec(compile(src, "T", "exec"), T.__dict__)
res = {}
for aa in (False, True):
  for ci, cam in enumerate(T.cameras()):
    a = T.activated(T.scene(seed=7, n_front=40, n_back=40))
    s = D.GaussianRasterizationSettings(image_height=T.H, image_width=T.W, tanfovx=math.tan(T.FOV/2), tanfovy=math.tan(T.FOV/2),
        bg=T.BG.float().cuda(), scale_modifier=1.0, viewmatrix=cam["V"].float().cuda(), projmatrix=cam["P"].float().cuda(),
        sh_degree=3, campos=cam["campos"].float().cuda(), prefiltered=False, debug=False, antialiasing=aa)
    R = D.GaussianRasterizer(s)
    L = {k: v.float().cuda().requires_grad_(True) for k, v in a.items()}
    m2 = torch.zeros(L["means"].shape[0], 3, device="cuda", requires_grad=True)
    color, radii, invd = R(L["means"], m2, L["opacities"], shs=L["sh"], scales=L["scales"], rotations=L["rotations"])
    g = torch.Generator().manual_seed(11)
    (torch.randn(color.shape, generator=g).cuda() * color).sum().add((torch.randn(invd.shape, generator=g).cuda() * invd).sum()).backward()
    key = f"aa{int(aa)}_cam{ci}"
    res[key] = dict(color=color.detach().cpu(), radii=radii.cpu(), invd=invd.detach().cpu(), m2=m2.grad.cpu(),
                    **{f"g_{k}": v.grad.cpu() for k, v in L.items()})
    # feature path (8 channels) and position path
    s8 = s._replace(bg=torch.zeros(8, device="cuda"))
    feats = torch.linspace(0, 1, L["means"].shape[0] * 8, device="cuda").reshape(-1, 8)
    fo, _, _ = D.GaussianRasterizer(s8).render_features(L["means"].detach(), torch.zeros_like(m2), feats, L["opacities"].detach(), scales=L["scales"].detach(), rotations=L["rotations"].detach())
    res[key]["features"] = fo.detach().cpu()
    s4 = s._replace(bg=torch.zeros(4, device="cuda"))
    po, _, _ = D.GaussianRasterizer(s4).render_positions(L["means"].detach(), torch.zeros_like(m2), L["means"].detach() + 0.1, L["opacities"].detach(), scales=L["scales"].detach(), rotations=L["rotations"].detach())
    res[key]["positions"] = po.detach().cpu()
torch.save(res, out_path)
print("saved", len(res))
