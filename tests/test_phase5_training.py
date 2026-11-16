"""Gate 6: Phase 5 Gaussian-only training with an immutable joint (spec 7, 10 Gate 6).

  * gaussian_rgbd on synthetic R (forward) and P (backward) scenes whose appearance and geometry were
    perturbed: the full-sweep objective decreases; the joint fingerprint (files + transport buffers) is
    identical before / after; joint_unchanged_proof.json says joint_optimized = false; lineage stays aligned
  * sh_only: xyz / rotation / scale / opacity bitwise unchanged after many updates; depth and mask are
    monitors (their per-view values are identical before and after); colour improves
  * loss gating: depth-only and mask-only objectives give zero colour gradient; RGB-off totals equal the
    depth + mask terms; each switch removes exactly its term
  * topology: densification in CANONICAL space (children of a moving Gaussian sit around its canonical mean,
    not its transported one), hard material and parent uid inherited, Adam state row-aligned
  * the sampler covers every (camera, frame) once per epoch
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import topology as TP  # noqa: E402
from recovery.articulated_scene import JointTransport, render_articulated  # noqa: E402
from recovery.contracts import Joint  # noqa: E402
from recovery.gaussians import MOVING, PROV_DENSIFY_SPLIT  # noqa: E402
from recovery.rgbd_loss import LossConfig, view_loss  # noqa: E402
from recovery.training import Phase5Config, ViewSampler, build_optimizer, make_trainable, train_phase5  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

SMALL = dict(iters_A=40, iters_B=300, iters_C=60, densify_from=60, densify_every=60, densify_until=240,
             sh_interval=100, val_every=0)


def _scene_with_files(jtype, direction, tmp):
    s = make_scene(jtype, direction, withhold_frac=0.0, seed=7)
    # write the joint to real files so the proof hashes bytes, as on real runs
    g = Path(tmp) / "geometry.json"
    t = Path(tmp) / "trajectory.csv"
    g.write_text(json.dumps({"axis": s.joint.axis, "point": s.joint.point}))
    t.write_text("processing_index,original_frame,q\n" + "".join(
        f"{k},{f},{q}\n" for k, (f, q) in enumerate(zip(s.joint.frames, s.joint.q))))
    j = Joint(**{**s.joint.__dict__, "geometry_file": str(g), "trajectory_file": str(t)})
    m = s.model
    m.lineage["material"] = s.gt_moving_model.astype(np.int8)
    gen = torch.Generator(device="cuda").manual_seed(0)
    m._features_dc = m._features_dc + 0.4 * torch.randn(m._features_dc.shape, generator=gen, device="cuda")
    m._xyz = m._xyz + 0.004 * torch.randn(m._xyz.shape, generator=gen, device="cuda")
    m.max_sh_degree = 3
    return s, j


def test_gaussian_rgbd_trains_and_never_touches_the_joint():
    for jt, d in (("revolute", "forward"), ("prismatic", "backward")):
        with tempfile.TemporaryDirectory() as tmp:
            s, j = _scene_with_files(jt, d, tmp)
            geo_bytes = Path(j.geometry_file).read_bytes()
            out = Path(tmp) / "phase5"
            r = train_phase5(s.model, j, s.views, s.cameras, Phase5Config(mode="gaussian_rgbd", **SMALL), out,
                             log=lambda *a: None)
            L0 = np.mean([x["total"] for x in r["initial"]])
            L1 = np.mean([x["total"] for x in r["final"]])
            proof = json.loads((out / "final" / "joint_unchanged_proof.json").read_text())
            print(f"  {jt}/{d}: loss {L0:.4f} -> {L1:.4f}, Gaussians {len(s.gt_moving_model)} -> {len(s.model)}, "
                  f"topology {[{k: v for k, v in e.items() if k in ('cloned', 'split', 'pruned')} for e in r['topology']][:3]}")
            assert L1 < 0.8 * L0
            assert proof["joint_optimized"] is False and proof["identical"]
            assert Path(j.geometry_file).read_bytes() == geo_bytes
            s.model.check()
            assert (out / "final" / "point_cloud.ply").is_file() and (out / "frame_camera_metrics.csv").is_file()


def test_sh_only_keeps_geometry_and_treats_depth_mask_as_monitors():
    with tempfile.TemporaryDirectory() as tmp:
        s, j = _scene_with_files("revolute", "forward", tmp)
        r = train_phase5(s.model, j, s.views, s.cameras, Phase5Config(mode="sh_only", iters_C=300), Path(tmp) / "p5",
                         log=lambda *a: None)
        assert all(r["sh_only_frozen_unchanged"].values())
        for a, b in zip(r["initial"], r["final"]):
            assert a["depth_l1"] == b["depth_l1"] and a["mask_l1"] == b["mask_l1"], "depth / mask must not change"
        assert np.mean([x["rgb_l1"] for x in r["final"]]) < np.mean([x["rgb_l1"] for x in r["initial"]])


def test_loss_switches_gate_their_terms():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=8)
    m = s.model
    m.lineage["material"] = s.gt_moving_model.astype(np.int8)
    gen = torch.Generator(device="cuda").manual_seed(1)       # non-zero residuals (L1 has zero slope at 0)
    m._xyz = m._xyz + 0.005 * torch.randn(m._xyz.shape, generator=gen, device="cuda")
    m._opacity = m._opacity - 1.0
    make_trainable(m, "gaussian_rgbd")
    tr = JointTransport(s.joint)
    vr = s.views[5]
    v = vr.get().to("cuda")
    mv = torch.as_tensor(s.gt_moving_model, device="cuda")

    def grads(**flags):
        for p in (m._xyz, m._features_dc, m._opacity):
            p.grad = None
        o = render_articulated(vr.cam, tr.at(vr.frame), m, mv, torch.ones(3, device="cuda"), "local_SH", 0)
        t = view_loss(o, v, LossConfig(**flags))
        t["total"].backward()
        return {k: getattr(m, k).grad.clone() if getattr(m, k).grad is not None else torch.zeros_like(getattr(m, k))
                for k in ("_xyz", "_features_dc", "_opacity")}, t
    g_d, _ = grads(use_rgb=False, use_mask=False)
    g_m, _ = grads(use_rgb=False, use_depth=False)
    assert float(g_d["_features_dc"].abs().max()) == 0 and float(g_m["_features_dc"].abs().max()) == 0
    assert float(g_d["_xyz"].abs().max()) > 0 and float(g_m["_opacity"].abs().max()) > 0
    g_dm, _ = grads(use_rgb=False)
    for k in g_dm:
        assert torch.allclose(g_dm[k], g_d[k] + g_m[k], atol=1e-6), k
    g_all, t_all = grads()
    g_r, t_r = grads(use_depth=False, use_mask=False)
    for k in g_all:
        assert torch.allclose(g_all[k], g_r[k] + g_dm[k], atol=1e-5), k


def test_topology_canonical_split_lineage_and_optimizer_alignment():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=9)
    m = s.model
    m.lineage["material"] = s.gt_moving_model.astype(np.int8)
    train = make_trainable(m, "gaussian_rgbd")
    opt = build_optimizer(m, train, Phase5Config(), 1.0)
    # one step so Adam has state
    o = render_articulated(s.cameras[0], JointTransport(s.joint).at(s.joint.frames[-1]), m,
                           torch.as_tensor(s.gt_moving_model, device="cuda"), torch.ones(3, device="cuda"), "local_SH", 0)
    o["color"].sum().backward()
    opt.step()
    n0 = len(m)
    mov = np.nonzero(s.gt_moving_model)[0][:5]
    gn = torch.zeros(n0, device="cuda")
    gn[torch.as_tensor(mov, device="cuda")] = 1.0
    parent_uid = m.lineage["gaussian_uid"][mov].copy()
    parent_xyz = m._xyz.detach()[torch.as_tensor(mov, device="cuda")].cpu().numpy()
    rec = TP.densify(m, opt, gn, 0.5, extent=1.0, percent_dense=0.0, N=2)    # percent_dense 0: all selected split
    assert rec["split"] == 5 and len(m) == n0 - 5 + 10
    kids = np.nonzero(m.lineage["provenance"] == PROV_DENSIFY_SPLIT)[0]
    assert (m.lineage["material"][kids] == MOVING).all()
    assert set(m.lineage["parent_uid"][kids]) == set(parent_uid)
    for k in kids:
        p = parent_xyz[list(parent_uid).index(m.lineage["parent_uid"][k])]
        assert np.linalg.norm(m._xyz.detach()[k].cpu().numpy() - p) < 0.1, "child far from its CANONICAL parent"
    for gr in opt.param_groups:
        p = gr["params"][0]
        assert p.shape[0] == len(m) and opt.state[p]["exp_avg"].shape == p.shape
    m.check()


def test_sampler_covers_every_view_each_epoch():
    smp = ViewSampler(37, seed=3)
    for _ in range(3):
        assert sorted(smp.next() for _ in range(37)) == list(range(37))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
