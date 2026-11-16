"""Gates 4 and 5: Phase 4B discovery + 4C acceptance on synthetic scenes with MISSING observed surfaces.

  * prismatic drawer, BACKWARD processing order, drawer floor removed from the model (the floor is visible
    only when the drawer is out): the late-visible surface is back-transported into ONE canonical moving
    part, accepted MOVING, and lies where the true floor is at the anchor; its first-visible frame is
    original-frame provenance (a frame with the drawer out), not a birth time
  * revolute door with its back face removed (12 mm behind the front face, opposite normal): proposals
    land on the back layer, never collapse onto the front face, are not duplicates of it
  * a planted wrong patch floating in front of the object is never accepted
  * N_final = N_original + N_accepted, original IDs untouched, objective improves
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import candidate_acceptance as C  # noqa: E402
from recovery import canonical_fusion as FU  # noqa: E402
from recovery import material_completion as MC  # noqa: E402
from recovery import residual_discovery as RD  # noqa: E402
from recovery.gaussians import MOVING, PROV_PHASE4_BIRTH, STATIC  # noqa: E402
from recovery.phase4 import run_phase4  # noqa: E402
from recovery.rgbd_loss import LossConfig  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

FCFG = FU.FusionConfig(voxel=0.015, min_obs=3, min_views=2)
POL = MC.PolicyConfig(max_component=128, sensitivity=False)


def _run(jtype, direction, drop, seed=4):
    s = make_scene(jtype, direction, drop=drop, withhold_frac=0.0, seed=seed, W=128, H=96)
    sc = MC.Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
    n0 = len(s.model)
    r = run_phase4(sc, s.core, POL, RD.DiscoveryConfig(edge_px=1), FCFG, C.AcceptConfig(), log=lambda *a: None)
    assert not r["aborted"]
    g = sc.g
    assert len(g) == n0 + r["n_accepted"]
    assert np.array_equal(g.lineage["original_id"][:n0], np.arange(n0))
    assert (g.lineage["provenance"][n0:] == PROV_PHASE4_BIRTH).all() and (g.lineage["original_id"][n0:] == -1).all()
    assert len(np.unique(g.lineage["gaussian_uid"])) == len(g)
    before = np.mean([x["total"] for x in r["audit_before"]])
    after = np.mean([x["total"] for x in r["audit_after"]])
    return s, sc, r, n0, before, after


def _gt_points(s, name):
    m = s.surface_full == name
    return s.full._xyz[torch.as_tensor(m, device=s.full._xyz.device)].cpu().numpy()


def test_drawer_floor_backward_is_back_transported():
    s, sc, r, n0, before, after = _run("prismatic", "backward", drop=("drawer_floor",))
    st = [d["status"] for d in r["decisions"]]
    new = np.arange(n0, len(sc.g))
    mat = sc.g.lineage["material"][new]
    print(f"  drawer: decisions {dict(zip(*np.unique(st, return_counts=True)))}, added {len(new)} "
          f"(moving {int((mat == MOVING).sum())}), loss {before:.4f} -> {after:.4f}")
    assert len(new) > 0 and (mat == MOVING).mean() > 0.9, "the floor must come back as MOVING geometry"
    X = sc.g._xyz[torch.as_tensor(new[mat == MOVING], device=sc.device)].detach().cpu().numpy()
    from scipy.spatial import cKDTree
    d_floor, _ = cKDTree(_gt_points(s, "drawer_floor")).query(X)
    off = np.abs(X[:, 1] - _gt_points(s, "drawer_floor")[:, 1].mean())
    assert np.median(off) < FCFG.voxel, f"accepted cells {np.median(off):.4f} m off the true floor plane"
    for d in r["decisions"]:
        if d["status"] == C.ACCEPTED_MOVING:
            f = d["first_visible_frame_provenance"]
            # backward: anchor = last original frame (drawer closed); the floor is first seen with the drawer out
            assert f != s.joint.anchor_frame and abs(s.joint.q_of(f)) > 0
    assert after < before


def test_door_back_face_no_front_back_collapse():
    s, sc, r, n0, before, after = _run("revolute", "forward", drop=("door_back",))
    new = np.arange(n0, len(sc.g))
    mat = sc.g.lineage["material"][new]
    st = [d["status"] for d in r["decisions"]]
    print(f"  door: decisions {dict(zip(*np.unique(st, return_counts=True)))}, added {len(new)} "
          f"(moving {int((mat == MOVING).sum())}), loss {before:.4f} -> {after:.4f}")
    assert len(new) > 0 and (mat == MOVING).mean() > 0.9
    X = sc.g._xyz[torch.as_tensor(new[mat == MOVING], device=sc.device)].detach().cpu().numpy()
    from scipy.spatial import cKDTree
    d_back, _ = cKDTree(_gt_points(s, "door_back")).query(X)
    d_front, _ = cKDTree(_gt_points(s, "door_front")).query(X)
    closer_back = (np.abs(X[:, 2] - _gt_points(s, "door_back")[:, 2].mean()) <
                   np.abs(X[:, 2] - _gt_points(s, "door_front")[:, 2].mean())).mean()
    assert closer_back > 0.8, f"only {closer_back:.2f} of the new cells sit on the back layer"
    assert after < before


def test_wrong_front_occluding_patch_is_rejected():
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=5)
    sc = MC.Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
    moving = torch.as_tensor(s.gt_moving_model, device=sc.device)
    # a 5 x 5 static sheet 6 cm in front of the closed door: in front of measured surfaces in every frame
    gx, gy = np.meshgrid(np.linspace(-0.08, 0.08, 5), np.linspace(-0.08, 0.08, 5))
    cells = [{"key": (int(i), 0, 0, 5), "mu": np.array([x, y, 1.94]), "normal": np.array([0, 0, -1.0]),
              "rgb": np.array([0.5, 0.5, 0.5]), "thickness": 0.001, "first_frame": 0, "src": np.array([i])}
             for i, (x, y) in enumerate(zip(gx.ravel(), gy.ravel()))]
    n0 = len(sc.g)
    _, _, dec = C.run_phase4c(sc, moving, [{STATIC: cells, MOVING: None, "overlap": 0.0}], C.AcceptConfig(),
                              voxel=0.03, log=lambda *a: None)
    assert dec[0]["status"] not in (C.ACCEPTED_STATIC, C.ACCEPTED_MOVING), dec[0]
    assert len(sc.g) == n0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
