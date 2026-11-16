"""Which axis line explains the observed depth? Support of each run, rotated (translated) about a candidate line
by the per-frame best angle (offset); score = consistent points / (support x cameras), tau 2 cm, all cameras."""
import glob, json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/home/samuelemara/Joint3DGS/RigEuler3DGS/results"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigeuler_tables as T
from gt_common import GT, SCENE, unit, RES
REAL = T.REAL
rows = []
for gname, sc in SCENE.items():
    g = GT[gname]; name = f"final_{sc}_episode_000000"; src = REAL / name
    camsall = {c["name"]: c for c in json.load(open(src / "metadata/cameras.json"))["cameras"]}
    for ver, run in (("v6", "ours_v6_two_cams"),):
        for arm in ("C", "majority", "unanimous"):
            B = Path(glob.glob(f"{RES}/{run}/{name}/{arm}/pipeline/*/bidirectional")[0])
            joint = sorted(B.glob("*_joint"))[0]; rigid = sorted(B.glob("*_rigid"))[0]
            cd = sorted(B.glob("*/phase1_centers"))[0]
            X0 = np.load(cd / "canonical_xyz.npy")
            mem = json.load(open(rigid / "support_coherence/support.json"))["members"]; X = X0[mem]
            fr = json.load(open(joint / "summary.json"))["original_frames_processing_order"]
            ks = np.linspace(0, len(fr) - 1, 12).astype(int)[1:]
            D = {(k, c): np.load(src / "depth_npy" / c / f"frame_{fr[k]:06d}.npy") for k in ks for c in camsall}
            R = json.load(open(joint / "revolute/geometry.json")); P = json.load(open(joint / "prismatic/geometry.json"))
            cands = {"est_revolute": ("R", np.asarray(R["axis_point"]), unit(R["axis"])),
                     "est_prismatic": ("P", None, unit(P["direction"]))}
            if g["joint_type"] == "revolute":
                cands["gt_revolute"] = ("R", g["axis_point"], unit(g["axis_direction_unit"]))
            else:
                cands["gt_prismatic"] = ("P", None, unit(g["axis_direction_unit"]))
            for cn, (kind, p, a) in cands.items():
                grid = np.radians(np.arange(-90, 91, 1.0)) if kind == "R" else np.arange(-0.40, 0.4001, 0.005)
                sc_k, best_k, zero_k, vio_k = [], [], [], []
                for k in ks:
                    best = (-1, 0, 0)
                    for x in grid:
                        if kind == "R":
                            Rm = Rot.from_rotvec(x * a).as_matrix(); Y = (X - p) @ Rm.T + p
                        else:
                            Y = X + x * a
                        C = V = 0
                        for c in camsall:
                            cl = T.classify(camsall[c], D[(k, c)], Y, 0.02); C += (cl == 1).sum(); V += (cl == 2).sum()
                        s = C / (len(Y) * len(camsall)); viol = V / (len(Y) * len(camsall))
                        if abs(x) < 1e-9: zero_k.append(s)
                        if s > best[0]: best = (s, x, viol)
                    sc_k.append(best[0]); best_k.append(best[1]); vio_k.append(best[2])
                rows.append(dict(scene=sc, arm=arm, candidate=cn, support=len(X), mean_best_consistency=np.mean(sc_k),
                                 end_consistency=sc_k[-1], end_violated=vio_k[-1], mean_violated=np.mean(vio_k), static_consistency_end=zero_k[-1],
                                 end_best=(np.degrees(best_k[-1]) if kind == "R" else 100 * best_k[-1]),
                                 end_best_unit="deg" if kind == "R" else "cm"))
                pass
df = pd.DataFrame(rows); df.to_csv(Path(__file__).resolve().parent / "axis_depth_test.csv", index=False)
pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
