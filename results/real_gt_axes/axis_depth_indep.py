"""Pipeline-independent support: points of the initial COLMAP cloud (the GT annotation cloud) whose static position
is in front of the observed end-frame depth (free space) in >= 1 camera -> they moved. Then the same axis test."""
import glob, json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/home/samuelemara/Joint3DGS/RigEuler3DGS/results"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigeuler_tables as T
from gt_common import GT, SCENE, unit, RES, xyz
rows = []
for gname, sc in SCENE.items():
    g = GT[gname]; name = f"final_{sc}_episode_000000"; src = T.REAL / name
    cams = {c["name"]: c for c in json.load(open(src / "metadata/cameras.json"))["cameras"]}
    B0 = Path(glob.glob(f"{RES}/ours_v6_two_cams/{name}/C/pipeline/*/bidirectional")[0])
    fr = json.load(open(sorted(B0.glob("*_joint"))[0] / "summary.json"))["original_frames_processing_order"]
    ks = np.linspace(0, len(fr) - 1, 12).astype(int)[1:]
    D = {(k, c): np.load(src / "depth_npy" / c / f"frame_{fr[k]:06d}.npy") for k in ks for c in cams}
    Xall = xyz(src / "colmap/sparse/0/points3D.ply")
    viol = np.zeros(len(Xall), int)
    for c in cams:
        viol += T.classify(cams[c], D[(ks[-1], c)], Xall, 0.02) == 2
    X = Xall[viol >= 1]
    cands = {}
    for arm in ("C", "majority", "unanimous"):
        B = Path(glob.glob(f"{RES}/ours_v6_two_cams/{name}/{arm}/pipeline/*/bidirectional")[0]); j = sorted(B.glob("*_joint"))[0]
        R = json.load(open(j / "revolute/geometry.json")); P = json.load(open(j / "prismatic/geometry.json"))
        cands[f"v6_{arm}_revolute"] = ("R", np.asarray(R["axis_point"]), unit(R["axis"]))
        cands[f"v6_{arm}_prismatic"] = ("P", None, unit(P["direction"]))
    cands["GT"] = ("R", g["axis_point"], unit(g["axis_direction_unit"])) if g["joint_type"] == "revolute" else ("P", None, unit(g["axis_direction_unit"]))
    for cn, (kind, p, a) in cands.items():
        grid = np.radians(np.arange(-90, 91, 1.0)) if kind == "R" else np.arange(-0.40, 0.4001, 0.005)
        sc_k, vi_k, bx = [], [], []
        for k in ks:
            best = (-1, 0, 0)
            for x in grid:
                Y = ((X - p) @ Rot.from_rotvec(x * a).as_matrix().T + p) if kind == "R" else X + x * a
                C = V = 0
                for c in cams:
                    cl = T.classify(cams[c], D[(k, c)], Y, 0.02); C += (cl == 1).sum(); V += (cl == 2).sum()
                n = len(Y) * len(cams)
                if C / n > best[0]: best = (C / n, x, V / n)
            sc_k.append(best[0]); bx.append(best[1]); vi_k.append(best[2])
        rows.append(dict(scene=sc, moved_points=len(X), of_cloud=len(Xall), candidate=cn, kind=kind,
                         mean_consistent=np.mean(sc_k), end_consistent=sc_k[-1], end_violated=vi_k[-1],
                         end_best=np.degrees(bx[-1]) if kind == "R" else 100 * bx[-1]))
df = pd.DataFrame(rows); df.to_csv(Path(__file__).resolve().parent / "axis_depth_indep.csv", index=False)
pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
