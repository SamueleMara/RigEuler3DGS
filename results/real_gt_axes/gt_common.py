import glob, json, sys, numpy as np, pandas as pd
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/samuelemara/Joint3DGS/RigEuler3DGS")
from observation_diagnostics.gt_evaluation import line_angle_deg, line_distance
RES = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26")
REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
GTF = REAL / "gt_joint_axes_3_real_scenes/gt_joint_axes_3_real_scenes.txt"

def parse(p):
    out, cur = {}, None
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        if ln.startswith("["): cur = ln[1:-1]; out[cur] = {}; continue
        k, v = [x.strip() for x in ln.split("=", 1)]
        try: v = np.array([float(x) for x in v.split(",")])
        except ValueError: pass
        out[cur][k] = v
    return out
GT = parse(GTF)
SCENE = {"red_fridge": "red_fridge", "wood_box": "wooden_box", "blue_box": "light_blue_box"}
RUNS = {"v6": "ours_v6_two_cams", "v5": "ours_v5_two_cams"}

def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def pt_line(x, p, a):
    a = unit(a); d = np.asarray(x) - p; return float(np.linalg.norm(d - (d @ a) * a))
def xyz(p):
    v = PlyData.read(p)["vertex"]; return np.stack([v["x"], v["y"], v["z"]], 1).astype(float)

