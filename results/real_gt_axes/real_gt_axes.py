import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_common import *
rows, frame = [], []
for gname, sc in SCENE.items():
    g = GT[gname]; name = f"final_{sc}_episode_000000"
    a_gt, p_gt = unit(g["axis_direction_unit"]), g["axis_point"]
    e1, e2 = g["endpoint_1"], g["endpoint_2"]
    cloud = cKDTree(xyz(REAL / name / "colmap/sparse/0/points3D.ply"))
    for ver, run in RUNS.items():
        for arm in ("C", "majority", "unanimous"):
            B = Path(glob.glob(f"{RES}/{run}/{name}/{arm}/pipeline/*/bidirectional")[0])
            joint = sorted(B.glob("*_joint"))[0]
            if ver == "v6" or arm == "C":       # frame check: endpoint Gaussians vs the GT cloud
                ep = sorted(B.parent.glob("static_bidi/*/point_cloud.ply"))[0]
                d, _ = cloud.query(xyz(ep))
                frame.append(dict(scene=sc, version=ver, arm=arm, endpoint=ep.parent.name,
                                  nn_median_mm=1e3 * np.median(d), nn_q90_mm=1e3 * np.quantile(d, .9)))
            R = json.load(open(joint / "revolute/geometry.json")); P = json.load(open(joint / "prismatic/geometry.json"))
            ra, rp, pd_ = unit(R["axis"]), np.asarray(R["axis_point"]), unit(P["direction"])
            rl = R.get("axis_radius_over_part_scale")
            row = dict(scene=sc, gt_name=gname, gt_type=g["joint_type"], version=ver, arm=arm,
                       type_decision="prismatic" if rl > 10 else "revolute", R_over_L=rl,
                       rev_axis_err_deg=line_angle_deg(ra, a_gt), pri_dir_err_deg=line_angle_deg(pd_, a_gt))
            if g["joint_type"] == "revolute":
                row.update(rev_line_dist_mm=1e3 * line_distance(rp, ra, p_gt, a_gt),
                           rev_dist_at_gt_mid_mm=1e3 * pt_line(p_gt, rp, ra),
                           rev_dist_at_gt_ends_max_mm=1e3 * max(pt_line(e1, rp, ra), pt_line(e2, rp, ra)))
            row["selected_axis_err_deg"] = row["pri_dir_err_deg"] if row["type_decision"] == "prismatic" else row["rev_axis_err_deg"]
            row["type_correct"] = row["type_decision"] == g["joint_type"]
            rows.append(row)
df = pd.DataFrame(rows); fr = pd.DataFrame(frame)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(fr.round(2).to_string(index=False)); print()
print(df.round(2).to_string(index=False))
df.to_csv(Path(__file__).resolve().parent / "real_gt_axes.csv", index=False)
