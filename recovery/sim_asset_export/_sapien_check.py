"""Standalone ManiSkill / SAPIEN + PArtTwin1D consumer check (separate interpreter with sapien, numpy and,
optionally, parttwin1d; no RigEuler3DGS import; nothing is written into either package).

    python _sapien_check.py model.urdf request.json out.json out.npz

* loads model.urdf with SAPIEN's URDF loader on a CPU PhysX scene (fix_root_link, root pose = T_world_root),
  reads back DOF / limits / joint parameters / link masses and inertias, and the live link poses at every q
* optional contact smoke test: a free sphere dropped on the static link
* if request["object_yaml"]: PArtTwin1D's load_config + load_gaussian_ply + load_scene + LinkAdapter, driven by the
  SAPIEN link poses (T_W_P = T_W_L @ T_L_P) -> transported Gaussian means / quaternions saved for comparison
"""
import json
import sys

import numpy as np


def main(urdf, req_path, out_json, out_npz):
    import sapien
    import sapien.physx as px
    req = json.load(open(req_path))
    scene = sapien.Scene([px.PhysxCpuSystem()])
    scene.set_timestep(1.0 / 500.0)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load(urdf)
    T = np.asarray(req["T_world_root"])
    art.set_root_pose(sapien.Pose(T))
    out = {"sapien_version": sapien.__version__, "dof": int(art.dof), "links": [l.name for l in art.links],
           "joints": [j.name for j in art.active_joints]}
    j = art.active_joints[0]
    out["joint"] = {"name": j.name, "type": str(j.type), "limits": art.get_qlimits().tolist(),
                    "damping": float(j.damping), "friction": float(j.friction), "armature": np.asarray(j.armature).tolist(),
                    "max_velocity": float(j.max_velocity), "force_limit": float(j.force_limit),
                    "pose_in_parent": j.pose_in_parent.to_transformation_matrix().tolist(),
                    "pose_in_child": j.pose_in_child.to_transformation_matrix().tolist()}
    out["inertial"] = {l.name: {"mass": float(l.mass), "inertia": np.asarray(l.inertia).tolist(),
                                "cmass_local_pose": l.cmass_local_pose.to_transformation_matrix().tolist(),
                                "collision_shapes": len(l.collision_shapes)} for l in art.links}
    poses = []
    for q in req["q"]:
        art.set_qpos(np.array([q], np.float32))
        poses.append({l.name: l.entity_pose.to_transformation_matrix().tolist() for l in art.links})
    out["poses"] = poses
    arrays = {}
    oy = req.get("object_yaml")
    if oy:
        try:
            import torch
            from parttwin1d.assets.gaussian_ply import load_gaussian_ply
            from parttwin1d.config import load_config
            from parttwin1d.rendering.gaussians import load_scene
            from parttwin1d.rendering.link_adapter import LinkAdapter
            cfg = load_config(oy)
            gs = load_scene(cfg.links, device="cpu")
            ad = LinkAdapter(gs)
            names = [b.link for b in cfg.links]
            out["parttwin"] = {"object_name": cfg.object_name, "links": names, "counts":
                               [load_gaussian_ply(b.ply).count for b in cfg.links],
                               "T_world_root_equal": bool(np.allclose(cfg.T_world_root, T, atol=0, rtol=0))}
            lk = {l.name: l for l in art.links}
            for k, q in enumerate(req["adapter_q"]):
                art.set_qpos(np.array([q], np.float32))
                Twl = torch.as_tensor(np.stack([lk[n].entity_pose.to_transformation_matrix() for n in names]),
                                      dtype=torch.float32)
                mw, qw = ad.update(Twl)
                arrays[f"means_{k}"] = mw.numpy().astype(np.float64)
                arrays[f"quats_{k}"] = qw.numpy().astype(np.float64)
            arrays["link_ids"] = gs.link_ids.numpy()
        except ImportError as e:
            out["parttwin"] = {"status": "NOT_RUN", "reason": f"parttwin1d not importable: {e}"}
    c = req.get("contact")
    if c:
        art.set_qpos(np.zeros(1, np.float32))
        for l in art.links:                                   # hold the passive joint still: geometry test only
            l.disable_gravity = True
        b = scene.create_actor_builder()
        b.add_sphere_collision(radius=c["radius"], material=px.PhysxMaterial(0.0, 0.0, 0.0))   # frictionless
        probe = b.build(name="probe")
        probe.set_pose(sapien.Pose(c["drop_world"]))
        n = int(round(c["seconds"] * 500))
        for _ in range(n):
            scene.step()
        out["contact"] = {"final_world": probe.pose.p.tolist(), "timestep": 1.0 / 500.0, "steps": n,
                          "articulation_q_final": float(art.get_qpos()[0])}
    np.savez(out_npz, **arrays)
    json.dump(out, open(out_json, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:5])
