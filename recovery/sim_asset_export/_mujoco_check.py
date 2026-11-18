"""Standalone MuJoCo check (separate interpreter with `mujoco` + numpy; no RigEuler3DGS import).

    python _mujoco_check.py model.xml request.json out.json out.npz

Compiles model.xml with mujoco.MjModel.from_xml_path, evaluates mj_forward at every requested q, reads back body
poses, inertials, joint parameters and the world-space vertices of every mesh geom (q = 0), and optionally drops a
free sphere on the static link (contact smoke test; MjSpec adds the probe to a COPY of the model in memory).
"""
import json
import sys

import mujoco
import numpy as np


def main(xml, req_path, out_json, out_npz):
    req = json.load(open(req_path))
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    names = [m.body(i).name for i in range(m.nbody)]
    out = {"mujoco_version": mujoco.__version__, "nq": int(m.nq), "nv": int(m.nv), "njnt": int(m.njnt),
           "nbody": int(m.nbody), "bodies": names, "ngeom": int(m.ngeom)}
    j = m.joint(0) if m.njnt else None
    if j is not None:
        out["joint"] = {"name": j.name, "type": int(m.jnt_type[0]), "type_name": {2: "slide", 3: "hinge"}.get(
                            int(m.jnt_type[0]), str(int(m.jnt_type[0]))),
                        "axis": m.jnt_axis[0].tolist(), "pos": m.jnt_pos[0].tolist(), "range": m.jnt_range[0].tolist(),
                        "limited": bool(m.jnt_limited[0]), "damping": float(m.dof_damping[0]),
                        "frictionloss": float(m.dof_frictionloss[0]), "armature": float(m.dof_armature[0]),
                        "actfrcrange": m.jnt_actfrcrange[0].tolist(), "body": names[int(m.jnt_bodyid[0])]}
    out["inertial"] = {names[i]: {"mass": float(m.body_mass[i]), "ipos": m.body_ipos[i].tolist(),
                                  "iquat": m.body_iquat[i].tolist(), "inertia": m.body_inertia[i].tolist()}
                       for i in range(1, m.nbody)}
    poses = []
    for q in req["q"]:
        mujoco.mj_resetData(m, d)
        d.qpos[0] = q
        mujoco.mj_forward(m, d)
        poses.append({names[i]: {"xpos": d.xpos[i].tolist(), "xmat": d.xmat[i].reshape(3, 3).tolist()}
                      for i in range(1, m.nbody)})
    out["poses"] = poses
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    arrays = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        arrays[m.geom(g).name] = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    np.savez(out_npz, **arrays)
    c = req.get("contact")
    if c:
        spec = mujoco.MjSpec.from_file(xml)
        b = spec.worldbody.add_body(name="probe", pos=c["drop_world"])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[c["radius"], 0, 0], condim=1, priority=1,
                   friction=[0.0, 0.0, 0.0])   # frictionless (priority: its contact parameters win): geometry only
        r = c["radius"]
        mass = 500.0 * 4.0 / 3.0 * np.pi * r ** 3          # the model compiles with inertiafromgeom="false"
        b.explicitinertial = True
        b.mass = mass
        b.inertia = [0.4 * mass * r * r] * 3
        b.ipos = [0.0, 0.0, 0.0]                            # MjSpec does not default ipos to the body origin
        b.iquat = [1.0, 0.0, 0.0, 0.0]
        spec.body("moving_link").gravcomp = 1.0               # hold the passive joint still: geometry test only
        mc = spec.compile()
        dc = mujoco.MjData(mc)
        pid = mc.body("probe").id
        for _ in range(int(round(c["seconds"] / mc.opt.timestep))):
            mujoco.mj_step(mc, dc)
        out["contact"] = {"final_world": dc.xpos[pid].tolist(), "timestep": float(mc.opt.timestep),
                          "steps": int(round(c["seconds"] / mc.opt.timestep)), "ncon_final": int(dc.ncon),
                          "articulation_q_final": float(dc.qpos[0])}
    json.dump(out, open(out_json, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:5])
