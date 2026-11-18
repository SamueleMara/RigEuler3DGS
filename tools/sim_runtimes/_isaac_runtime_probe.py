"""Runs INSIDE Isaac Sim's python.sh (never imported by RigEuler3DGS):

    ~/isaac-sim/python.sh _isaac_runtime_probe.py request.json out.json

Opens model.usd as a reference under /World/asset in a headless Isaac Sim, instantiates the PhysX articulation
(SingleArticulation at the referenced articulation root), reads DOF count / names / limits, sets every requested q with
gravity off and reads the PhysX rigid-body poses of both links after one step. The caller compares against the
analytic SE(3).
"""
import json
import sys

req = json.load(open(sys.argv[1]))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
try:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
    world.get_physics_context().set_gravity(0.0)
    add_reference_to_stage(usd_path=req["usd"], prim_path="/World/asset")
    art = SingleArticulation(prim_path="/World/asset", name="asset")
    world.scene.add(art)
    world.reset()
    out = {"num_dof": int(art.num_dof), "dof_names": list(art.dof_names)}
    try:
        props = art.dof_properties
        out["dof_lower"] = float(props["lower"][0])
        out["dof_upper"] = float(props["upper"][0])
    except Exception as e:                                   # API detail differs across releases: report, not fail
        out["dof_limits_error"] = repr(e)
    links = {n: SingleRigidPrim(prim_path=f"/World/asset/{n}") for n in ("static_link", "moving_link")}
    poses = []
    for q in req["q"]:
        art.set_joint_positions(np.array([q], dtype=np.float32))
        art.set_joint_velocities(np.zeros(1, dtype=np.float32))
        world.step(render=False)
        p = {}
        for n, rp in links.items():
            pos, quat = rp.get_world_pose()
            p[n] = {"pos": np.asarray(pos, float).tolist(), "quat_wxyz": np.asarray(quat, float).tolist()}
        p["q_read"] = float(np.asarray(art.get_joint_positions())[0])
        poses.append(p)
    out["poses"] = poses
    try:
        import isaacsim
        out["isaac_version"] = getattr(isaacsim, "__version__", None)
    except Exception:
        pass
    json.dump(out, open(sys.argv[2], "w"))
finally:
    app.close()
