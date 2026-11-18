"""Standalone OpenUSD writer / reader for the sim asset. Runs in a SEPARATE interpreter that provides `pxr`
(usd-core, or Isaac Sim's kit python); needs no numpy and no RigEuler3DGS import.

    python _usd_writer.py write payload.json out.usd readback.json
    python _usd_writer.py read  in.usd readback.json

All unit conversions are done by the caller (recovery/sim_asset_export/usd_export.py); this file only writes the
given values and reads back what the stage actually contains.
"""
import json
import sys


def _mat(T):
    """Column-vector 4x4 (list of rows) -> Gf.Matrix4d (USD row-vector convention = transpose)."""
    from pxr import Gf
    return Gf.Matrix4d(*[T[c][r] for r in range(4) for c in range(4)])


def _colvec(M):
    """Gf.Matrix4d (row-vector) -> column-vector list of rows."""
    return [[M[c][r] for c in range(4)] for r in range(4)]


def write(payload, out_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    P = payload
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root_path = "/" + P["name"]
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTransformOp().Set(_mat(P["T_world_root"]))
    rp = root.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    rp.SetDocumentation(P["doc"])
    for k, v in P["root_metadata"].items():
        typ = {"asset": Sdf.ValueTypeNames.Asset, "string": Sdf.ValueTypeNames.String,
               "bool": Sdf.ValueTypeNames.Bool, "double2": Sdf.ValueTypeNames.Double2}[v["type"]]
        rp.CreateAttribute(f"rigeuler:{k}", typ, custom=True).Set(
            Sdf.AssetPath(v["value"]) if v["type"] == "asset" else (Gf.Vec2d(*v["value"]) if v["type"] == "double2"
                                                                     else v["value"]))
    scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    g = P["gravity"]
    mag = sum(x * x for x in g) ** 0.5
    scene.CreateGravityDirectionAttr(Gf.Vec3f(*[x / mag for x in g]))
    scene.CreateGravityMagnitudeAttr(mag)
    mat = None
    if P.get("contact"):
        c = P["contact"]
        mat = UsdShade.Material.Define(stage, root_path + "/PhysicsMaterials/contact")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(c["static_friction"])
        api.CreateDynamicFrictionAttr(c["dynamic_friction"])
        api.CreateRestitutionAttr(c["restitution"])
    for L in P["links"]:
        lp = root_path + "/" + L["name"]
        x = UsdGeom.Xform.Define(stage, lp)
        x.AddTransformOp().Set(_mat(L["T_root_link"]))
        prim = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(L["mass"])
        m.CreateCenterOfMassAttr(Gf.Vec3f(*L["com"]))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*L["diagonal_inertia"]))
        m.CreatePrincipalAxesAttr(Gf.Quatf(*L["principal_axes_wxyz"]))
        prim.CreateAttribute("rigeuler:visualPly", Sdf.ValueTypeNames.Asset, custom=True).Set(Sdf.AssetPath(L["ply"]))
        prim.CreateAttribute("rigeuler:T_link_ply", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(
            Vt.DoubleArray([v for r in L["T_link_ply"] for v in r]))
        prim.CreateAttribute("rigeuler:T_link_ply_convention", Sdf.ValueTypeNames.String, custom=True).Set(
            "row-major 4x4, COLUMN-vector convention: x_link = T x_ply (not Gf row-vector)")
        for k, part in enumerate(L["collision"]):
            mp = f"{lp}/collision_{k:03d}"
            mesh = UsdGeom.Mesh.Define(stage, mp)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in part["vertices"]]))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(part["faces"])))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in part["faces"] for i in f]))
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            mesh.GetPrim().CreateAttribute("rigeuler:sourceObj", Sdf.ValueTypeNames.Asset, custom=True).Set(
                Sdf.AssetPath(part["obj"]))
            if mat is not None:
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    J = P["joint"]
    jp = root_path + "/joints/" + J["name"]
    if J["type"] == "revolute":
        j = UsdPhysics.RevoluteJoint.Define(stage, jp)
    else:
        j = UsdPhysics.PrismaticJoint.Define(stage, jp)
    j.CreateBody0Rel().SetTargets([Sdf.Path(root_path + "/static_link")])
    j.CreateBody1Rel().SetTargets([Sdf.Path(root_path + "/moving_link")])
    j.CreateLocalPos0Attr(Gf.Vec3f(*J["localPos0"]))
    j.CreateLocalRot0Attr(Gf.Quatf(*J["localRot0_wxyz"]))
    j.CreateLocalPos1Attr(Gf.Vec3f(*J["localPos1"]))
    j.CreateLocalRot1Attr(Gf.Quatf(*J["localRot1_wxyz"]))
    j.CreateAxisAttr("Z")
    j.CreateCollisionEnabledAttr(False)                  # no static <-> moving contact (same as MJCF exclude, PhysX)
    j.CreateLowerLimitAttr(J["lower_usd"])
    j.CreateUpperLimitAttr(J["upper_usd"])
    jprim = j.GetPrim()
    d = UsdPhysics.DriveAPI.Apply(jprim, J["drive_token"])
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(J["damping_usd"])
    d.CreateMaxForceAttr(J["effort"])
    d.CreateTargetVelocityAttr(0.0)
    jprim.AddAppliedSchema("PhysxJointAPI")              # raw: valid USD with or without the PhysxSchema plugin
    jprim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(J["max_velocity_usd"])
    if J.get("armature") is not None:
        jprim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(J["armature"])
    fx = UsdPhysics.FixedJoint.Define(stage, root_path + "/joints/root_fixed")
    fx.CreateBody1Rel().SetTargets([Sdf.Path(root_path + "/static_link")])
    Tw = P["T_world_root"]
    fx.CreateLocalPos0Attr(Gf.Vec3f(Tw[0][3], Tw[1][3], Tw[2][3]))
    fx.CreateLocalRot0Attr(Gf.Quatf(*P["T_world_root_quat_wxyz"]))
    fx.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fx.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    stage.GetRootLayer().Save()


def read(path):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open({path}) failed")
    out = {"usd_version": list(Usd.GetVersion()), "layer_format": stage.GetRootLayer().GetFileFormat().formatId,
           "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
           "kilograms_per_unit": UsdPhysics.GetStageKilogramsPerUnit(stage)}
    dp = stage.GetDefaultPrim()
    out["default_prim"] = str(dp.GetPath())
    out["articulation_roots"] = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    cache = UsdGeom.XformCache()
    links, joints, cols = {}, {}, {}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(p)
            q = m.GetPrincipalAxesAttr().Get()
            links[p.GetName()] = {
                "path": str(p.GetPath()), "T_world": _colvec(cache.GetLocalToWorldTransform(p)),
                "mass": m.GetMassAttr().Get(), "com": list(m.GetCenterOfMassAttr().Get()),
                "diagonal_inertia": list(m.GetDiagonalInertiaAttr().Get()),
                "principal_axes_wxyz": [q.GetReal(), *q.GetImaginary()],
                "visual_ply": p.GetAttribute("rigeuler:visualPly").Get().path,
                "T_link_ply": list(p.GetAttribute("rigeuler:T_link_ply").Get())}
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            mesh = UsdGeom.Mesh(p)
            cols[str(p.GetPath())] = {"parent": p.GetParent().GetName(), "n_points": len(mesh.GetPointsAttr().Get()),
                                      "approximation": UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get(),
                                      "T_world": _colvec(cache.GetLocalToWorldTransform(p)),
                                      "points": [list(v) for v in mesh.GetPointsAttr().Get()]}
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            r0, r1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
            d = {"type": p.GetTypeName(), "body0": [str(t) for t in j.GetBody0Rel().GetTargets()],
                 "body1": [str(t) for t in j.GetBody1Rel().GetTargets()],
                 "localPos0": list(j.GetLocalPos0Attr().Get()), "localRot0_wxyz": [r0.GetReal(), *r0.GetImaginary()],
                 "localPos1": list(j.GetLocalPos1Attr().Get()), "localRot1_wxyz": [r1.GetReal(), *r1.GetImaginary()],
                 "collision_enabled": j.GetCollisionEnabledAttr().Get()}
            if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
                jj = UsdPhysics.RevoluteJoint(p) if p.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(p)
                d.update(axis=jj.GetAxisAttr().Get(), lower=jj.GetLowerLimitAttr().Get(),
                         upper=jj.GetUpperLimitAttr().Get())
                tok = "angular" if p.IsA(UsdPhysics.RevoluteJoint) else "linear"
                dr = UsdPhysics.DriveAPI(p, tok)
                d["drive"] = {"token": tok, "stiffness": dr.GetStiffnessAttr().Get(),
                              "damping": dr.GetDampingAttr().Get(), "max_force": dr.GetMaxForceAttr().Get()}
                a = p.GetAttribute("physxJoint:maxJointVelocity")
                d["max_joint_velocity"] = a.Get() if a else None
                d["applied_schemas"] = list(p.GetAppliedSchemas())
            joints[p.GetName()] = d
    out.update(links=links, joints=joints, collisions=cols,
               root_metadata={a.GetName(): str(a.Get()) for a in dp.GetAttributes()
                              if a.GetName().startswith("rigeuler:")})
    try:
        from pxr import PhysxSchema  # noqa: F401
        out["physx_schema_plugin"] = True
    except ImportError:
        out["physx_schema_plugin"] = False
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write":
        payload = json.load(open(sys.argv[2]))
        write(payload, sys.argv[3])
        json.dump(read(sys.argv[3]), open(sys.argv[4], "w"))
    elif cmd == "read":
        json.dump(read(sys.argv[2]), open(sys.argv[3], "w"))
    else:
        raise SystemExit(f"unknown command {cmd}")
