"""Deterministic SE(3) conventions of the simulation asset (spec section 3). float64 throughout.

Column-vector transforms: T_A_B maps coordinates in frame B to frame A.
    Wc  Phase-5 canonical capture world (the XYZ stored in both visual PLYs)
    B   static_link frame, chosen EQUAL to Wc  (T_B_Pstatic = I)
    J0  moving_link frame at the canonical Phase-3 anchor q = 0, local +Z = joint axis a
    W   simulator world, T_W_B from a registration or the declared gauge W = Wc

    T_B_J0          = [R_B_J, o; 0 1]
    T_J0_Pmoving    = inv(T_B_J0)                  (PLY coordinates stay in Wc = B)
    T_B_Lmoving(q)  = T_B_J0 @ M_z(q)
    T_B_Pmoving(q)  = T_B_J0 @ M_z(q) @ inv(T_B_J0)
which is exactly recovery.contracts.Joint.transform:
    revolute   x(q) = Exp([a]x q)(x - o) + o        prismatic  x(q) = x + q a
"""
from __future__ import annotations

import math

import numpy as np

SE3_ORTHO_TOL = 1e-6        # |R^T R - I|_max accepted for user-supplied rigid transforms (typed / exported matrices)


class FrameError(ValueError):
    pass


def axis_frame(a) -> np.ndarray:
    """R_B_J: orthonormal, right-handed, third column = a. The auxiliary vector is the basis vector with the
    smallest |dot| with a (lowest index on ties), so the result is a deterministic function of a."""
    a = np.asarray(a, np.float64)
    n = np.linalg.norm(a)
    if a.shape != (3,) or not np.isfinite(a).all() or n == 0:
        raise FrameError(f"joint axis {a} is not a finite non-zero 3-vector")
    a = a / n
    k = int(np.argmin(np.abs(a)))
    e = np.zeros(3)
    e[k] = 1.0
    x = e - (e @ a) * a
    x /= np.linalg.norm(x)
    y = np.cross(a, x)
    R = np.stack([x, y, a], axis=1)
    if abs(np.linalg.det(R) - 1.0) > 1e-12 or np.abs(R.T @ R - np.eye(3)).max() > 1e-12:
        raise FrameError("axis_frame produced a non-rotation (internal error)")
    return R


def se3(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, np.float64)
    T[:3, 3] = np.asarray(t, np.float64)
    return T


def inv_se3(T) -> np.ndarray:
    T = np.asarray(T, np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def check_se3(T, name="transform", tol=SE3_ORTHO_TOL) -> np.ndarray:
    """A 4x4 rigid transform: orthonormal rotation with det +1, bottom row [0 0 0 1], no scale / shear."""
    T = np.asarray(T, np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise FrameError(f"{name}: expected a finite 4x4 matrix, got shape {T.shape}")
    if np.abs(T[3] - [0, 0, 0, 1]).max() > 0:
        raise FrameError(f"{name}: bottom row must be exactly [0, 0, 0, 1], got {T[3].tolist()}")
    R = T[:3, :3]
    e = np.abs(R.T @ R - np.eye(3)).max()
    if e > tol:
        raise FrameError(f"{name}: rotation block is not orthonormal (|R^T R - I| = {e:.3g} > {tol}); scale or shear "
                         f"inside an SE(3) is not allowed -- convert units explicitly instead")
    if np.linalg.det(R) < 0:
        raise FrameError(f"{name}: det(R) = {np.linalg.det(R):.6f} < 0 (reflection / handedness flip)")
    return T


def rot_z(q) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def M_z(joint_type: str, q: float) -> np.ndarray:
    if joint_type == "revolute":
        return se3(rot_z(q), np.zeros(3))
    if joint_type == "prismatic":
        return se3(np.eye(3), [0.0, 0.0, float(q)])
    raise FrameError(f"joint type {joint_type!r}")


def joint_frame(joint_type: str, axis, point=None, moving_xyz=None) -> tuple[np.ndarray, np.ndarray, str]:
    """(T_B_J0, o, origin_kind). Revolute: o = joint.point EXACTLY. Prismatic: o = the moving canonical XYZ centroid,
    a documented frame gauge that cancels from the translation kinematics (not a fitted joint location)."""
    R = axis_frame(axis)
    if joint_type == "revolute":
        if point is None:
            raise FrameError("revolute joint without a hinge point")
        o, kind = np.asarray(point, np.float64), "revolute_hinge_point"
    elif joint_type == "prismatic":
        if moving_xyz is None or len(moving_xyz) == 0:
            raise FrameError("prismatic gauge needs the moving canonical XYZ")
        o, kind = np.asarray(moving_xyz, np.float64).mean(0), "prismatic_gauge_moving_xyz_centroid"
    else:
        raise FrameError(f"joint type {joint_type!r}")
    if o.shape != (3,) or not np.isfinite(o).all():
        raise FrameError(f"joint origin {o} is not finite")
    return se3(R, o), o, kind


def T_B_L(joint_type, T_B_J0, q) -> np.ndarray:
    return np.asarray(T_B_J0) @ M_z(joint_type, q)


def T_B_P_moving(joint_type, T_B_J0, q) -> np.ndarray:
    return T_B_L(joint_type, T_B_J0, q) @ inv_se3(T_B_J0)


def apply(T, x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


# ----------------------------------------------------------------------------------------- rotation formats
def quat_wxyz_from_R(R) -> np.ndarray:
    """Unit quaternion (w, x, y, z), Hamilton, scalar first, w >= 0 (Shepperd's method)."""
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def R_from_quat_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a ⊗ b of (w, x, y, z) quaternions, batched over leading dims."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def urdf_rpy_from_R(R) -> np.ndarray:
    """URDF <origin rpy>: fixed-axis roll-pitch-yaw, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    R = np.asarray(R, np.float64)
    sp = -R[2, 0]
    if abs(sp) < 1.0 - 1e-12:
        pitch = math.asin(max(-1.0, min(1.0, sp)))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal: pitch = +-pi/2, roll folded into yaw
        pitch = math.copysign(math.pi / 2, sp)
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def R_from_urdf_rpy(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ----------------------------------------------------------------------------------------- inertia
def transform_inertial(T_L_S, mass, com_S, I_com_S):
    """Rigid change of frame S -> L of (mass, COM, inertia about the COM). I_L = R I_S R^T, com_L = T com_S."""
    T = check_se3(T_L_S, "inertial transform")
    R = T[:3, :3]
    I = np.asarray(I_com_S, np.float64)
    return float(mass), apply(T, com_S), R @ I @ R.T


def principal_axes(I) -> tuple[np.ndarray, np.ndarray]:
    """(diag, V) with I = V diag V^T, V a proper rotation (det +1), eigenvalues ascending."""
    I = 0.5 * (np.asarray(I, np.float64) + np.asarray(I, np.float64).T)
    d, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    return d, V
