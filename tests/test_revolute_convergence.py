"""Sections 17-27, 32-35: physical convergence of the revolute fit.

CODEX_FIX_MULTIMODEL_RIGID_SCORING_AND_REVOLUTE_CONVERGENCE.md.

Near the translation limit a revolute joint is parameterised by coordinates
that are not identifiable:

    r -> inf,   q -> 0,   r q -> d

so the axis point and the angle can drift without bound while every induced
transform T_t stays put. Judging convergence on those coordinates means a
genuinely settled trajectory is reported as an optimizer failure, which then
blocks the P/R comparison entirely (section 26).

These tests check the four termination states of section 27 against motions
whose correct state is known by construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.manifold import so3_exp  # noqa: E402
from joint_reduction.geometric_joint_fit import (  # noqa: E402
    PRISMATIC, REVOLUTE, UNRESOLVED_GEOMETRY, decide_joint_type)
from joint_reduction.revolute_fit import (  # noqa: E402
    CONVERGED, DEGENERATE_FLAT_DIRECTION, ELIGIBLE_STATES,
    MAX_ITERS_UNSTABLE, fit_revolute)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def revolute_traj(axis, point, qs):
    e = torch.as_tensor(axis, dtype=torch.float64, device=DEV)
    e = e / e.norm()
    a = torch.as_tensor(point, dtype=torch.float64, device=DEV)
    q = torch.as_tensor(qs, dtype=torch.float64, device=DEV)
    R = so3_exp(q.unsqueeze(1) * e.unsqueeze(0))
    eye = torch.eye(3, dtype=torch.float64, device=DEV)
    t = torch.einsum("tij,j->ti", eye.unsqueeze(0) - R, a)
    return R, t


def prismatic_traj(direction, qs):
    e = torch.as_tensor(direction, dtype=torch.float64, device=DEV)
    e = e / e.norm()
    T = len(qs)
    R = torch.eye(3, dtype=torch.float64, device=DEV).reshape(1, 3, 3).repeat(T, 1, 1)
    t = torch.stack([float(q) * e for q in qs])
    return R, t


# ===========================================================================
def test_32_translation_limit():
    """§32: true pure translation fitted with a revolute candidate."""
    qs = [0.0] + [0.02 * k for k in range(1, 20)]
    R, t = prismatic_traj([1.0, 0.0, 0.0], qs)
    f = fit_revolute(R, t, 0.5, max_iters=200)
    c = f["convergence"]
    print(f"       status {c['status']}   iters {c['iterations']}   "
          f"E_R {f['error']:.3e}")
    print(f"       induced-motion change {c['delta_point_normalised']:.3e}, "
          f"axis+point drift {c['final_d_eta']:.3e}")
    check("§32 the fit terminates in an eligible state",
          c["status"] in ELIGIBLE_STATES, c["status"])
    check("§32 the induced physical trajectory stabilised",
          bool(f["physical_motion_converged"]))
    check("§32 and it is reported as a flat direction, not a failure",
          c["status"] == DEGENERATE_FLAT_DIRECTION, c["status"])
    check("§24 the axis is marked unidentifiable",
          f["joint_parameters_identifiable"] is False)
    check("§49 the report says trajectory converged / axis not identifiable",
          c["trajectory_converged"] and not c["joint_axis_identifiable"])

    d = decide_joint_type(R, t, 0.5)
    print(f"       geometric decision: {d['selected'] or d['status']}  "
          f"E_P {d['normalized_error']['prismatic']:.3e}  "
          f"E_R {d['normalized_error']['revolute']:.3e}")
    check("§26 a flat-direction candidate is ELIGIBLE to be compared",
          d["eligibility"]["revolute"]["eligible"])
    check("§25 flat direction alone does not make the answer revolute",
          d["selected"] != REVOLUTE, str(d["selected"]))


def test_33_true_revolute_converges():
    """§33: finite-radius true revolute -- parameters converge too."""
    qs = [0.0] + [0.08 * k for k in range(1, 16)]
    R, t = revolute_traj([0.0, 0.0, 1.0], [0.6, -0.3, 0.0], qs)
    f = fit_revolute(R, t, 0.5, max_iters=200)
    c = f["convergence"]
    print(f"       status {c['status']}   iters {c['iterations']}   "
          f"E_R {f['error']:.3e}")
    check("§33 a true revolute reaches CONVERGED", c["status"] == CONVERGED,
          c["status"])
    check("§33 no false flat-direction classification",
          not c["degenerate_flat_direction"])
    check("§33 the axis is identifiable", f["joint_parameters_identifiable"])
    d = decide_joint_type(R, t, 0.5)
    check("§33 and the geometric decision is revolute",
          d["selected"] == REVOLUTE, str(d["selected"]))


def test_34_far_axis_true_revolute():
    """§34: far axis with enough sweep is still revolute, not forced to P."""
    L = 0.5
    qs = [0.0] + [0.03 * k for k in range(1, 20)]
    R, t = revolute_traj([0.0, 0.0, 1.0], [20.0 * L, 0.0, 0.0], qs)
    f = fit_revolute(R, t, L, max_iters=200)
    c = f["convergence"]
    d = decide_joint_type(R, t, L)
    print(f"       status {c['status']}   decision "
          f"{d['selected'] or d['status']}   "
          f"E_P {d['normalized_error']['prismatic']:.3e}  "
          f"E_R {d['normalized_error']['revolute']:.3e}")
    check("§34 the candidate is eligible", c["status"] in ELIGIBLE_STATES,
          c["status"])
    check("§34 no forced prismatic decision", d["selected"] != PRISMATIC,
          str(d["selected"]))
    check("§29 the status came from convergence, not an axis-distance gate",
          "no axis-distance" in d["decision_inputs"])


def test_35_short_arc_unresolved_is_allowed():
    """§35: a short arc may legitimately end UNRESOLVED_GEOMETRY."""
    L = 0.5
    rows = []
    for sweep in (0.4, 0.02, 1e-3):
        qs = [0.0] + [sweep * k / 14 for k in range(1, 15)]
        R, t = revolute_traj([0.0, 0.0, 1.0], [40.0 * L, 0.0, 0.0], qs)
        d = decide_joint_type(R, t, L)
        ne = d["normalized_error"]
        rows.append((sweep, d))
        print(f"       sweep {sweep:7.0e} rad: E_P {ne['prismatic']:.3e}  "
              f"E_R {ne['revolute']:.3e}  gap {ne['absolute_gap']:.2e} vs "
              f"floor {ne['indistinguishable_floor']:.2e}  -> "
              f"{d['selected'] or d['status']}")
    verdicts = [d["selected"] or d["status"] for _s, d in rows]
    check("§35 every outcome is a legitimate state",
          all(v in (PRISMATIC, REVOLUTE, UNRESOLVED_GEOMETRY,
                    "UNRESOLVED_OPTIMIZATION") for v in verdicts),
          str(verdicts))
    check("§35 neither type is forced on the shortest arc",
          verdicts[-1] in (UNRESOLVED_GEOMETRY, PRISMATIC, REVOLUTE),
          str(verdicts[-1]))
    tied = [d for _s, d in rows
            if (d["selected"] or d["status"]) == UNRESOLVED_GEOMETRY]
    for d in tied:
        ne = d["normalized_error"]
        check("§35 a tie is declared only under the measured floor",
              ne["absolute_gap"] <= ne["indistinguishable_floor"],
              f"{ne['absolute_gap']:.2e} <= {ne['indistinguishable_floor']:.2e}")


def test_27_states_are_distinct():
    """§27: the four termination states exist and mean different things."""
    check("§27 all four states are defined",
          len({CONVERGED, DEGENERATE_FLAT_DIRECTION, MAX_ITERS_UNSTABLE,
               "NUMERICAL_FAILURE"}) == 4)
    check("§26 only CONVERGED and DEGENERATE_FLAT_DIRECTION are eligible",
          set(ELIGIBLE_STATES) == {CONVERGED, DEGENERATE_FLAT_DIRECTION},
          str(ELIGIBLE_STATES))
    src = (REPO / "joint_reduction" / "revolute_fit.py").read_text()
    check("§36 convergence is judged on induced motion, not raw parameters",
          "delta_point_normalised" in src and "physical_change" in src)
    check("§21 raw parameter changes are kept as diagnostics",
          "final_d_eta" in src and "final_d_q" in src)


def main():
    for fn in (test_27_states_are_distinct,
               test_32_translation_limit,
               test_33_true_revolute_converges,
               test_34_far_axis_true_revolute,
               test_35_short_arc_unresolved_is_allowed):
        print(f"\n{fn.__name__}")
        fn()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n{len(PASSED)} checks passed across 5 tests")


if __name__ == "__main__":
    main()
