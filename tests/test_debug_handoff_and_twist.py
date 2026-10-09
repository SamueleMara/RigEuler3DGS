"""CODEX_DEBUG_SOFTP_GRADIENTS_STORAGE_DIAGNOSIS, Part VIII (§40-§46).

Soft-P handoff (§40, §41), weighted gradient probes (§42, §43), twist-level
gradients (§44), and the screw-geometry convention (§45, §46).

§45 is the one that must not be guessed: the spec explicitly forbids
hard-coding an untested sign for the axis-point formula, so the convention is
pinned against a constructed rotation here.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import tempfile
from pathlib import Path

import torch

from rigid_parts import shared_killing_solver as sks
from rigid_parts.handoff_diagnostics import (
    format_softp_handoff,
    polarization_row,
    soft_probability_statistics,
)
from rigid_parts.inner_support import build_robust_inner_support
from rigid_parts.robust_inner_killing import rigid_field, se3_exp_twist
from rigid_parts.twist_geometry import (
    axis_distance_statistics,
    rotational_field_variation,
    screw_decomposition,
    twist_gradient_diagnostics,
)

DT = torch.float64
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# §40 / §41 soft-P handoff
# ---------------------------------------------------------------------------
def test_40_support_receives_soft_probabilities_unchanged():
    """§9/§40: the exact spec fixture. Support must see these values, not one-hot."""
    p = torch.tensor([[0.40, 0.60], [0.20, 0.80], [0.48, 0.52], [0.05, 0.95]], dtype=DT)
    xyz = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]], dtype=DT)
    z = p.argmax(dim=-1)

    seen: dict = {}
    original = build_robust_inner_support

    def spy(*args, **kwargs):
        seen["probs"] = kwargs["soft_part_probs"].clone()
        return original(*args, **kwargs)

    sks.build_robust_inner_support = spy
    try:
        support = original(xyz, p, z, 1, knn=2, cluster_knn=2)
    finally:
        sks.build_robust_inner_support = original

    # The support object itself must carry the soft values through.
    check("§40 dynamic confidence is the soft value, not 1.0",
          torch.allclose(support.dynamic_confidence, p[:, 1], atol=1e-12),
          str([round(float(v), 3) for v in support.dynamic_confidence]))
    check("§40 no value was rounded to one-hot",
          float(support.dynamic_confidence.min()) < 0.99
          and float(support.dynamic_confidence.max()) < 1.0 - 1e-9)

    stats = soft_probability_statistics(p, z)
    check("§40 a soft input is NOT flagged as hard", not stats["input_is_effectively_hard"])
    check("§40 L1-to-onehot is strictly positive", stats["l1_to_onehot_mean"] > 0.1,
          f"{stats['l1_to_onehot_mean']:.4f}")

    onehot = torch.zeros_like(p).scatter_(1, z[:, None], 1.0)
    hard_stats = soft_probability_statistics(onehot, z)
    check("§5 a one-hot input IS flagged as hard", hard_stats["input_is_effectively_hard"])
    check("§5 L1-to-onehot is exactly 0 for one-hot",
          hard_stats["l1_to_onehot_mean"] == 0.0, f"{hard_stats['l1_to_onehot_mean']:.3e}")
    check("§5 the two inputs are distinguishable",
          stats["l1_to_onehot_mean"] > 1e3 * max(hard_stats["l1_to_onehot_mean"], 1e-12))
    hard_support = build_robust_inner_support(xyz, onehot, z, 1, knn=2, cluster_knn=2)
    check("§40 one-hot collapses every confidence to 1.0",
          float(hard_support.dynamic_confidence[z > 0].min()) == 1.0)


def test_40b_pipeline_hands_over_the_persistent_soft_p():
    """§8: the integration point must pass the persistent P, and name it."""
    src = Path(inspect.getfile(
        __import__("rigid_parts.nested_pipeline", fromlist=["x"]))).read_text(encoding="utf-8")
    check("§8 the persistent soft P is forwarded",
          '_extra["soft_part_probs"] = soft_part_probs.detach()' in src)
    check("§8 the source tensor is named",
          '"soft_part_probs (persistent outer state)"' in src)
    check("§8 the one-hot fallback is labelled as such",
          "render_graph/legacy-hard mode has no soft P" in src)
    solver = Path(sks.__file__).read_text(encoding="utf-8")
    check("§3 statistics come from the tensor actually passed",
          "handoff = soft_probability_statistics(support_probs, z)" in solver)
    check("§26 the support builder is given that same tensor",
          "soft_part_probs=support_probs" in solver)
    # The old bug: prob (the frozen one-hot) reaching the support builder.
    check("§8 the frozen one-hot is no longer the support input",
          "soft_part_probs=prob," not in solver)


def test_41_handoff_artifacts():
    """§7/§41: both artifacts exist and carry valid statistics."""
    solver = Path(sks.__file__).read_text(encoding="utf-8")
    for artifact in ("soft_probability_handoff.pt", "soft_probability_handoff.json"):
        check(f"§7 {artifact} is written", artifact in solver)
    for field in ('"P": support_probs', '"z": z.detach', '"dynamic_confidence"',
                  '"max_part_confidence"'):
        check(f"§7 the PT payload has {field}", field in solver)

    p = torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.5, 0.5]], dtype=DT)
    stats = soft_probability_statistics(p, p.argmax(dim=-1))
    for key in ("row_sum_error", "entropy_mean", "entropy_median", "entropy_p10",
                "fraction_entries_near_binary", "fraction_rows_near_one_hot",
                "fraction_max_above_0.90", "fraction_max_above_0.99",
                "fraction_max_above_0.999", "l1_to_onehot_mean", "l1_to_onehot_median",
                "l1_to_onehot_max", "fraction_l1_below_1e-6", "fraction_l1_below_1e-3",
                "fraction_l1_below_1e-2", "candidate_count"):
        check(f"§4/§5/§6 statistic {key} is present", key in stats)
    check("§4 row-sum error is ~0 for normalised P", stats["row_sum_error"] < 1e-12)
    check("§6 candidate confidence is reported", "candidate_dynamic" in stats)
    check("§8 the console block renders", "[SoftPHandoff]" in format_softp_handoff("P", p, stats))
    check("§10 the warning appears only for hard input",
          "WARNING" not in format_softp_handoff("P", p, stats))
    onehot = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DT)
    check("§10 the warning appears for hard input",
          "WARNING: INNER SUPPORT RECEIVED NEAR-HARD" in format_softp_handoff(
              "P", onehot, soft_probability_statistics(onehot, onehot.argmax(dim=-1))))


def test_12_polarization_row():
    """§11/§12: the per-outer polarization row."""
    a = torch.tensor([[0.5, 0.5], [0.6, 0.4]], dtype=DT)
    b = torch.tensor([[0.05, 0.95], [0.99, 0.01]], dtype=DT)
    row = polarization_row(3, b, a)
    for key in ("outer", "entropy_mean", "entropy_median", "fraction_max_above_0.90",
                "fraction_max_above_0.99", "fraction_one_hot_like", "mean_l1_to_onehot",
                "mean_delta_P", "argmax_label_changes"):
        check(f"§12 row has {key}", key in row)
    check("§12 polarization is detected", row["entropy_mean"] < polarization_row(3, a)["entropy_mean"])
    check("§11 delta P is measured", row["mean_delta_P"] > 0.5, f"{row['mean_delta_P']:.4f}")
    check("§11 label flips are counted", row["argmax_label_changes"] == 1,
          str(row["argmax_label_changes"]))


# ---------------------------------------------------------------------------
# §42 / §43 weighted gradient probes
# ---------------------------------------------------------------------------
def test_42_gradient_norms_use_weighted_losses():
    """§14/§42: the reported norm must be of lambda*L, not L."""
    a = torch.nn.Parameter(torch.ones(4, dtype=DT))
    raw = {"x": (2.0 * a).sum()}
    lam = {"x": 7.0}
    weighted = sks.per_loss_gradient_norms({"x": lam["x"] * raw["x"]}, [a])
    unweighted = sks.per_loss_gradient_norms(raw, [a])
    check("§42 the weighted norm is lambda times the raw norm",
          abs(weighted["x"] - lam["x"] * unweighted["x"]) < 1e-12,
          f"{weighted['x']:.6f} vs {lam['x'] * unweighted['x']:.6f}")
    check("§42 a zero lambda gives a zero norm",
          sks.per_loss_gradient_norms({"x": 0.0 * raw["x"]}, [a])["x"] == 0.0)
    src = inspect.getsource(sks.solve_shared_killing_sequence)
    check("§15/§16 the solver probes the WEIGHTED terms",
          "_weighted_terms(" in src, "no weighted-term construction found")


def test_43_probes_do_not_contaminate_grad():
    """§17/§43: diagnostic probes must not touch .grad."""
    a = torch.nn.Parameter(torch.ones(3, dtype=DT))
    b = torch.nn.Parameter(torch.ones(3, dtype=DT))
    a.grad = torch.full((3,), 5.0, dtype=DT)
    before = a.grad.clone()
    sks.per_loss_gradient_norms({"one": (a * b).sum(), "two": (a * a).sum()}, [a, b])
    check("§43 an existing .grad is unchanged", torch.equal(a.grad, before))
    check("§43 an absent .grad stays absent", b.grad is None)
    check("§17 the probe uses autograd.grad, not backward",
          "torch.autograd.grad(" in inspect.getsource(sks.per_loss_gradient_norms)
          and ".backward()" not in inspect.getsource(sks.per_loss_gradient_norms))


def test_18_gradient_ratio_reporting():
    """§18/§19: data-vs-regularizer ratios, computed from GRADIENT norms."""
    geo = {"render": 1.0, "depth": 0.0, "motion_direction": 3.0, "final_pcd": 2.0,
           "strain": 12.0, "rigid": 0.0}
    r = sks.gradient_ratios(geo, block="geometry")
    check("§18 G_data sums the informative terms", abs(r["G_data"] - 6.0) < 1e-12)
    check("§18 G_reg sums the regularizers", abs(r["G_reg"] - 12.0) < 1e-12)
    check("§18 R = G_reg / G_data", abs(r["R"] - 2.0) < 1e-9, f"{r['R']:.4f}")
    check("§18 motion_dir/render is reported", abs(r["motion_dir_over_render"] - 3.0) < 1e-9)
    check("§18 strain/render is reported", abs(r["strain_over_render"] - 12.0) < 1e-9)
    sc = {"render": 2.0, "depth": 0.0, "motion_magnitude": 1.0, "final_pcd": 1.0,
          "continuity": 0.5, "flip": 8.0, "jerk": 0.5}
    rs = sks.gradient_ratios(sc, block="scalar")
    check("§19 scalar G_data", abs(rs["G_data"] - 4.0) < 1e-12)
    check("§19 scalar G_reg", abs(rs["G_reg"] - 9.0) < 1e-12)
    check("§19 flip/render is reported", abs(rs["flip_over_render"] - 4.0) < 1e-9)
    check("§19 flip/motion_mag is reported", abs(rs["flip_over_motion_mag"] - 8.0) < 1e-9)


# ---------------------------------------------------------------------------
# §44 / §45 / §46 twist-level and screw diagnostics
# ---------------------------------------------------------------------------
def test_44_twist_gradients():
    """§22/§23/§44: dL/domega and dL/du, finite and correctly shaped."""
    torch.manual_seed(5)
    x = torch.rand((20, 3), dtype=DT) * 2 - 1
    omega = torch.tensor([0.1, -0.2, 0.7], dtype=DT, requires_grad=True)
    u = torch.tensor([0.3, 0.0, -0.1], dtype=DT, requires_grad=True)
    target = torch.randn((20, 3), dtype=DT)
    losses = {"a": (rigid_field(x, omega, u) - target).square().mean(),
              "b": rigid_field(x, omega, u).abs().mean()}
    out = twist_gradient_diagnostics(losses, omega, u)
    for name in ("a", "b"):
        check(f"§23 grad_omega norm for {name}", math.isfinite(out["grad_omega_norm"][name])
              and out["grad_omega_norm"][name] > 0)
        check(f"§23 grad_u norm for {name}", math.isfinite(out["grad_u_norm"][name])
              and out["grad_u_norm"][name] > 0)
    check("§24 pairwise cosines are reported", "cosine_omega" in out and "cosine_u" in out)
    key = "a|b" if "a|b" in out["cosine_omega"] else list(out["cosine_omega"])[0]
    check("§24 cosine is in [-1, 1]", -1.0 - 1e-9 <= out["cosine_omega"][key] <= 1.0 + 1e-9,
          f"{out['cosine_omega'][key]:.4f}")
    check("§22 the probe leaves .grad alone", omega.grad is None and u.grad is None)

    # A loss that cannot see omega must report exactly zero there.
    only_u = {"u_only": (u * u).sum()}
    out2 = twist_gradient_diagnostics(only_u, omega, u)
    check("§23 an omega-independent loss reports 0", out2["grad_omega_norm"]["u_only"] == 0.0)
    check("§23 and a non-zero u gradient", out2["grad_u_norm"]["u_only"] > 0)


def test_45_screw_axis_convention():
    """§27/§28/§45: pitch, axis direction and axis POINT, with the sign of the
    axis-point formula verified against this repository's v(x) = u + omega x x,
    not assumed."""
    # A pure rotation of magnitude |omega| about the z axis through point c.
    omega = torch.tensor([0.0, 0.0, 2.0], dtype=DT)
    c = torch.tensor([0.7, -0.4, 0.0], dtype=DT)
    # v(x) = omega x (x - c) = (omega x x) + (c x omega)  =>  u = c x omega
    u = torch.linalg.cross(c, omega)
    dec = screw_decomposition(omega, u)

    check("§26 |omega| is reported", abs(dec["omega_norm"] - 2.0) < 1e-12)
    check("§27 a pure rotation has zero pitch", abs(dec["pitch"]) < 1e-12, f"{dec['pitch']:.3e}")
    check("§26 the axis direction is +z",
          abs(abs(float(torch.tensor(dec["axis_direction"], dtype=DT) @
                        torch.tensor([0.0, 0.0, 1.0], dtype=DT))) - 1.0) < 1e-12)
    # The axis point must reproduce c up to a slide along the axis.
    point = torch.tensor(dec["axis_point"], dtype=DT)
    residual = (point - c) - ((point - c) @ omega / omega.square().sum()) * omega
    check("§28 the axis point lies on the true axis (sign verified)",
          float(residual.norm()) < 1e-12, f"offset {float(residual.norm()):.3e}")
    check("§28 the axis point is the minimum-norm one (perpendicular to omega)",
          abs(float(point @ omega)) < 1e-12, f"{float(point @ omega):.3e}")
    # And the field really vanishes there, which is what "on the axis" means.
    check("§28 the velocity vanishes on the axis",
          float(rigid_field(point[None], omega, u).norm()) < 1e-12)

    # A screw: pitch is the along-axis component per unit rotation.
    u_screw = u + 0.35 * omega / omega.norm()
    screw = screw_decomposition(omega, u_screw)
    check("§27 a screw has non-zero pitch", abs(screw["pitch"]) > 1e-6, f"{screw['pitch']:.4f}")
    check("§27 pitch = <omega,u>/|omega|^2",
          abs(screw["pitch"] - float((omega * u_screw).sum() / omega.square().sum())) < 1e-12)

    # A pure translation has no axis at all.
    pure = screw_decomposition(torch.zeros(3, dtype=DT), torch.tensor([0.0, 1.0, 0.0], dtype=DT))
    check("§26 a pure translation reports no axis", pure["axis_point"] is None
          and pure["is_pure_translation"])


def test_46_far_center_rotation_is_pseudo_translation():
    """§29/§30/§46: a rotation whose axis is far from a small cluster looks like
    a translation over that cluster -- which is the candidate explanation for
    storage's |omega| ~ 1 against a prismatic GT."""
    torch.manual_seed(7)
    cluster = torch.rand((60, 3), dtype=DT) * 0.2                # small support
    w = torch.ones(60, dtype=DT)

    # Axis far away along +x, rotation about z.
    omega = torch.tensor([0.0, 0.0, 0.01], dtype=DT)
    far_c = torch.tensor([200.0, 0.0, 0.0], dtype=DT)
    u_far = torch.linalg.cross(far_c, omega)
    dec_far = screw_decomposition(omega, u_far)
    dist_far = axis_distance_statistics(cluster, w, dec_far)
    var_far = rotational_field_variation(cluster, w, omega, u_far)

    # A nearby axis through the cluster.
    near_c = torch.tensor([0.1, 0.1, 0.0], dtype=DT)
    u_near = torch.linalg.cross(near_c, omega)
    dec_near = screw_decomposition(omega, u_near)
    dist_near = axis_distance_statistics(cluster, w, dec_near)
    var_near = rotational_field_variation(cluster, w, omega, u_near)

    check("§29 the far axis is far in normalized units",
          dist_far["median_over_bbox"] > 100, f"{dist_far['median_over_bbox']:.1f}")
    check("§29 the near axis is close", dist_near["median_over_bbox"] < 2.0,
          f"{dist_near['median_over_bbox']:.3f}")
    check("§30 the far rotation varies little relative to its translation",
          var_far["rot_over_trans"] < 1e-2, f"{var_far['rot_over_trans']:.3e}")
    check("§30 the near rotation does not look like a translation",
          var_near["rot_over_trans"] > 10 * var_far["rot_over_trans"],
          f"near {var_near['rot_over_trans']:.3e} far {var_far['rot_over_trans']:.3e}")
    check("§46 the far case is flagged as pseudo-translation",
          dist_far["pseudo_translation"] and not dist_near["pseudo_translation"])
    for key in ("min", "median", "mean", "p90", "median_over_bbox"):
        check(f"§29 axis distance reports {key}", key in dist_far)


def test_twist_geometry_artifacts_are_wired():
    """§25/§48: the diagnostics reach disk."""
    src = inspect.getsource(sks.solve_shared_killing_sequence)
    for artifact in ("loss_gradient_diagnostics.json", "twist_gradient_diagnostics.json",
                     "twist_geometry_diagnostics.json"):
        check(f"§48 {artifact} is written", artifact in src)
    check("§23 twist gradients are computed per geometry loss",
          "twist_gradient_diagnostics(" in src)
    check("§26-§30 screw geometry is computed", "screw_decomposition(" in src
          and "axis_distance_statistics(" in src and "rotational_field_variation(" in src)
    check("§20 no automatic lambda adaptation was added",
          "GradNorm" not in Path(sks.__file__).read_text(encoding="utf-8"))


def test_2_prior_stays_off_by_default():
    """§2/§50: the prismatic prior must remain OFF by default."""
    args_src = Path("arguments/__init__.py").read_text(encoding="utf-8")
    block = args_src.split('"--inner_motion_prior"')[1][:200]
    check("§2 the motion prior defaults to none", 'default="none"' in block, block[:80])
    check("§2 the prior weights default to 0",
          '"--lambda_prismatic_prior", type=float, default=0.0' in args_src
          and '"--lambda_revolute_prior", type=float, default=0.0' in args_src)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
