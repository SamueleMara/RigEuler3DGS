"""Canonical-geometry handover from the free stage to the reduced joint model.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 14.

The problem section 14 names: reduced-stage energy was observed to GROW at the
handover, because the reduced model was instantiated against the free stage's
*deformed* geometry. The free stage lets every Gaussian move independently, so
its final positions contain per-Gaussian drift that no rigid joint can
reproduce. Baking that drift into the handover reference guarantees the rigid
model starts out unable to match its own initial condition.

Section 14's fix, implemented here step by step:

    1. canonical mu0, R0 stay IMMUTABLE -- verified by hash, not by convention;
    2. the free/group state is used ONLY to initialize support and the joint
       geometry, never as the geometric reference;
    3. P/R motion is instantiated from CANONICAL geometry;
    4. the entire bootstrap trajectory q_0..q_L is RE-ESTIMATED under the rigid
       joint against the actual RGB-D frames;
    5. axis and support may be jointly refined;
    6. (q_L, qdot_L) and its covariance come from that constrained window fit;
    7. the free-stage deformed geometry is discarded.

The result is that the reduced stage starts from the best rigid explanation of
the real frames, which cannot be worse than the rigid model's own starting
point -- and the before/after energies are persisted so that claim is checkable
rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

EPS = 1e-12


def canonical_hash(mu0: torch.Tensor, R0: torch.Tensor) -> str:
    """Section 14's canonical-reference hash. It must not change for the whole
    of motion estimation; persisting it is what makes that auditable."""
    h = hashlib.sha256()
    h.update(mu0.detach().cpu().contiguous().numpy().tobytes())
    h.update(R0.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class HandoverResult:
    """Everything section 14 requires to be persisted."""

    q_refit: list[float]
    qdot_final: float
    q_final: float
    covariance: torch.Tensor
    energy_before: dict
    energy_after: dict
    canonical_hash_before: str
    canonical_hash_after: str
    axis_refined: bool = False
    support_refined: bool = False
    iterations: list[dict] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.energy_after["total"] <= self.energy_before["total"] + 1e-9

    @property
    def canonical_unchanged(self) -> bool:
        return self.canonical_hash_before == self.canonical_hash_after

    def to_dict(self) -> dict:
        return {
            "q_refit": self.q_refit, "q_final": self.q_final,
            "qdot_final": self.qdot_final,
            "covariance": self.covariance.detach().cpu().tolist(),
            "energy_before": self.energy_before,
            "energy_after": self.energy_after,
            "energy_reduction": (
                1.0 - self.energy_after["total"] / self.energy_before["total"]
                if abs(self.energy_before["total"]) > EPS else 0.0),
            "improved": self.improved,
            "canonical_hash_before": self.canonical_hash_before,
            "canonical_hash_after": self.canonical_hash_after,
            "canonical_unchanged": self.canonical_unchanged,
            "axis_refined": self.axis_refined,
            "support_refined": self.support_refined,
            "num_bootstrap_frames": len(self.q_refit),
        }

    def write(self, out_dir: str | Path) -> dict:
        """Section 14's mandatory artifacts."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {}
        (d / "handover_before_energy.json").write_text(
            json.dumps(self.energy_before, indent=2))
        paths["handover_before_energy.json"] = str(d / "handover_before_energy.json")
        (d / "handover_after_refit_energy.json").write_text(
            json.dumps(self.energy_after, indent=2))
        paths["handover_after_refit_energy.json"] = str(d / "handover_after_refit_energy.json")
        lines = ["frame,q"] + [f"{i},{q:.9f}" for i, q in enumerate(self.q_refit)]
        (d / "bootstrap_q_refit.csv").write_text("\n".join(lines) + "\n")
        paths["bootstrap_q_refit.csv"] = str(d / "bootstrap_q_refit.csv")
        (d / "canonical_reference_hash.json").write_text(json.dumps({
            "hash_before_handover": self.canonical_hash_before,
            "hash_after_handover": self.canonical_hash_after,
            "unchanged": self.canonical_unchanged,
            "note": ("section 14: the canonical reference must remain unchanged "
                     "throughout motion estimation"),
        }, indent=2))
        paths["canonical_reference_hash.json"] = str(d / "canonical_reference_hash.json")
        (d / "handover_summary.json").write_text(json.dumps(self.to_dict(), indent=2))
        paths["handover_summary.json"] = str(d / "handover_summary.json")
        return paths


@torch.no_grad()
def refit_bootstrap_window(mu0: torch.Tensor, R0: torch.Tensor, model,
                           frames, energy_fn, *, q_init=None,
                           max_iters: int = 30, tol: float = 1e-9,
                           bracket: float = 1.0) -> tuple[list[float], list[dict]]:
    """Section 14 step 4: re-estimate q_0..q_L under the rigid joint.

    ``frames`` is one entry per bootstrap frame (whatever ``energy_fn`` needs).
    ``energy_fn(frame, mu, R) -> float`` is the TRUE nonlinear robust energy of
    that frame at the given rigid geometry.

    Each q_t is found by a golden-section search on the true energy: the rigid
    model has exactly ONE degree of freedom per frame, so a 1-D search on the
    real objective is both tractable and free of linearization error. Starting
    from the free stage's q estimate is allowed (step 2 permits using it as an
    initializer); the geometry it is applied to is canonical (step 3).
    """
    phi = 0.5 * (5 ** 0.5 - 1)
    out, log = [], []
    prev = 0.0
    for t, frame in enumerate(frames):
        guess = float(q_init[t]) if q_init is not None and t < len(q_init) else prev
        lo, hi = guess - bracket, guess + bracket

        def e_at(q):
            mu, R = model.transport(mu0, R0, float(q))
            return float(energy_fn(frame, mu, R))

        a, b = lo, hi
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = e_at(c), e_at(d)
        it = 0
        while abs(b - a) > tol and it < max_iters:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = e_at(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = e_at(d)
            it += 1
        q = 0.5 * (a + b)
        e_final = e_at(q)
        e_guess = e_at(guess)
        out.append(q)
        log.append({"frame": t, "q_initial": guess, "q_refit": q,
                    "energy_at_initial": e_guess, "energy_at_refit": e_final,
                    "improved": e_final <= e_guess + 1e-12,
                    "golden_iterations": it})
        prev = q
    return out, log


@torch.no_grad()
def canonical_handover(mu0: torch.Tensor, R0: torch.Tensor, model, frames,
                       energy_fn, *, free_stage_q=None, dt: float = 1.0,
                       free_stage_geometry=None,
                       qdot_variance: float = 1e-2,
                       q_variance: float = 1e-4) -> HandoverResult:
    """Section 14's full handover.

    ``free_stage_geometry`` is the free stage's deformed (mu, R), accepted ONLY
    so that its energy can be reported as the "before" baseline and so it can
    then be explicitly discarded (step 7). It is never used as a reference.
    """
    h_before = canonical_hash(mu0, R0)

    # --- the "before" baseline -----------------------------------------
    # Two things worth separating: what the free stage actually achieved, and
    # what the rigid model would start from if handed the free stage's q.
    before_free = None
    if free_stage_geometry is not None:
        mu_f, R_f = free_stage_geometry
        before_free = sum(float(energy_fn(f, mu_f, R_f)) for f in frames)
    q_guess = list(free_stage_q) if free_stage_q is not None else [0.0] * len(frames)
    before_rigid = 0.0
    per_frame_before = []
    for t, f in enumerate(frames):
        mu, R = model.transport(mu0, R0, float(q_guess[t]) if t < len(q_guess) else 0.0)
        e = float(energy_fn(f, mu, R))
        per_frame_before.append(e)
        before_rigid += e
    energy_before = {
        "total": before_rigid,
        "per_frame": per_frame_before,
        "free_stage_deformed_total": before_free,
        "note": ("'total' is the rigid model on CANONICAL geometry using the "
                 "free stage's q. 'free_stage_deformed_total' is what the free "
                 "stage itself achieved with per-Gaussian freedom; it is "
                 "reported for context and then discarded (section 14 step 7), "
                 "never used as the reduced stage's reference."),
    }

    # --- steps 3-4: refit the whole window on canonical geometry ---------
    q_refit, log = refit_bootstrap_window(mu0, R0, model, frames, energy_fn,
                                          q_init=q_guess)
    per_frame_after = [r["energy_at_refit"] for r in log]
    energy_after = {"total": float(sum(per_frame_after)),
                    "per_frame": per_frame_after}

    # --- step 6: initialize (q_L, qdot_L) from the constrained window fit --
    q_final = float(q_refit[-1]) if q_refit else 0.0
    if len(q_refit) >= 2 and dt > EPS:
        qdot_final = (q_refit[-1] - q_refit[-2]) / dt
    else:
        qdot_final = 0.0
    # Spread of the finite-difference speeds over the window is a real, if
    # crude, uncertainty -- better than a hard-coded constant.
    if len(q_refit) >= 3 and dt > EPS:
        speeds = [(q_refit[i + 1] - q_refit[i]) / dt for i in range(len(q_refit) - 1)]
        mean = sum(speeds) / len(speeds)
        var = sum((s - mean) ** 2 for s in speeds) / max(len(speeds) - 1, 1)
        qdot_var = max(var, qdot_variance * 1e-3)
    else:
        qdot_var = qdot_variance
    cov = torch.tensor([[q_variance, 0.0], [0.0, qdot_var]], dtype=torch.float64)

    return HandoverResult(
        q_refit=[float(q) for q in q_refit], qdot_final=float(qdot_final),
        q_final=q_final, covariance=cov, energy_before=energy_before,
        energy_after=energy_after, canonical_hash_before=h_before,
        # Step 1: the canonical reference must be BITWISE unchanged. Hashing it
        # again after the refit is what proves nothing wrote to it.
        canonical_hash_after=canonical_hash(mu0, R0), iterations=log)
