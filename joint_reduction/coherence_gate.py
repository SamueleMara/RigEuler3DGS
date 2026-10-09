"""Multi-criterion reduction gate over a sliding window.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 18.

Section 18 closes with the instruction that motivated this file:

    "Do not simply increase --coherence-residual from 0.35 to 0.55 to make the
     ladder finish."

That is exactly what the previous USB run needed, and it was a symptom, not a
fix. A single residual threshold asks one question -- "is the velocity field
well explained by one twist right now?" -- and that question has the wrong
answer early in a sequence for reasons that have nothing to do with whether the
motion is a joint: the twist estimate is still converging, the support is still
moving around, and the residual is dominated by Gaussians that do not belong to
the part at all.

So the gate asks several independent questions over a WINDOW instead, each in
units that make "stable" meaningful:

  * has the twist DIRECTION stopped turning           (theta_omega, degrees)
  * has the speed MAGNITUDE stopped changing          (Delta_v, relative)
  * is the twist estimate itself precise              (covariance / rank)
  * is the residual small RELATIVE TO ITS OWN UNCERTAINTY, not to an
    absolute constant
  * is the support set stable                         (Jaccard)
  * is the state update locally certified             (section 17's status)
  * is the model choice stable                        (same winner each frame)

Reduction triggers only when all of the configured criteria agree, which is
section 18's "stability + observability, not only point scatter".
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch

EPS = 1e-12


@dataclass
class WindowSample:
    """One frame's evidence for the gate."""

    frame: int
    omega: torch.Tensor | None = None       # [3] fitted angular velocity
    v: torch.Tensor | None = None           # [3] fitted linear velocity
    residual: float = float("nan")          # relative twist-fit residual
    residual_sigma: float = float("nan")    # predicted velocity uncertainty
    twist_covariance_trace: float = float("nan")
    observable_rank: int = 0
    support: torch.Tensor | None = None     # [C] bool/int support labels
    certificate_status: str = ""
    model_name: str = ""
    model_score: float = float("nan")

    def to_dict(self) -> dict:
        return {"frame": self.frame,
                "omega": None if self.omega is None else self.omega.tolist(),
                "v": None if self.v is None else self.v.tolist(),
                "residual": self.residual,
                "residual_sigma": self.residual_sigma,
                "twist_covariance_trace": self.twist_covariance_trace,
                "observable_rank": self.observable_rank,
                "certificate_status": self.certificate_status,
                "model_name": self.model_name,
                "model_score": self.model_score,
                "support_size": (None if self.support is None
                                 else int((self.support > 0).sum()))}


@dataclass
class GateConfig:
    """Section 18's thresholds. Every one is a STABILITY criterion -- none of
    them is 'make the residual threshold looser until it passes'."""

    window: int = 3
    max_omega_turn_deg: float = 8.0         # theta_omega
    max_speed_change: float = 0.15          # Delta_v, relative
    max_normalized_residual: float = 3.0    # residual / its own sigma
    min_support_jaccard: float = 0.85
    min_observable_rank: int = 3
    require_certified: bool = True
    require_stable_model: bool = True
    max_twist_covariance_trace: float = float("inf")


def angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """Section 18's theta_omega = angle(omega_t, omega_{t-1}), in degrees."""
    na, nb = float(a.norm()), float(b.norm())
    if na < EPS or nb < EPS:
        return float("nan")
    c = float((a @ b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def relative_speed_change(a: torch.Tensor, b: torch.Tensor) -> float:
    """Section 18's Delta_v = | ||v_t|| - ||v_{t-1}|| | / (||v_{t-1}|| + eps)."""
    na, nb = float(a.norm()), float(b.norm())
    return abs(na - nb) / (nb + EPS)


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Support stability. 1.0 means the moving set did not change at all."""
    x = (a > 0).reshape(-1).bool()
    y = (b > 0).reshape(-1).bool()
    if x.numel() != y.numel():
        return 0.0
    union = int((x | y).sum())
    if union == 0:
        return 1.0           # both empty: perfectly stable, if uninformative
    return int((x & y).sum()) / union


class CoherenceGate:
    """A sliding window of :class:`WindowSample` with section 18's decision."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self.samples: deque[WindowSample] = deque(maxlen=max(2, self.cfg.window))
        self.history: list[dict] = []

    def push(self, sample: WindowSample) -> None:
        self.samples.append(sample)

    # -- the individual criteria ------------------------------------------
    def _window_direction_kind(self) -> str:
        """Decide ONCE per window whether rotation or translation carries the
        motion, then monitor that component for every pair.

        Deciding per pair lets the classification itself flip between frames
        when |omega| sits near the threshold, and a flip empties the angle list
        and reports the direction as undefined -- the same instability the
        choice was introduced to remove. A window-level median is stable.
        """
        s = list(self.samples)
        ws = sorted(float(x.omega.norm()) for x in s if x.omega is not None)
        vs = sorted(float(x.v.norm()) for x in s if x.v is not None)
        if not ws and not vs:
            return "none"
        mw = ws[len(ws) // 2] if ws else 0.0
        mv = vs[len(vs) // 2] if vs else 0.0
        if mw > 1e-3 and mw >= 0.05 * mv:
            return "omega"
        return "v" if mv > 1e-9 else "none"

    @staticmethod
    def _direction_of(sample, kind: str):
        if kind == "omega":
            return sample.omega
        if kind == "v":
            return sample.v
        return None

    @staticmethod
    def _direction(sample) -> tuple[torch.Tensor | None, str]:
        """The twist direction that is actually observable for this sample.

        Section 18's theta_omega assumes there IS a rotation to track. For a
        prismatic joint omega is ~0 and its DIRECTION is pure noise: measured on
        a clean synthetic prismatic fit (|omega| = 0.003 against |v| = 0.29, twist
        residual 0.033) the frame-to-frame omega angle swung 70-108 degrees and
        the gate rejected a textbook-correct estimate. So the monitored
        direction is omega-hat only where omega carries the motion, and v-hat
        otherwise -- in both cases the direction of the component that is large
        enough to have one.
        """
        w = sample.omega
        v = sample.v
        nw = float(w.norm()) if w is not None else 0.0
        nv = float(v.norm()) if v is not None else 0.0
        # omega is in rad/s and v in m/s; compare them as the tangential speed
        # omega would produce over a unit extent, which is just |omega|.
        if nw > 1e-3 and nw >= 0.05 * nv:
            return w, "omega"
        if nv > 1e-9:
            return v, "v"
        return None, "none"

    def _pairwise(self) -> dict:
        s = list(self.samples)
        turns, speeds, jacs = [], [], []
        kind = self._window_direction_kind()
        self._direction_kinds = [kind]
        for prev, cur in zip(s[:-1], s[1:]):
            dp = self._direction_of(prev, kind)
            dc = self._direction_of(cur, kind)
            if dp is not None and dc is not None:
                t = angle_between(dp, dc)
                if math.isfinite(t):
                    turns.append(t)
            if prev.v is not None and cur.v is not None:
                speeds.append(relative_speed_change(cur.v, prev.v))
            if prev.support is not None and cur.support is not None:
                jacs.append(jaccard(prev.support, cur.support))
        return {"omega_turn_deg": turns, "speed_change": speeds,
                "support_jaccard": jacs}

    def evaluate(self) -> dict:
        """All of section 18's required quantities, plus the verdict."""
        cfg = self.cfg
        s = list(self.samples)
        if len(s) < cfg.window:
            out = {"ready": False, "reduce": False,
                   "reason": [f"only {len(s)} of {cfg.window} window frames"]}
            self.history.append(out)
            return out

        pw = self._pairwise()
        reasons: list[str] = []
        crit: dict = {}

        # 1. twist direction stability
        turn = max(pw["omega_turn_deg"]) if pw["omega_turn_deg"] else float("nan")
        crit["omega_turn_deg"] = turn
        crit["direction_monitored"] = getattr(self, "_direction_kinds", [])
        ok_turn = math.isfinite(turn) and turn <= cfg.max_omega_turn_deg
        if not ok_turn:
            kinds = set(getattr(self, "_direction_kinds", []) or ["?"])
            reasons.append(
                f"the twist direction ({'/'.join(sorted(kinds))}) is still "
                f"turning ({turn:.2f} deg over the window, limit "
                f"{cfg.max_omega_turn_deg:.2f})"
                if math.isfinite(turn) else
                "the twist axis direction is undefined (||omega|| ~ 0)")

        # 2. speed magnitude stability
        dv = max(pw["speed_change"]) if pw["speed_change"] else float("nan")
        crit["speed_change"] = dv
        ok_speed = math.isfinite(dv) and dv <= cfg.max_speed_change
        if not ok_speed:
            reasons.append(f"the speed is still changing ({dv:.1%} over the "
                           f"window, limit {cfg.max_speed_change:.1%})")

        # 3. residual normalized by its OWN uncertainty, not an absolute number
        norm_res = []
        for x in s:
            if math.isfinite(x.residual) and math.isfinite(x.residual_sigma) \
                    and x.residual_sigma > EPS:
                norm_res.append(x.residual / x.residual_sigma)
        nr = max(norm_res) if norm_res else float("nan")
        crit["normalized_residual"] = nr
        ok_res = math.isfinite(nr) and nr <= cfg.max_normalized_residual
        if not ok_res:
            reasons.append(
                f"the twist residual is large relative to its own uncertainty "
                f"({nr:.2f} sigma, limit {cfg.max_normalized_residual:.2f})"
                if math.isfinite(nr) else
                "no uncertainty-normalized residual was available")

        # 4. support stability
        jac = min(pw["support_jaccard"]) if pw["support_jaccard"] else float("nan")
        crit["support_jaccard"] = jac
        ok_sup = (not math.isfinite(jac)) or jac >= cfg.min_support_jaccard
        if math.isfinite(jac) and not ok_sup:
            reasons.append(f"the support set is still changing (Jaccard "
                           f"{jac:.3f}, need {cfg.min_support_jaccard:.3f})")

        # 5. observability
        rank = min(x.observable_rank for x in s)
        crit["min_observable_rank"] = rank
        ok_rank = rank >= cfg.min_observable_rank
        if not ok_rank:
            reasons.append(f"the twist is not observable enough (rank {rank}, "
                           f"need {cfg.min_observable_rank})")

        # 6. the state update is locally certified
        from estimator.certificate_v2 import USABLE
        statuses = [x.certificate_status for x in s]
        crit["certificate_statuses"] = statuses
        ok_cert = (not cfg.require_certified) or all(
            st in USABLE for st in statuses if st)
        if not ok_cert:
            bad = sorted({st for st in statuses if st and st not in USABLE})
            reasons.append(f"the state update is not locally certified "
                           f"({', '.join(bad)})")

        # 7. model evidence stability
        names = [x.model_name for x in s if x.model_name]
        crit["model_names"] = names
        ok_model = (not cfg.require_stable_model) or (
            len(set(names)) <= 1 if names else False)
        if not ok_model:
            reasons.append(f"the preferred model keeps changing "
                           f"({' -> '.join(names)})")

        # 8. twist covariance
        tr = max(x.twist_covariance_trace for x in s
                 if math.isfinite(x.twist_covariance_trace)) \
            if any(math.isfinite(x.twist_covariance_trace) for x in s) else float("nan")
        crit["twist_covariance_trace"] = tr
        ok_cov = (not math.isfinite(tr)) or tr <= cfg.max_twist_covariance_trace
        if math.isfinite(tr) and not ok_cov:
            reasons.append(f"the fitted twist is imprecise (covariance trace "
                           f"{tr:.3e}, limit {cfg.max_twist_covariance_trace:.3e})")

        reduce_now = all([ok_turn, ok_speed, ok_res, ok_sup, ok_rank,
                          ok_cert, ok_model, ok_cov])
        out = {"ready": True, "reduce": bool(reduce_now), "reason": reasons,
               "criteria": crit,
               "passed": {"omega_turn": ok_turn, "speed": ok_speed,
                          "residual": ok_res, "support": ok_sup,
                          "observable_rank": ok_rank, "certified": ok_cert,
                          "model_stable": ok_model, "covariance": ok_cov},
               "window_frames": [x.frame for x in s]}
        self.history.append(out)
        return out

    def to_dict(self) -> dict:
        return {"config": {k: getattr(self.cfg, k)
                           for k in self.cfg.__dataclass_fields__},
                "window": [s.to_dict() for s in self.samples],
                "history": self.history}
