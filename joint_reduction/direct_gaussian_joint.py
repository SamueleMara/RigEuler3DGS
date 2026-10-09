"""Direct anisotropic-information track fitting; no per-frame rigid compression.

The small transform is differentiated analytically by forward-mode AD. The
renderer is never differentiated numerically or through autograd.
"""
from dataclasses import dataclass, replace

import torch

from .continuous_joint_geometry import continuous_joint_transform, skew, tangent_basis


@dataclass
class JointChart:
    c: torch.Tensor
    d: torch.Tensor
    eta: torch.Tensor
    s: torch.Tensor
    length: float
    fixed_curvature: bool = False
    finite_axis: bool = False

    @property
    def geometry_dim(self):
        return 2 if self.fixed_curvature else 4

    @property
    def dimension(self):
        return self.geometry_dim + len(self.s) - 1

    def basis(self):
        return tangent_basis(self.d, 1, True)[:3, :2]

    def transform(self, frame, local=None):
        # Coordinates are scaled by the canonical length, NOT dataset gates.
        n = self.geometry_dim + int(frame > 0)
        v = torch.zeros(n, device=self.d.device, dtype=torch.float64) if local is None else local
        B = self.basis()
        d = self.d + B @ v[:2]
        d = d / d.norm()
        eta = self.eta if self.fixed_curvature else self.eta + B @ v[2:4] / self.length
        eta = eta - d * (d @ eta)
        st = self.s[frame] + (v[-1] * self.length if frame > 0 else 0.)
        if self.finite_axis:
            # Minimal finite-axis branch for a line through the centroid.
            from estimator.manifold import so3_exp
            R = so3_exp((st * d).reshape(1, 3))[0]
            a = self.c + eta * self.length ** 2
            return R, a - R @ a
        R, p = continuous_joint_transform(self.c, d, eta, st.reshape(1))
        return R[0], p[0]

    def indices(self, frame):
        ids = list(range(self.geometry_dim))
        if frame > 0:
            ids.append(self.geometry_dim + frame - 1)
        return torch.tensor(ids, device=self.d.device, dtype=torch.long)

    def retract(self, step):
        B = self.basis()
        d = self.d + B @ step[:2]
        d = d / d.norm()
        eta = torch.zeros_like(self.eta) if self.fixed_curvature else self.eta + B @ step[2:4] / self.length
        eta = eta - d * (d @ eta)
        s = self.s.clone()
        s[1:] += step[self.geometry_dim:] * self.length
        return replace(self, d=d.detach(), eta=eta.detach(), s=s.detach())

    def pose_jacobian(self, x0, frame):
        """[N,6,m]: centre derivative and LEFT rotation-vector derivative."""
        with torch.enable_grad():
            z = torch.zeros(self.geometry_dim + int(frame > 0), device=self.d.device, dtype=torch.float64)
            derivative = torch.func.jacfwd(lambda v: torch.cat([
                self.transform(frame, v)[0].reshape(-1), self.transform(frame, v)[1]]))(z)
        R, p = self.transform(frame)
        dR = derivative[:9].reshape(3, 3, -1)
        dp = derivative[9:]
        gx = torch.einsum('ijm,nj->nim', dR, x0.double()) + dp[None]
        W = torch.einsum('ijm,kj->ikm', dR, R)
        omega = torch.stack([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) * .5
        return torch.cat([gx, omega[None].expand(len(x0), -1, -1)], dim=1).detach()

    def summary(self):
        from .continuous_joint_geometry import continuous_to_finite_axis
        f = None if self.finite_axis else continuous_to_finite_axis(self.c, self.d, self.eta, self.s, kappa_floor=0.)
        if self.finite_axis:
            a = self.c + self.eta * self.length ** 2
            axis, point, q = self.d, a - self.d * (self.d @ a), self.s
        elif f is not None:
            axis, point, q = f['axis'], f['axis_point_origin'], f['q']
        else:
            axis, point, q = self.d, None, self.s
        return {'d': self.d, 'eta': self.eta, 's': self.s, 'c': self.c,
                'kappa': float(self.eta.norm()) if not self.finite_axis else None,
                'axis': axis, 'axis_point': point, 'q': q,
                'length_scale': self.length, 'finite_axis_fallback': self.finite_axis}


def solve_weighted_twist(x0, velocity, information):
    """Closed-form [u,omega] solve, centred for world-frame conditioning."""
    x0, velocity, information = x0.double(), velocity.double(), information.double()
    c = x0.mean(0)
    Ai = torch.cat([torch.eye(3, device=x0.device).double().expand(len(x0), -1, -1), -skew(x0 - c)], dim=2)
    A = torch.einsum('nij,nik,nkl->jl', Ai, information, Ai)
    b = torch.einsum('nij,nik,nk->j', Ai, information, velocity)
    from .joint_information import information_inverse
    inv = information_inverse(A)
    centred = inv['inverse'] @ b
    xi = centred.clone()
    xi[:3] -= torch.linalg.cross(xi[3:], c)
    M = torch.eye(6, device=x0.device, dtype=torch.float64)
    M[:3, 3:] = skew(c)
    return {'xi': xi, 'centred_xi': centred, 'information': A,
            'covariance': M @ inv['inverse'] @ M.T, 'rank': inv['rank'], 'centre': c}


def initialise_one_dof_screw(x0, tracks, information, dt=1.):
    """Information-weighted rank-one twist factorization; initializer only."""
    fits = [solve_weighted_twist(x0, (tracks[t] - tracks[t-1]) / dt, information[t])
            for t in range(1, len(tracks))]
    if not fits:
        raise ValueError('at least two frames are required')
    X = torch.stack([f['centred_xi'] for f in fits])
    F = torch.stack([f['information'] for f in fits])
    L = float((x0 - x0.mean(0)).square().sum(1).mean().sqrt())
    S = torch.diag(torch.tensor([1., 1., 1., 1/max(L, 1e-300), 1/max(L, 1e-300), 1/max(L, 1e-300)], device=x0.device).double())
    y = torch.linalg.svd(X @ torch.linalg.inv(S), full_matrices=False).Vh[0]
    direction = S @ y
    from .joint_information import information_inverse
    for _ in range(12):
        rates = torch.einsum('i,tij,tj->t', direction, F, X) / torch.einsum('i,tij,j->t', direction, F, direction).clamp_min(1e-300)
        A = torch.einsum('t,tij->ij', rates.square(), F)
        b = torch.einsum('t,tij,tj->i', rates, F, X)
        direction = information_inverse(A)['inverse'] @ b
        direction /= (torch.linalg.inv(S) @ direction).norm().clamp_min(1e-300)
    centre_velocity = direction[:3]
    speed = centre_velocity.norm()
    if float(direction.norm()) == 0.:
        # No measurement-supported twist: keep a valid numerical chart and let
        # the information rank report the unresolved geometry.
        direction[0] = 1.
        centre_velocity, speed = direction[:3], direction[:3].norm()
    # Exact numerical degeneracy uses the finite-axis branch, never a P/R gate.
    finite = float(speed) <= torch.finfo(torch.float64).eps * float(direction.norm())
    if finite:
        d = direction[3:] / direction[3:].norm().clamp_min(1e-300)
        eta = torch.zeros_like(d)
        rates *= direction[3:].norm()
        s = torch.cat([rates.new_zeros(1), torch.cumsum(rates * dt, 0)])
    else:
        d = centre_velocity / speed
        eta = direction[3:] / speed
        eta -= d * (d @ eta)
        s = torch.cat([rates.new_zeros(1), torch.cumsum(rates * speed * dt, 0)])
    chart = JointChart(x0.mean(0).double(), d, eta, s, max(L, 1e-300), finite_axis=finite)
    return chart, {'xi_t': torch.stack([f['xi'] for f in fits]),
                   'twist_covariance': torch.stack([f['covariance'] for f in fits]),
                   'twist_information': F, 'xi0_centred': direction, 'rates': rates}


def optimise_joint(initial, linearize, energy, max_iters=40, precision=torch.float64):
    """Shared minimal-coordinate GN/LM with actual/predicted reduction."""
    chart = initial
    E = float(energy(chart))
    history = []
    lam = 0.
    status = 'MAX_ITERS'
    for _ in range(max_iters):
        A, g, _details = linearize(chart)
        from .joint_information import information_inverse
        spec = information_inverse(A)
        diagonal = A.diagonal().clamp_min(torch.finfo(torch.float64).tiny)
        eps = torch.finfo(precision).eps
        if float(g.norm()) <= eps ** .5 * max(float(A.norm()) ** .5 * max(E, 0.) ** .5, torch.finfo(torch.float64).tiny):
            status = 'CONVERGED_STATIONARY'
            break
        accepted = False
        for trial in range(16):
            if trial == 0 and spec['rank'] == chart.dimension:
                step = spec['inverse'] @ g
            else:
                lam = max(lam, eps ** .5)
                step = torch.linalg.solve(A + lam * torch.diag(diagonal), g)
            predicted = float(2 * (g @ step) - step @ A @ step)
            candidate = chart.retract(step)
            En = float(energy(candidate))
            actual = E - En
            rho = actual / predicted if predicted > 0 else float('-inf')
            if actual > 8 * torch.finfo(torch.float64).eps * max(abs(E), abs(En), torch.finfo(torch.float64).tiny) and rho > .1:
                chart, E, accepted = candidate, En, True
                lam /= 3
                history.append({'energy': E, 'rho': rho, 'step_norm': float(step.norm()), 'lambda': lam})
                break
            lam = max(lam * 10, eps ** .5)
        if not accepted:
            # A tiny heavily damped trial is not evidence of convergence.
            optimal = spec['inverse'] @ g
            precision_limited = (float(optimal.norm()) <= eps ** .5 or
                float(g @ optimal) <= eps * max(E, torch.finfo(torch.float64).tiny))
            status = 'CONVERGED_AT_PRECISION' if precision_limited else 'STALLED'
            break
    A, g, details = linearize(chart)
    return {'chart': chart, **chart.summary(), 'energy': E, 'status': status,
            'converged': status.startswith('CONVERGED'), 'history': history,
            'information': A, 'score': g, 'details': details,
            'orientation_renderer_jacobian': False}


def fit_joint_from_gaussian_tracks(x0, tracks, information, *, initial=None,
                                   fixed_curvature=False, dt=1., max_iters=40):
    """Full 3x3 image information weights; no track-isotropic replacement."""
    x0, tracks, information = x0.double(), tracks.double(), information.double()
    if tracks.shape != (len(tracks), len(x0), 3) or information.shape != tracks.shape[:2] + (3, 3):
        raise ValueError('tracks [T,N,3] and information [T,N,3,3] are required')
    init_info = None
    if initial is None:
        initial, init_info = initialise_one_dof_screw(x0, tracks, information, dt)
    chart = replace(initial, fixed_curvature=fixed_curvature,
                    eta=torch.zeros_like(initial.eta) if fixed_curvature else initial.eta,
                    finite_axis=False if fixed_curvature else initial.finite_axis)

    def system(q):
        A = x0.new_zeros(q.dimension, q.dimension)
        g = x0.new_zeros(q.dimension)
        blocks = []
        for t in range(1, len(tracks)):
            R, p = q.transform(t)
            r = tracks[t] - (x0 @ R.T + p)
            G = q.pose_jacobian(x0, t)[:, :3]
            H = torch.einsum('nij,nik,nkl->jl', G, information[t], G)
            b = torch.einsum('nij,nik,nk->j', G, information[t], r)
            ix = q.indices(t)
            A[ix[:, None], ix[None, :]] += H
            g[ix] += b
            score = g.new_zeros(q.dimension); score[ix] = b
            blocks.append({'frame': t, 'score': score})
        return A, g, {'blocks': blocks}

    def energy(q):
        out = 0.
        for t in range(1, len(tracks)):
            R, p = q.transform(t)
            r = tracks[t] - (x0 @ R.T + p)
            out += float(torch.einsum('ni,nij,nj->', r, information[t], r))
        return out
    fit = optimise_joint(chart, system, energy, max_iters)
    fit['initializer'] = init_info
    return fit


# ===========================================================================
# Sections 7, 33, 35: the dual differential + finite chain.
#
# Section 33 says Branch B must stop jumping from tracks straight to a
# nonlinear P/R fit, and instead walk the physical chain
#
#   tracks -> velocities -> affine G -> D + Omega -> rigid twist xi_t
#          -> finite centred R_t -> differential/finite consistency
#          -> common one-DoF screw -> P/R candidate.
#
# Everything above the last arrow is measurement; the P/R question is asked
# only once the support has been shown to be rigid and one-degree-of-freedom.
# Section 38 is the reason: "did a more flexible model reduce one objective
# past a floor?" is not the same question as "is this a hinge?", and only the
# second is answerable from geometry.
# ===========================================================================
MODEL_MISMATCH = "MODEL_MISMATCH"
UNRESOLVED_GEOMETRY = "UNRESOLVED_GEOMETRY"
PRISMATIC_AT_RESOLUTION = "PRISMATIC_AT_RESOLUTION"
REVOLUTE = "REVOLUTE"


def dual_differential_finite_evidence(x0, tracks, *, information=None,
                                      weights=None, times=None, dt=1.0,
                                      edges=None, velocity=None):
    """Sections 7-26: every measurement the hierarchy of section 35 needs.

    Returns evidence, never a verdict. `x0` is the canonical support, `tracks`
    is [T, N, 3], and `information` is the per-frame per-point 3x3 position
    information from Phase 1 (section 9 forbids flattening it to a scalar when
    the directional form exists).
    """
    from .common_screw import fit_weighted_common_screw
    from .differential_finite_consistency import (attribute_angular_origin,
                                                  compare_angular_velocity,
                                                  compare_screw_trajectory,
                                                  incremental_spatial_omega)
    from .finite_relative_rotation import finite_rotation_sequence
    from .rigid_twist_from_tracks import twist_sequence
    from .velocity_gradient_rigidity import (decompose_velocity_gradient,
                                             differential_edge_rigidity,
                                             fit_affine_velocity_field,
                                             instantaneous_velocity,
                                             rigidity_likelihood_ratio,
                                             strain_covariance,
                                             strain_statistic)

    x0 = torch.as_tensor(x0).double()
    X = torch.as_tensor(tracks).double()
    T = int(X.shape[0])
    L = float((x0 - x0.mean(0)).norm(dim=1).square().mean().sqrt().clamp_min(1e-12))

    vel = instantaneous_velocity(X, times=times, dt=dt, velocity=velocity)
    V = vel["velocity"]

    # ---- the differential half (sections 10-14) -------------------------
    affine, strain, qd_stats = [], [], []
    for t in range(T):
        info_t = None if information is None else information[t]
        f = fit_affine_velocity_field(X[t], V[t], information=info_t,
                                      weights=weights)
        dec = decompose_velocity_gradient(f["G"])
        cov = strain_covariance(f)
        st = strain_statistic(dec["D"], cov["covariance"])
        affine.append({**f, **dec})
        strain.append(st)
        qd_stats.append(st["Q_D"])
    strain_stat = torch.tensor(qd_stats, dtype=torch.float64)

    edge_rigidity = None
    if edges is not None:
        edge_rigidity = [
            differential_edge_rigidity(X[t], V[t], edges) for t in range(T)]

    # ---- the exact rigid twist (sections 16-19) -------------------------
    ts = twist_sequence(X, times=times, dt=dt, information=information,
                        weights=weights, velocity=velocity)
    omega_info = torch.stack([f["omega_information"] for f in ts["fits"]])
    rigid_error = torch.tensor([f["rigid_fit_error"] for f in ts["fits"]],
                               dtype=torch.float64)
    # Section 35 A: the statistic with a known null distribution. The affine
    # model supplies the noise scale, so this is not the rigid residual being
    # judged against itself.
    nested = [rigidity_likelihood_ratio(affine[t], ts["fits"][t])
              for t in range(T)]
    nested_chi2 = torch.tensor([r["chi2_rigid_vs_affine"] for r in nested],
                               dtype=torch.float64)

    # ---- the finite half (sections 22-24) -------------------------------
    fin = finite_rotation_sequence(X, weights=weights)

    # ---- do the two halves agree? (sections 26, 39) ----------------------
    inc = incremental_spatial_omega(fin["R"], times=times, dt=dt)
    # omega^diff is stamped at frames, omega^finite at interval midpoints, so
    # the differential estimates are averaged onto the same stamps before the
    # comparison rather than silently paired off by index.
    om_diff_mid = 0.5 * (ts["omega"][1:] + ts["omega"][:-1])
    om_cov = torch.stack([f["omega_covariance"] for f in ts["fits"]])
    om_cov_mid = 0.25 * (om_cov[1:] + om_cov[:-1])
    # omega^finite = log(R_{t+1} R_t^T)/dt, so its covariance is that of the
    # two Kabsch rotations it was built from, divided by dt^2.
    fin_cov = fin["rotation_covariance"]
    step = max(float(dt), 1e-300) ** 2
    fin_cov_mid = (fin_cov[1:] + fin_cov[:-1]) / step
    cmp = compare_angular_velocity(om_diff_mid, inc["omega"],
                                   covariance_diff=om_cov_mid,
                                   covariance_finite=fin_cov_mid)
    attribution = attribute_angular_origin(
        cmp, strain_statistic=strain_stat,
        omega_information=0.5 * (omega_info[1:] + omega_info[:-1]))

    # ---- one common screw? (sections 18-21) ------------------------------
    screw = fit_weighted_common_screw(ts["xi_spatial"], length_scale=L,
                                      omega=ts["omega"],
                                      omega_information=omega_info)

    return {"length_scale": L, "num_frames": T, "num_points": int(x0.shape[0]),
            "velocity": vel, "affine": affine, "strain": strain,
            "strain_statistic": strain_stat,
            "edge_rigidity": edge_rigidity,
            "twist": ts, "omega_information": omega_info,
            "rigid_twist_error": rigid_error,
            "rigidity_nested": nested, "rigidity_chi2": nested_chi2,
            "finite": fin, "incremental_omega": inc,
            "angular_comparison": cmp, "angular_attribution": attribution,
            "screw": screw}


def _chart_from_screw(x0, evidence, kind, dt=1.0):
    """Turn a common screw into the continuous (c, d, eta, s) chart.

    The chart is what Branch C consumes, so the geometric candidate has to be
    expressible in it. eta = 0 IS the prismatic chart, which is why the two
    candidates differ only by that constraint and not by a change of
    parameterisation.
    """
    from .common_screw import estimate_qdot, integrate_q

    c = x0.double().mean(0)
    L = evidence["length_scale"]
    xi = evidence["twist"]["xi_spatial"]
    screw = evidence["screw"]
    gen = screw["prismatic"] if kind == "prismatic" else screw["revolute"]
    xi_J = gen["xi_J"] if kind == "prismatic" else screw["twist_rank"]["xi_J"]
    rates = estimate_qdot(xi, xi_J, evidence["twist"]["information"])
    q = integrate_q(rates, dt=dt)

    if kind == "prismatic":
        d = gen["direction_of_travel"]
        eta = torch.zeros(3, dtype=torch.float64, device=d.device)
        s = q * 1.0
        return JointChart(c, d, eta, s - s[0], L, fixed_curvature=True)

    v0, om = xi_J[:3], xi_J[3:]
    # velocity of the support centre under this screw
    vc = v0 + torch.linalg.cross(om, c)
    speed = vc.norm()
    if float(speed) <= 0.0:
        d = om / om.norm().clamp_min(1e-300)
        eta = torch.zeros_like(d)
        s = q * float(om.norm())
        return JointChart(c, d, eta, s - s[0], L, finite_axis=True)
    d = vc / speed
    eta = om / speed
    eta = eta - d * (d @ eta)
    s = q * float(speed)
    return JointChart(c, d, eta, s - s[0], L)


def geometric_joint_candidate(evidence, *, confidence: float = 0.95,
                              mismatch_confidence: float = 1.0 - 1e-6,
                              x0=None, dt: float = 1.0) -> dict:
    """Section 35 A-C: rigid? one-DoF? then and only then, P or R.

    The order is the whole point. Asking "P or R" of a support that is not
    rigid, or whose motion is not one degree of freedom, produces a confident
    answer to a question the data cannot support -- which is how a prismatic
    drawer acquires a revolute axis. Each stage can stop the chain with
    MODEL_MISMATCH or UNRESOLVED_GEOMETRY instead.

    Every cut is a chi-square quantile at one `confidence` shared by the whole
    run (section 62: no dataset-specific logic), never a magnitude threshold.
    """
    from .common_screw import chi2_quantile

    screw = evidence["screw"]
    ang = screw["angular_rank"]
    full = screw["twist_rank"]
    T = evidence["num_frames"]

    # ---- A. rigid compatibility (section 35 A) ---------------------------
    # The nested rigid-versus-affine chi-square is the statistic that decides
    # this stage, because its null distribution is known: the six parameters
    # separating the two models ARE vech(D). Q_D is carried alongside as
    # section 13's reported diagnostic.
    chi = evidence["rigidity_chi2"]
    qd = evidence["strain_statistic"]
    # Section 35 A says MODEL_MISMATCH when the one-rigid-body model "CLEARLY
    # fails", and the word carries weight: this verdict halts the analysis and
    # asserts that no joint type applies at all, which is a stronger claim
    # than choosing between two joint types. It therefore takes stronger
    # evidence, and `mismatch_confidence` is set far tighter than the P/R
    # confidence for that reason.
    #
    # The asymmetry is also what keeps the test from becoming a hair trigger.
    # A chi-square's power grows with the number of supernodes, so at a 0.05
    # cut a large well-behaved support is rejected for a departure far too
    # small to matter -- which is the mirror image of the magnitude thresholds
    # sections 6, 18 and 25 forbid. One value is used for every scene.
    strain_cut = chi2_quantile(6, mismatch_confidence)
    frac = float((chi > strain_cut).double().mean()) if chi.numel() else 0.0
    rigid_ok = frac <= 0.5
    stage_a = {"median_chi2_rigid_vs_affine":
                   float(chi.median()) if chi.numel() else 0.0,
               "cut": strain_cut, "mismatch_confidence": mismatch_confidence,
               "fraction_above_cut": frac,
               "passes": bool(rigid_ok),
               "median_Q_D": float(qd.median()) if qd.numel() else 0.0,
               "median_rigid_twist_error":
                   float(evidence["rigid_twist_error"].median())
                   if evidence["rigid_twist_error"].numel() else 0.0,
               "statistic": ("section 35 A: chi2(6) comparing the rigid fit "
                             "with the affine fit that contains it")}

    # ---- B. one-DoF compatibility (section 35 B) -------------------------
    dof1 = max(full["dof_rank1"], 1)
    one_dof_cut = chi2_quantile(dof1, confidence)
    one_dof_ok = full["chi2_rank1"] <= one_dof_cut
    stage_b = {"chi2_rank1": full["chi2_rank1"], "dof": dof1,
               "cut": one_dof_cut, "passes": bool(one_dof_ok),
               "per_dof": full["rank1_per_dof"]}

    # ---- C. P/R geometric candidate (section 35 C) -----------------------
    # H_P is omega = 0 (section 6). The angular rank-0 chi-square IS that
    # hypothesis test, taken against the information that survives after
    # translation is eliminated (section 17).
    ang_dof = max(ang["dof_rank0"], 1)
    ang_cut = chi2_quantile(ang_dof, confidence)
    angular_present = ang["chi2_rank0"] > ang_cut
    ang_rank1_cut = chi2_quantile(max(ang["dof_rank1"], 1), confidence)
    angular_one_axis = ang["chi2_rank1"] <= ang_rank1_cut
    stage_c = {"chi2_omega_zero": ang["chi2_rank0"], "dof": ang_dof,
               "cut": ang_cut, "angular_present": bool(angular_present),
               "chi2_one_axis": ang["chi2_rank1"],
               "one_axis_cut": ang_rank1_cut,
               "angular_shares_one_axis": bool(angular_one_axis),
               "axis": ang["axis"].tolist()}

    attribution = evidence["angular_attribution"]["attribution"]

    if not rigid_ok:
        candidate, status = None, MODEL_MISMATCH
        why = ("section 35 A: the differential strain statistic exceeds its "
               "own chi-square cut on most frames, so this support does not "
               "move as one rigid body and no joint type applies to it")
    elif not one_dof_ok:
        candidate, status = None, MODEL_MISMATCH
        why = ("section 35 B: the twist sequence is not rank 1, so the motion "
               "is not generated by a single degree of freedom")
    elif attribution == "ANGULAR_INTRODUCED_DOWNSTREAM":
        candidate, status = None, UNRESOLVED_GEOMETRY
        why = ("section 39: the angular motion is absent from the Phase-1 "
               "velocities and present only in the finite rotations, so it "
               "cannot be attributed to the joint")
    elif attribution == "ANGULAR_INCONSISTENT":
        candidate, status = None, UNRESOLVED_GEOMETRY
        why = ("section 26: the differential and finite angular velocities do "
               "not share an axis, so the motion model is not trustworthy")
    elif angular_present and angular_one_axis:
        candidate, status = "revolute", REVOLUTE
        why = ("section 35 C: omega = 0 is rejected at the stated confidence "
               "and every frame's angular velocity shares one axis")
    elif angular_present:
        candidate, status = None, UNRESOLVED_GEOMETRY
        why = ("section 18: angular motion is present but does not lie on one "
               "fixed axis, so it is not a one-DoF revolute joint")
    else:
        candidate, status = "prismatic", PRISMATIC_AT_RESOLUTION
        why = ("section 35 C: omega = 0 is not rejected at the stated "
               "confidence, so the geometry is prismatic AT THIS RESOLUTION "
               "-- which is a statement about what was measurable, not a "
               "claim that the axis is exactly at infinity")

    out = {"candidate": candidate, "status": status, "reason": why,
           "confidence": confidence,
           "mismatch_confidence": mismatch_confidence,
           "stage_a_rigid": stage_a, "stage_b_one_dof": stage_b,
           "stage_c_angular": stage_c,
           "angular_attribution": evidence["angular_attribution"],
           "decision_inputs": (
               "sections 35, 38: finite pairwise rigidity, differential "
               "strain, rigid-twist residual, differential/finite angular "
               "consistency and one-DoF screw structure; no gain-versus-floor "
               "comparison and no magnitude threshold")}
    if x0 is not None and status in (REVOLUTE, PRISMATIC_AT_RESOLUTION):
        kind = "revolute" if candidate == "revolute" else "prismatic"
        out["chart"] = _chart_from_screw(torch.as_tensor(x0).double(),
                                         evidence, kind, dt)
        out["chart_prismatic"] = _chart_from_screw(
            torch.as_tensor(x0).double(), evidence, "prismatic", dt)
        out["chart_revolute"] = _chart_from_screw(
            torch.as_tensor(x0).double(), evidence, "revolute", dt)
    return out
