"""Joint geometry + trajectory GN using analytic CUDA rasterizer JVP/VJP.

Only the 12-number SE(3) kinematic map uses forward AD. No pixel/Gaussian
Jacobian is materialized and no renderer finite difference runs in inference.
"""
from dataclasses import replace

import torch

from estimator.measurement import build_precision
from estimator.renderer_bridge import FrameLinearization, build_linearization
from .direct_gaussian_joint import optimise_joint
from .joint_information import (accumulate_block_information, accumulate_camera_information,
                                accumulate_gaussian_information, alpha_responsibility,
                                gaussian_joint_influence, information_inverse)
from .joint_uncertainty import cluster_robust_covariance, curvature_covariance


def measurement_terms(view, observation, cfg, model_error=None, camera=0):
    """Consistent Student-t likelihood/IRLS precision with physical R_eff."""
    if observation is None:
        zc, zd = torch.zeros_like(view.rgb), torch.zeros_like(view.depth)
        return (zc, zd), (zc, zd), (0., 0.), (zc, zd)
    rgb, depth, valid = [v.to(view.rgb.device) for v in observation]
    rr, rd = rgb - view.rgb, depth - view.depth
    rgb_valid = torch.isfinite(rr) & bool(cfg.use_rgb)
    depth_valid = valid & torch.isfinite(depth) & torch.isfinite(rd) & (view.alpha > cfg.min_alpha_for_depth) & bool(cfg.use_depth)
    rr = torch.where(rgb_valid, rr, 0.)
    rd = torch.where(depth_valid, rd, 0.)
    pr = build_precision(rr, sigma_rgb=cfg.rgb_sigma, valid_mask=rgb_valid,
                         robust=False, layout=('rgb',)*3, channel_dim=0)
    depth_model = cfg.depth_sigma_model if ':' in str(cfg.depth_sigma_model) else {'kind':cfg.depth_sigma_model, 'a':cfg.depth_sigma_a, 'b':cfg.depth_sigma_b}
    pd = build_precision(rd[None], valid_mask=depth_valid[None], sigma_rgb=(),
                         robust=False, depth_sigma_model=depth_model,
                         depth_range=torch.where(depth_valid, depth, 0.)[None], layout=('depth',), channel_dim=0)[0]
    if model_error is not None:
        pr = model_error.effective_rgb_precision(camera, cfg.rgb_sigma) * rgb_valid
        sensor_var = torch.where(pd > 0, pd.clamp_min(torch.finfo(pd.dtype).tiny).reciprocal(), 0.)
        pd = model_error.effective_depth_precision(camera, sensor_var) * depth_valid
    weights, energies, precisions = [], [], []
    for r, p in ((rr, pr), (rd, pd)):
        u2 = r.double().square() * p.double()
        if cfg.robust_measurement:
            nu = cfg.student_t_nu
            w = ((nu+1)/(nu+u2)).to(p.dtype)
            e = float(((nu+1)*torch.log1p(u2/nu)).sum())
        else:
            w = torch.ones_like(p); e = float(u2.sum())
        weights.append(w); energies.append(e); precisions.append(p*w)
    return tuple(precisions), (rr, rd), tuple(energies), tuple(weights)


def build_joint_renderer_jacobian(linearization, pose_jacobian):
    """H G, retaining orientation/covariance derivatives in the six pose DOFs."""
    return [linearization.apply_H(pose_jacobian[:, :, k].float().contiguous())
            for k in range(pose_jacobian.shape[-1])]


class RGBDJointProblem:
    def __init__(self, objective, member, *, base_states=None, camera_ids=None,
                 block_frames=10):
        self.objective = objective
        self.member = member.bool()
        self.base_states = base_states
        self.camera_ids = list(range(len(objective.cameras))) if camera_ids is None else list(camera_ids)
        self.camera_labels = getattr(objective, 'camera_labels', list(range(len(objective.cameras))))
        self.block_frames = block_frames
        self.weight_writer = None

    def state(self, chart, frame):
        obj = self.objective
        mu, R = (obj.mu0, obj.R0) if self.base_states is None else self.base_states[frame]
        mu, R = mu.clone(), R.clone()
        Rt, p = chart.transform(frame)
        mu[self.member] = (obj.mu0[self.member].double() @ Rt.T + p).to(mu.dtype)
        R[self.member] = (Rt[None] @ obj.R0[self.member].double()).to(R.dtype)
        return obj._state(mu, R)

    @torch.no_grad()
    def evaluate(self, chart, *, raw=False):
        obj = self.objective
        rows = []
        for t, (obs, fid) in enumerate(obj.frames):
            state = self.state(chart, t)
            for ci in self.camera_ids:
                lin = build_linearization(state, [obj.cameras[ci]], eps_depth=obj.cfg.eps_depth)
                prec, resid, energies, _w = measurement_terms(lin.views[0], obs[ci], obj.cfg, obj.model_error, ci)
                if raw:
                    # Raw observed RGB/depth residual, physically whitened, no robust reweighting.
                    from copy import copy
                    cfg = copy(obj.cfg); cfg.robust_measurement = False
                    _p, _r, energies, _w = measurement_terms(lin.views[0], obs[ci], cfg, obj.model_error, ci)
                rows.append({'frame':t, 'frame_id':fid, 'camera':self.camera_labels[ci], 'rgb':energies[0], 'depth':energies[1],
                             'loss':sum(energies), 'valid_rgb':int((prec[0]>0).sum()), 'valid_depth':int((prec[1]>0).sum())})
                del lin
        return rows

    @torch.no_grad()
    def linearize(self, chart, diagnostics=False):
        obj = self.objective
        A = torch.zeros(chart.dimension, chart.dimension, device=obj.mu0.device, dtype=torch.float64)
        g = A.new_zeros(chart.dimension)
        rows, gaussian = [], []
        for t, (obs, fid) in enumerate(obj.frames):
            state = self.state(chart, t)
            ownG = chart.pose_jacobian(obj.mu0[self.member], t)
            G = ownG.new_zeros(len(obj.mu0), 6, ownG.shape[-1]); G[self.member] = ownG
            ix = chart.indices(t)
            for ci in self.camera_ids:
                lin = build_linearization(state, [obj.cameras[ci]], eps_depth=obj.cfg.eps_depth)
                view = lin.views[0]
                precision, residual, energies, robust_weights = measurement_terms(view, obs[ci], obj.cfg, obj.model_error, ci)
                columns = build_joint_renderer_jacobian(lin, G)
                # Only [pixels x local joint DOFs], never [pixels x Gaussians].
                C = torch.stack([torch.cat([v[0][0].reshape(-1), v[0][1].reshape(-1)]) for v in columns], 1).double()
                p = torch.cat([precision[0].reshape(-1), precision[1].reshape(-1)]).double()
                r = torch.cat([residual[0].reshape(-1), residual[1].reshape(-1)]).double()
                h, b = C.T @ (p[:, None] * C), C.T @ (p * r)
                Ac, gc = torch.zeros_like(A), torch.zeros_like(g)
                Ac[ix[:, None], ix[None, :]] = h; gc[ix] = b
                A += Ac; g += gc
                rows.append({'frame':t, 'frame_id':fid, 'camera':self.camera_labels[ci], 'A':Ac, 'score':gc,
                             'rgb':energies[0], 'depth':energies[1]})
                if diagnostics:
                    if self.weight_writer is not None:
                        self.weight_writer(t,self.camera_labels[ci],robust_weights)
                    info = accumulate_gaussian_information(lin, [precision], [residual])['cameras'][0]
                    gaussian.append({'frame':t, 'camera':self.camera_labels[ci], 'indices':ix,
                                     'G':ownG.cpu(), 'F_pose':info['F_pose'][self.member].cpu(),
                                     'b_pose':info['b_pose'][self.member].cpu(),
                                     'alpha_mass':alpha_responsibility(state, obj.cameras[ci])[self.member].cpu(),
                                     'robust_weight_rgb_summary':[float(robust_weights[0].min()),float(robust_weights[0].mean()),float(robust_weights[0].max())],
                                     'robust_weight_depth_summary':[float(robust_weights[1].min()),float(robust_weights[1].mean()),float(robust_weights[1].max())]})
                del lin, columns, C
        return A, g, {'camera_rows': rows,
                      'cameras': accumulate_camera_information(rows, chart.dimension, A.device),
                      'blocks': accumulate_block_information(rows, self.block_frames), 'gaussian':gaussian}


def fit_joint_from_rgbd(problem, initial, *, fixed_curvature=False, max_iters=30,
                        diagnostics=True):
    chart = replace(initial, fixed_curvature=fixed_curvature,
                    eta=torch.zeros_like(initial.eta) if fixed_curvature else initial.eta,
                    finite_axis=False if fixed_curvature else initial.finite_axis)
    fit = optimise_joint(chart, problem.linearize,
                         lambda q: sum(r['loss'] for r in problem.evaluate(q)),
                         max_iters, precision=torch.float32)
    A, g, details = problem.linearize(fit['chart'], diagnostics=diagnostics)
    fit.update(information=A, score=g, details=details, orientation_renderer_jacobian=True)
    blocks = details['blocks']
    uncertainty = cluster_robust_covariance(A, torch.stack([b['score'] for b in blocks]))
    fit['uncertainty'] = uncertainty
    fit['curvature'] = curvature_covariance(fit['chart'], uncertainty['covariance'], A)
    inv = information_inverse(A)['inverse']
    fit['camera_influence'] = [{'camera': c['camera'], 'delta': -inv @ c['score'],
                               'proposal':information_inverse(c['A'])['inverse'] @ c['score'],
                               'predicted_gain':float(c['score'] @ information_inverse(c['A'])['inverse'] @ c['score'])}
                              for c in details['cameras']]
    fit['block_influence'] = [{'camera':b['camera'], 'window':b['window'], 'delta':-inv @ b['score']}
                             for b in blocks]
    return fit


def evaluate_rgbd_joint_candidate(problem, chart):
    return problem.evaluate(chart, raw=True)


def heldout_camera_validation(problem, pfit, rfit, *, folds=2, max_iters=30):
    """Fast diagnostic folds; Phase-2 support/init kept, held-out q frozen."""
    comparisons, fits = [], []
    ids = problem.camera_ids
    if len(ids) < 2:
        return comparisons, fits
    for fold in range(min(folds, len(ids))):
        test = [c for j,c in enumerate(ids) if j % min(folds,len(ids)) == fold]
        train = [c for c in ids if c not in test]
        training = RGBDJointProblem(problem.objective, problem.member, base_states=problem.base_states,
                                    camera_ids=train, block_frames=problem.block_frames)
        fp = fit_joint_from_rgbd(training, pfit['chart'], fixed_curvature=True, max_iters=max_iters, diagnostics=False)
        fr = fit_joint_from_rgbd(training, rfit['chart'], max_iters=max_iters, diagnostics=False)
        heldout = RGBDJointProblem(problem.objective, problem.member, base_states=problem.base_states,
                                   camera_ids=test, block_frames=problem.block_frames)
        comparisons += compare_candidates(heldout, fp['chart'], fr['chart'], fold, 'heldout_fast')
        fits.append({'fold':fold, 'train_cameras':train, 'heldout_cameras':test,
                     'P':fp, 'R':fr, 'phase12_recomputed':False,
                     'converged':fp['converged'] and fr['converged']})
    return comparisons, fits


def compare_candidates(problem, pchart, rchart, fold=-1, role='train'):
    P, R = evaluate_rgbd_joint_candidate(problem, pchart), evaluate_rgbd_joint_candidate(problem, rchart)
    rows = []
    for p,r in zip(P,R):
        rows.append({'fold':fold, 'frame':p['frame'], 'camera':p['camera'], 'train_or_heldout':role,
                     'E_P_rgb':p['rgb'], 'E_R_rgb':r['rgb'], 'E_P_depth':p['depth'], 'E_R_depth':r['depth'],
                     'gain_rgb':p['rgb']-r['rgb'], 'gain_depth':p['depth']-r['depth'],
                     'gain_total':p['loss']-r['loss'], 'valid_rgb':p['valid_rgb'], 'valid_depth':p['valid_depth']})
    return rows


# ===========================================================================
# Sections 34, 35 D, 37: the renderer as the FINAL observational confirmation.
#
# Section 34 is a change of role, not of machinery. Branch C no longer
# competes with a transform-only fit to decide the joint type; it receives the
# physically meaningful candidate that the differential and finite analysis
# produced and asks one question of the original images: do they support the
# angular component, or are they equally well explained without it?
#
# Section 36 explains why the geometry cannot answer that alone. Even purely
# prismatic data produces a small noisy angular block, so a common-screw fit
# will always find SOME axis. Only the pixels can say whether that axis is
# carrying real information.
# ===========================================================================
PRISMATIC_AT_RESOLUTION = 'PRISMATIC_AT_RESOLUTION'
REVOLUTE = 'REVOLUTE'
UNRESOLVED_GEOMETRY = 'UNRESOLVED_GEOMETRY'
MODEL_MISMATCH = 'MODEL_MISMATCH'


def rgbd_null_test_for_prismatic(problem, candidate, *, confidence=.95,
                                 max_iters=30, folds=2):
    """Section 37: test H_0 beta = 0 at the best prismatic candidate.

    `candidate` is the dict from `geometric_joint_candidate`. Its prismatic and
    revolute charts are refit against the real RGB-D residual and compared;
    the comparison is then repeated on held-out cameras, because an in-sample
    improvement from a more flexible model is not evidence (that is exactly
    the `gain > floor` reasoning section 38 replaces).

    Returns the four outcomes of section 35 D. The geometric candidate is
    carried through unchanged whenever the images cannot overturn it -- the
    renderer confirms or refuses, it does not re-classify from scratch.
    """
    from .common_screw import chi2_quantile

    status = candidate.get('status')
    if status in (MODEL_MISMATCH,):
        return {'status': MODEL_MISMATCH, 'confirmed': False,
                'reason': ('section 35 A/B already rejected the one-rigid-'
                           'body or one-DoF model; the renderer is not asked '
                           'to classify a support that has no joint')}
    pchart = candidate.get('chart_prismatic')
    rchart = candidate.get('chart_revolute')
    if pchart is None or rchart is None:
        return {'status': UNRESOLVED_GEOMETRY, 'confirmed': False,
                'reason': ('no geometric candidate chart was produced, so '
                           'there is nothing for the images to confirm')}

    pfit = fit_joint_from_rgbd(problem, pchart, fixed_curvature=True,
                               max_iters=max_iters)
    rfit = fit_joint_from_rgbd(problem, rchart, max_iters=max_iters)
    train = compare_candidates(problem, pfit['chart'], rfit['chart'], -1,
                               'train')
    heldout, fold_fits = heldout_camera_validation(problem, pfit, rfit,
                                                   folds=folds,
                                                   max_iters=max_iters)

    train_gain = float(sum(r['gain_total'] for r in train))
    held_gain = float(sum(r['gain_total'] for r in heldout)) if heldout else None
    # Curvature costs two parameters over the whole window (the direction of
    # eta in the plane perpendicular to d), so the in-sample improvement is
    # read against chi2(2).
    cut = chi2_quantile(2, confidence)
    supports_rotation = train_gain > cut
    held_supports = None if held_gain is None else held_gain > 0.0

    geometric = candidate.get('candidate')
    if held_supports is None:
        status = UNRESOLVED_GEOMETRY
        why = ('section 34: fewer than two cameras, so no held-out evidence '
               'exists and an in-sample improvement cannot be trusted')
        confirmed = False
    elif geometric == 'revolute' and supports_rotation and held_supports:
        status, confirmed = REVOLUTE, True
        why = ('section 35 D: the geometric candidate is revolute and the '
               'original RGB-D residual improves with the angular component '
               'on cameras that were not fitted')
    elif geometric == 'revolute' and not held_supports:
        status, confirmed = UNRESOLVED_GEOMETRY, False
        why = ('section 34: the geometry proposed an axis but held-out images '
               'do not support it, so the angular component is not '
               'observationally confirmed')
    elif geometric == 'prismatic' and not supports_rotation:
        status, confirmed = PRISMATIC_AT_RESOLUTION, True
        why = ('section 37: H_0 beta = 0 is not rejected -- the images are '
               'explained as well without the angular component as with it')
    elif geometric == 'prismatic' and supports_rotation and held_supports:
        status, confirmed = UNRESOLVED_GEOMETRY, False
        why = ('section 34: the geometry found no resolvable rotation, yet '
               'the images prefer one on held-out cameras; the two disagree '
               'and neither is adopted')
    else:
        status = candidate.get('status', UNRESOLVED_GEOMETRY)
        confirmed = False
        why = ('section 34: the image evidence neither confirms nor '
               'overturns the geometric candidate')

    return {'status': status, 'confirmed': confirmed, 'reason': why,
            'geometric_candidate': geometric,
            'train_gain_total': train_gain, 'heldout_gain_total': held_gain,
            'chi2_cut_curvature': cut, 'confidence': confidence,
            'train_rows': train, 'heldout_rows': heldout,
            'fold_fits': fold_fits, 'P': pfit, 'R': rfit,
            'decision_inputs': ('sections 34-37: the geometric candidate from '
                                'the dual differential/finite analysis, '
                                'confirmed or refused by the original RGB-D '
                                'residual on held-out cameras')}
