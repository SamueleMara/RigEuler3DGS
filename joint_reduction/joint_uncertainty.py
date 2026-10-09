"""Camera/time-cluster sandwich uncertainty and held-out nested evidence."""
import math

import torch
from scipy.stats import chi2, t as student_t

from .joint_information import information_inverse


def cluster_robust_covariance(A, scores):
    analysis = information_inverse(A)
    scores = torch.as_tensor(scores, device=A.device, dtype=torch.float64)
    if scores.ndim != 2 or scores.shape[1] != A.shape[0]:
        raise ValueError('block scores must have shape [B, parameters]')
    B = len(scores)
    correction = B / (B - 1) if B > 1 else float('nan')
    meat = scores.T @ scores
    covariance = analysis['inverse'] @ meat @ analysis['inverse']
    covariance *= correction
    return {'covariance': covariance, 'information_rank': analysis['rank'],
            'information_spectrum': analysis['spectrum'], 'num_blocks': B,
            'finite_block_correction': correction,
            'correction_definition': 'B/(B-1); no pixel/Gaussian sample-count correction',
            'source': 'camera x contiguous temporal window cluster sandwich',
            'estimable': B > 1 and analysis['rank'] == A.shape[0]}


def curvature_covariance(chart, covariance, information):
    B = chart.basis()
    beta = B.T @ chart.eta
    if chart.fixed_curvature or chart.finite_axis:
        return {'beta': beta, 'eta': chart.eta, 'kappa': float(chart.eta.norm()),
                'rank_beta': 0, 'Sigma_beta': torch.zeros(2, 2, device=B.device).double(),
                'Sigma_eta': torch.zeros(3, 3, device=B.device).double(),
                'Q_eta': None, 'curvature_snr': None, 'estimable': False,
                'reason': 'curvature chart unavailable for the finite-axis branch or constrained P fit'}
    Sigma = covariance[2:4, 2:4] / chart.length ** 2
    ids = torch.tensor([i for i in range(chart.dimension) if i not in (2, 3)], device=B.device)
    profiled = information[2:4, 2:4] - information[2:4][:, ids] @ information_inverse(information[ids[:, None], ids[None, :]])['inverse'] @ information[ids][:, 2:4]
    rank = information_inverse(profiled)['rank']
    finite = bool(torch.isfinite(Sigma).all())
    inv = information_inverse(Sigma) if finite else None
    Q = float(beta @ inv['inverse'] @ beta) if inv else None
    # A covariance supported on fewer directions cannot certify a Wald test.
    return {'beta': beta, 'eta': chart.eta, 'kappa': float(chart.eta.norm()),
            'Sigma_beta': Sigma, 'Sigma_eta': B @ Sigma @ B.T,
            'rank_beta': rank, 'covariance_rank_beta': inv['rank'] if inv else 0,
            'Q_eta': Q, 'curvature_snr': math.sqrt(max(Q, 0.)) if Q is not None else None,
            'estimable': finite and rank == 2 and inv['rank'] == 2}


def heldout_predictive_gain(rows, confidence=.95):
    gains = torch.tensor([r['gain_total'] for r in rows], dtype=torch.float64)
    B = len(gains)
    if B < 2:
        return {'num_blocks': B, 'estimable': False, 'gain_by_block': gains.tolist(),
                'mean': float(gains.mean()) if B else None, 'se': None, 'ci': None}
    mean, se = float(gains.mean()), float(gains.std(unbiased=True) / math.sqrt(B))
    critical = float(student_t.ppf((1 + confidence) / 2, B - 1))
    ci = [mean - critical * se, mean + critical * se]
    bootstrap_ci = None
    if B >= 10:
        rng = torch.Generator().manual_seed(0)
        samples = gains[torch.randint(B, (1000, B), generator=rng)].mean(1)
        bootstrap_ci = samples.quantile(torch.tensor([(1-confidence)/2, (1+confidence)/2]).double()).tolist()
    return {'num_blocks': B, 'estimable': True, 'mean': mean, 'median': float(gains.median()),
            'se': se, 'ci': ci, 'block_bootstrap_ci': bootstrap_ci,
            'positive_blocks': int((gains > 0).sum()), 'negative_blocks': int((gains < 0).sum()),
            'gain_by_block': gains.tolist(), 'confidence_level': confidence,
            'independent_unit': 'held-out camera/time block'}


def curvature_evidence(curvature, predictive, *, confidence=.95, converged=True,
                       model_mismatch=False, authoritative=False):
    if not 0 < confidence < 1:
        raise ValueError('joint confidence must be between zero and one')
    rank = curvature.get('rank_beta', 0)
    critical = float(chi2.ppf(confidence, rank)) if rank else None
    out = {**curvature, 'confidence_level': confidence, 'dof_eta': rank,
           'Q_critical': critical, 'predictive': predictive, 'selected': None,
           'status': 'UNRESOLVED_GEOMETRY', 'validation_mode': 'authoritative' if authoritative else 'fast_diagnostic'}
    if model_mismatch:
        out['status'] = 'MODEL_MISMATCH'
    elif not converged:
        out['status'] = 'UNRESOLVED_OPTIMIZATION'
    elif not curvature.get('estimable') or not predictive.get('estimable'):
        out['reason'] = 'curvature covariance or independent held-out evidence is not estimable'
    elif curvature['Q_eta'] > critical and predictive['ci'][0] > 0:
        out.update(selected='revolute', status='REVOLUTE')
    elif curvature['Q_eta'] <= critical and predictive['ci'][1] < 0:
        out.update(selected='prismatic', status='PRISMATIC_AT_RESOLUTION')
    else:
        out['reason'] = 'curvature uncertainty and held-out prediction do not resolve a common family'
    # The diagnostic never becomes a production claim through Phase-2 leakage.
    out['production_authoritative'] = authoritative
    return out


def model_adequacy(rows, confidence=.95):
    """Known-noise lack-of-fit check, uncertainty across blocks, not pixels."""
    P, R = [], []
    for row in rows:
        n = row.get('valid_rgb', 0) + row.get('valid_depth', 0)
        if n:
            P.append((row['E_P_rgb'] + row['E_P_depth']) / n)
            R.append((row['E_R_rgb'] + row['E_R_depth']) / n)
    def interval(values):
        if len(values) < 2:
            return None
        v = torch.tensor(values).double()
        radius = float(student_t.ppf((1+confidence)/2,len(v)-1)) * float(v.std()/math.sqrt(len(v)))
        return [float(v.mean())-radius, float(v.mean())+radius]
    p, r = interval(P), interval(R)
    return {'P_whitened_loss_per_channel_ci':p, 'R_whitened_loss_per_channel_ci':r,
            'expected_known_noise_variance':1., 'num_blocks':len(P),
            'model_mismatch':bool(p and r and p[0]>1 and r[0]>1),
            'confidence_level':confidence,
            'note':'block variation supplies uncertainty; no independent-pixel covariance correction'}
