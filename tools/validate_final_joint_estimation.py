#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.final_joint_estimation import JointEstimationConfig, estimate_final_joint


def make_tracks(kind: str):
    torch.manual_seed(7)
    T, Nbg, Ns, Nm = 24, 40, 80, 120
    N = Nbg + Ns + Nm
    center = torch.tensor([0.15, -0.05, 0.10])
    axis = torch.nn.functional.normalize(torch.tensor([0.2, -0.1, 1.0]), dim=0)
    anchor = torch.tensor([0.28, 0.05, 0.08])
    # Put anchor in plane normal to axis through center.
    delta = anchor - center
    anchor = center + delta - torch.dot(delta, axis) * axis

    xbg = torch.randn(Nbg, 3) * 0.30 + torch.tensor([0.0, 1.0, 0.0])
    xs = torch.randn(Ns, 3) * 0.18 + torch.tensor([-0.35, 0.0, 0.0])
    xm = torch.randn(Nm, 3) * 0.18 + torch.tensor([0.35, 0.0, 0.0])
    x = torch.cat([xbg, xs, xm], dim=0)

    fg = torch.zeros(N, dtype=torch.bool)
    fg[Nbg:] = True
    p = torch.zeros(N)
    p[Nbg:Nbg+Ns] = 0.02
    p[Nbg+Ns:] = 0.98
    cid = torch.full((N,), -1, dtype=torch.long)
    cid[Nbg:Nbg+Ns] = 0
    cid[Nbg+Ns:] = 1

    times = torch.linspace(0, 1, T)
    rate = 0.7 + 0.2 * torch.sin(2 * torch.pi * times)
    if kind == 'prismatic':
        omega = torch.zeros(T, 3)
        trans = rate[:, None] * axis[None]
    else:
        omega = rate[:, None] * axis[None]
        h = 0.0 if kind == 'revolute' else 0.12
        trans = torch.cross(omega, (center - anchor).expand_as(omega), dim=-1) + h * rate[:, None] * axis[None]

    rel = x[None] - center[None, None]
    rigid = torch.cross(omega[:, None].expand(T, N, 3), rel.expand(T, N, 3), dim=-1) + trans[:, None]
    # Non-foreground is allowed arbitrary proposal motion but must not affect estimation.
    rigid[:, :Nbg] = torch.randn(T, Nbg, 3) * 3.0

    tracks = {
        'canonical_positions': x,
        'dynamic_foreground_mask': fg,
        'dynamic_prob': p[:, None],
        'joint_cluster_id': cid,
        'rigid_omega': omega,
        'rigid_translation': trans,
        'rigid_center': center,
        'rigid_velocities': rigid,
    }
    return tracks


def main():
    cfg = JointEstimationConfig(dynamic_threshold=0.8, min_moving_points=32, max_fit_points=512)
    for expected in ('prismatic', 'revolute', 'screw'):
        result = estimate_final_joint(make_tracks(expected), cfg)
        got = result['joint_type']
        print(expected, '->', got, result['candidate_scores'])
        assert got == expected, (expected, got, result)
        assert result['semantic_isolation']['non_foreground_used_for_joint_fit'] is False
    print('FINAL JOINT ESTIMATION VALIDATION: PASS')


if __name__ == '__main__':
    main()
