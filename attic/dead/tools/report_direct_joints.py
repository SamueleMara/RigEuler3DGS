#!/usr/bin/env python
"""Read completed estimator artifacts and report measured results, including GT only after inference."""
import argparse
import json
from pathlib import Path

import torch


def fmt(value):
    return '—' if value is None else f'{value:.5g}' if isinstance(value,float) else str(value)


def generate(paths, destination, *, focus=None):
    rows=[]; details=[]
    for path in paths:
        path=Path(path)
        artifact=path/'direct_joint_estimates.json'
        if not artifact.exists():
            details.append(f'- `{path}`: no completed direct-estimator artifact.');continue
        payload=json.loads(artifact.read_text())
        gt=path/'dataset_joint_ground_truth.json'
        truth=json.loads(gt.read_text()).get('gt_joint_type') if gt.exists() else None
        for entry in payload['per_part']:
            A,B,C=[entry.get(k) or {} for k in ['SE3','Gaussian','RGBD']]
            decision=entry.get('decision') or {}; predictive=decision.get('predictive') or {}
            uncertainty=C.get('uncertainty') or {}
            bdecision=B.get('decision') or {}
            label=f'{path.parent.name}/{path.name} part {entry["part"]}'
            row=[label,truth,A.get('selected'),bdecision.get('selected') or bdecision.get('status'),
                 decision.get('selected'),A.get('kappa'),B.get('kappa'),C.get('kappa'),
                 decision.get('curvature_snr'),predictive.get('mean'),decision.get('status') or entry.get('status')]
            rows.append(row)
            details += ['',f'## {label}','',f'Artifacts: `{path}`.',
                f'Information rank: {uncertainty.get("information_rank")}; camera/time blocks: {uncertainty.get("num_blocks")}.',
                f'Covariance source: {uncertainty.get("source")}.',
                f'Validation: {decision.get("validation_mode")}; production authoritative: {decision.get("production_authoritative",False)}.',
                f'Held-out block gain CI: {predictive.get("ci")}.',
                f'Physical trajectory discrepancy: `{json.dumps(entry.get("disagreement"))}`.',
                f'Convergence: A={A.get("status")}, B={B.get("status")}, C={C.get("status")}, P={entry.get("RGBD_P",{}).get("status")}.']
            if focus and focus in str(path):
                details += ['', 'Curvature vectors (SE3, Gaussian, RGB-D):',
                    f'`{json.dumps([f.get("eta") for f in [A,B,C]])}`.',
                    f'RGB-D curvature covariance: `{json.dumps(C.get("curvature",{}).get("Sigma_eta"))}`.']
                camera=entry.get('camera_influence') or []
                camera=sorted(camera,key=lambda r:sum(v*v for v in r['delta'][2:4]),reverse=True)
                details += [f'Camera influence on scaled beta, descending: `{json.dumps([{ "camera":r["camera"], "delta_beta":r["delta"][2:4]} for r in camera])}`.',
                    f'Information-mass curve: `{json.dumps(entry.get("information_mass_curve"))}`.',
                    'Per-frame and per-camera gains are in `joint_support_by_frame.csv` and `joint_support_by_camera.csv`; block gains are in `heldout_joint_comparison.csv`.']
                support=path/'gaussian_joint_support.pt'
                if support.exists():
                    data=torch.load(support,map_location='cpu',weights_only=False)
                    part=next((r for r in data if r['part']==entry['part']),None)
                    if part:
                        gain=part['predicted_gain_R'].sum(0)
                        positive=gain.clamp_min(0);order=positive.argsort(descending=True)
                        total=float(positive.sum());ids=part['gaussian_id']
                        top=order[:10]
                        details += [f'Top Gaussian R-support IDs: `{ids[top].tolist()}`; quadratic gains: `{gain[top].tolist()}`.']
                        if total:
                            cumulative=positive[order].cumsum(0)
                            n50=int(torch.searchsorted(cumulative,.5*total))+1
                            n90=int(torch.searchsorted(cumulative,.9*total))+1
                            visible=part['visible_camera_count'].mean(0)
                            details += [f'{n50}/{len(gain)} Gaussians account for 50% of positive quadratic R support; {n90}/{len(gain)} account for 90%.',
                                f'Mean visible cameras: top ten support={float(visible[top].mean()):.4g}, whole support={float(visible.mean()):.4g}.',
                                'Quadratic per-Gaussian support omits inter-Gaussian cross terms; aggregate pixel-space held-out loss is the family-comparison evidence. Boundary/occlusion causation needs visual inspection of the accompanying HTML.']
    title=['# Direct Gaussian and analytic RGB-D joint validation','',
           'Only completed artifacts are included. Unresolved geometry is an uncertainty result, not a wrong family label.','',
           '| Scene | GT | SE3 type | Gaussian type | RGB-D type | kappa SE3 | kappa G | kappa RGBD | curvature SNR | held-out gain | status |',
           '|---|---|---|---|---|---:|---:|---:|---:|---:|---|']
    title += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows]
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(title+details)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--focus',default=None,help='path substring for expanded influence analysis')
    args=parser.parse_args()
    generate(args.paths,args.out,focus=args.focus)
    print(args.out)
