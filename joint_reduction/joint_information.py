"""Analytic renderer information and continuous influence diagnostics."""
import torch


def information_inverse(A):
    A = (A.double() + A.double().T) * .5
    values, vectors = torch.linalg.eigh(A)
    tol = max(A.shape) * torch.finfo(A.dtype).eps * float(values.abs().max())
    supported = values > tol
    inverse = (vectors * torch.where(supported, values.clamp_min(torch.finfo(A.dtype).tiny).reciprocal(), 0.)) @ vectors.T
    return {'inverse': inverse, 'rank': int(supported.sum()), 'spectrum': values,
            'condition': float(values[-1] / values[supported][0]) if bool(supported.any()) else float('inf'),
            'null_basis': vectors[:, ~supported]}


def accumulate_gaussian_information(linearization, precisions, residuals):
    """Image-only Fisher and score. Alpha responsibility is inside CUDA H."""
    from estimator.renderer_bridge import FrameLinearization
    cameras = []
    for view, precision, residual in zip(linearization.views, precisions, residuals):
        one = FrameLinearization([view], linearization.num_gaussians)
        F = one.build_block_fisher([precision]).double()
        b = one.apply_HT([(precision[0] * residual[0], precision[1] * residual[1])]).double()
        cameras.append({'F_pose': F, 'F_xyz': F[:, :3, :3], 'b_pose': b, 'b_xyz': b[:, :3]})
    if not cameras:
        raise ValueError('at least one camera is required')
    return {'F_image': sum(c['F_xyz'] for c in cameras),
            'F_prior': None, 'prior_note': 'image-only pass; process prior not treated as an observation',
            'cameras': cameras}


def accumulate_camera_information(camera_rows, dimension, device):
    out = {}
    for row in camera_rows:
        cid = row['camera']
        if cid not in out:
            out[cid] = {'camera': cid, 'A': torch.zeros(dimension, dimension, device=device, dtype=torch.float64),
                        'score': torch.zeros(dimension, device=device, dtype=torch.float64)}
        out[cid]['A'] += row['A']
        out[cid]['score'] += row['score']
    return list(out.values())


def accumulate_block_information(rows, block_frames):
    if block_frames < 1:
        raise ValueError('joint block width must be positive')
    blocks = {}
    for row in rows:
        key = (row['camera'], row['frame'] // block_frames)
        if key not in blocks:
            blocks[key] = {'camera': key[0], 'window': key[1],
                           'A': torch.zeros_like(row['A']), 'score': torch.zeros_like(row['score'])}
        blocks[key]['A'] += row['A']
        blocks[key]['score'] += row['score']
    return list(blocks.values())


def gaussian_joint_influence(G, F, b, inverse, step):
    Ai = torch.einsum('nij,nik,nkl->njl', G.double(), F.double(), G.double())
    bi = torch.einsum('nij,nik,nk->nj', G.double(), F.double(), b.double())
    return {'joint_information': Ai.diagonal(dim1=-2, dim2=-1).sum(-1),
            'leverage': torch.einsum('ij,nji->n', inverse, Ai),
            'predicted_delta_E': -2 * (bi @ step) + torch.einsum('i,nij,j->n', step, Ai, step),
            'joint_score': bi}


def alpha_responsibility(state, camera):
    """Reuse the existing sparse pixel-to-Gaussian ownership accumulator."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from estimator.state import matrix_to_quaternion_wxyz
    dev = state.mu.device
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
        bg=torch.zeros(3, device=dev), scale_modifier=1.,
        viewmatrix=camera.viewmatrix, projmatrix=camera.projmatrix,
        sh_degree=0, campos=torch.linalg.inv(camera.viewmatrix)[3, :3],
        prefiltered=False, debug=False, antialiasing=False)
    h, w = camera.image_height, camera.image_width
    out = GaussianRasterizer(settings).accumulate_pixel_features_to_gaussians(
        state.mu.float(), torch.ones(1, h, w, device=dev), torch.ones(h, w, device=dev),
        state.opacity.float(), scales=state.scales.float(),
        rotations=matrix_to_quaternion_wxyz(state.R).float())
    return out[5].reshape(-1).double()
