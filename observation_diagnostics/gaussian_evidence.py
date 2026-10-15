"""Per-Gaussian accounting of actual pixel contributions, frame by frame.

For source d, Gaussian i and pixel p with composited weight q = T * alpha
(observation_diagnostics.renderer_access) and the Gaussian's view depth z
(its centre's camera-space depth, as the renderer uses it in Z):

  kappa_t(p, i) = valid_gt(p) AND |z - D_gt(p)| <= t        (contributor at the
                                                             observed surface)
  mass_total     sum_p q
  mass_valid     sum_p q valid_gt
  mass_kappa     sum_p q kappa                              (primary t)
  mass_supported sum_p q kappa S_d^depth(p)                 (this source explains p)
  witness        sum_p q kappa W_d-only(p)                  (U_i: d explains, the
                                                             other source does not)
  witness_improve sum_p q kappa improvement_d(p) / sigma_d  (continuous U_i)
  mass_front     sum_p q valid [z < D_gt - t]                (in front of the
                                                             observed surface)
  q2_kappa[t]    sum_p q^2 kappa_t                          (pair normaliser)
  q2_all         sum_p q^2

kappa keeps a background splat from taking credit for a pixel whose rendered
depth is right only because a foreground splat dominates it.

q describes the Gaussian's share INSIDE its own source's render. It is not the
change in residual after deleting it, nor what it would do inside a third
model (the counterfactual in tools/analyze_fb_pixel_overlap.py is a separate,
labelled experiment).
"""

from __future__ import annotations

import numpy as np
import torch

from .pixel_evidence import DEPTH_SIGMA_M

PAIR_TOLS_M = (0.02, 0.04, 0.08)
COLUMNS = ("mass_total", "mass_valid", "mass_kappa", "mass_supported", "witness",
           "witness_improve", "mass_front", "mass_gt_invalid", "q2_all",
           "q2_kappa_0", "q2_kappa_1", "q2_kappa_2", "witness_records")


def new_frame_accumulator(N: int, device) -> dict:
    acc = {c: torch.zeros(N, dtype=torch.float64, device=device) for c in COLUMNS}
    acc["cams_visible"] = torch.zeros(N, dtype=torch.int32, device=device)
    acc["cams_supported"] = torch.zeros(N, dtype=torch.int32, device=device)
    acc["cams_witness"] = torch.zeros(N, dtype=torch.int32, device=device)
    return acc


def accumulate(acc: dict, contrib, z_of_gaussian: torch.Tensor, fields: dict,
               gt_depth: torch.Tensor, key: str, cam_index: int,
               contrib_tol: float) -> dict:
    """Add one view's contributions of source `key` ('F' or 'B')."""
    gid = contrib.gid.long()
    if gid.numel() == 0:
        return {"records": 0}
    pix = contrib.pixel
    q = contrib.q.double()
    z = z_of_gaussian[gid].double()
    valid = fields["valid"].reshape(-1)[pix]
    dgt = gt_depth.reshape(-1)[pix].double()
    dz = (z - dgt).abs()
    kappa = valid & (dz <= contrib_tol)
    own_S = (fields["SdF"] if key == "F" else fields["SdB"]).reshape(-1)[pix]
    W = (fields["W_Fonly"] if key == "F" else fields["W_Bonly"]).reshape(-1)[pix]
    imp = (fields["improve_F"] if key == "F" else fields["improve_B"]).reshape(-1)[pix].double()
    N = acc["mass_total"].shape[0]

    def add(name, w):
        acc[name].index_add_(0, gid, w)

    add("mass_total", q)
    add("mass_valid", q * valid)
    add("mass_kappa", q * kappa)
    add("mass_supported", q * (kappa & own_S))
    add("witness", q * (kappa & W))
    add("witness_improve", q * kappa * imp / DEPTH_SIGMA_M)
    add("mass_front", q * (valid & (z < dgt - contrib_tol)))
    add("mass_gt_invalid", q * (~valid))
    add("q2_all", q * q)
    for k, t in enumerate(PAIR_TOLS_M):
        add(f"q2_kappa_{k}", q * q * (valid & (dz <= t)))
    add("witness_records", (kappa & W).double())
    bit = 1 << cam_index

    def mark(name, w):
        hit = torch.zeros(N, dtype=torch.float64, device=gid.device).index_add_(0, gid, w) > 0
        acc[name] |= hit.int() * bit

    mark("cams_visible", q)
    mark("cams_supported", q * (kappa & own_S))
    mark("cams_witness", q * (kappa & W))
    return {"records": int(gid.numel()),
            "records_kappa": int(kappa.sum()),
            "max_contributors_per_pixel": int(contrib.counts.max())}


def to_numpy(acc: dict) -> dict:
    out = {}
    for k, v in acc.items():
        out[k] = v.cpu().numpy().astype(np.float32 if v.dtype == torch.float64 else np.int32)
    return out
