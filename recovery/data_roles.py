"""Immutable PROPOSAL / FIT / VALIDATION pixel roles for Phase 4B / 4C (directive 2026-11-11, section 6).

Assigned from (camera index, processing row, pixel) alone, BEFORE any residual or proposal is computed:

    block (bx, by) = (x // B, y // B),   time block tb = row // T
    role = 1 + (bx + 2 by + tb + camera_index) mod 3          1 PROPOSAL, 2 FIT, 3 VALIDATION
    GUARD (0) = pixels within G of a block border

Horizontally and vertically adjacent blocks always carry different roles, and a fixed block cycles through
all three roles over successive time blocks, so a surface seen for a while is proposed, fitted and
validated from DIFFERENT pixels and times. With G >= SSIM_R + 1 the 11 x 11 SSIM window of any scored pixel
lies inside its own block, so a role's score never reads another role's observed values.

What this is NOT: statistically independent evidence. Neighbouring blocks see the same surface, the same
calibration and -- above all -- the same Phase-3 joint, which was fitted on every frame. Validation is
therefore reported as `spatially_and_temporally_separated_conditional_on_phase3_joint`; even / odd rows alone
are never claimed as independence.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .exact_objective import SSIM_R

GUARD, PROPOSAL, FIT, VALIDATION = 0, 1, 2, 3
ROLE_NAMES = {GUARD: "GUARD", PROPOSAL: "PROPOSAL", FIT: "FIT", VALIDATION: "VALIDATION"}
INDEPENDENCE_LABEL = "spatially_and_temporally_separated_conditional_on_phase3_joint"


@dataclass
class RoleConfig:
    block_px: int = 64
    guard_px: int = SSIM_R + 1
    time_block: int = 8

    def to_dict(self):
        return asdict(self)


def role_map(cam_index: int, row: int, H: int, W: int, cfg: RoleConfig, device="cpu") -> torch.Tensor:
    if cfg.guard_px < SSIM_R + 1:
        raise ValueError(f"guard_px {cfg.guard_px} < SSIM_R + 1 = {SSIM_R + 1}: SSIM windows would cross roles")
    B, G = cfg.block_px, cfg.guard_px
    y = torch.arange(H, device=device)[:, None]
    x = torch.arange(W, device=device)[None, :]
    role = 1 + (x // B + 2 * (y // B) + row // cfg.time_block + cam_index) % 3
    guard = ((x % B) < G) | ((x % B) >= B - G) | ((y % B) < G) | ((y % B) >= B - G)
    return torch.where(guard, torch.zeros_like(role), role).to(torch.uint8)


class Roles:
    """Role maps of every view of a run (computed on demand, deterministic)."""

    def __init__(self, views, cfg: RoleConfig):
        self.cfg = cfg
        names, rows = [], {}
        self.cam_index, self.row = [], []
        for vr in views:
            if vr.cam.name not in names:
                names.append(vr.cam.name)
            r = rows.get(vr.cam.name, 0)
            rows[vr.cam.name] = r + 1
            self.cam_index.append(names.index(vr.cam.name))
            self.row.append(r)
        self.shapes = [(vr.cam.H, vr.cam.W) for vr in views]

    def get(self, i, device="cpu") -> torch.Tensor:
        H, W = self.shapes[i]
        return role_map(self.cam_index[i], self.row[i], H, W, self.cfg, device)

    def manifest(self) -> dict:
        h = hashlib.sha256()
        for i in range(len(self.shapes)):
            h.update(self.get(i).numpy().tobytes())
        frac = np.bincount(self.get(0).numpy().ravel(), minlength=4) / float(np.prod(self.shapes[0]))
        return {"config": self.cfg.to_dict(), "rule": "role = 1 + (bx + 2 by + row // T + camera) mod 3; GUARD at block "
                "borders", "roles": ROLE_NAMES, "sha256_all_views": h.hexdigest(),
                "fraction_view0": {ROLE_NAMES[k]: float(frac[k]) for k in range(4)},
                "independence": INDEPENDENCE_LABEL}
