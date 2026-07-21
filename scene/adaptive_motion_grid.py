#
# Sparse adaptive motion grid for the Eulerian velocity-field extension.
# The grid controls where the velocity field has capacity. It never renders
# directly and does not replace Gaussian densification.
#

import torch
from torch import nn


class AdaptiveMotionGrid(nn.Module):
    def __init__(self, cell_centers, gaussian_cell_id, feature_dim=32, voxel_size=0.01):
        super().__init__()
        device = cell_centers.device
        self.feature_dim = int(feature_dim)
        self.voxel_size = float(voxel_size)
        self.register_buffer("cell_centers", cell_centers.float().contiguous())
        self.register_buffer("cell_level", torch.zeros(cell_centers.shape[0], dtype=torch.long, device=device))
        self.register_buffer("active_mask", torch.ones(cell_centers.shape[0], dtype=torch.bool, device=device))
        self.register_buffer("parent_id", -torch.ones(cell_centers.shape[0], dtype=torch.long, device=device))
        self.register_buffer("gaussian_cell_id", gaussian_cell_id.long().contiguous())
        self.features = nn.Parameter(torch.zeros(cell_centers.shape[0], self.feature_dim, device=device))

        # Persistent ambiguity score used for stable adaptive splitting.
        # Updated as: ema <- decay * ema + (1 - decay) * current_score.
        # Stored in the topology state so resuming/checkpointing preserves the
        # accumulated evidence about problematic cells.
        self.register_buffer("score_ema", torch.zeros(cell_centers.shape[0], device=device))
        self.register_buffer("score_ema_updates", torch.zeros(1, dtype=torch.long, device=device))
        self.last_update_info = {}

    @classmethod
    def build_from_gaussians(cls,
                             xyz,
                             opacity=None,
                             feature_dim=32,
                             voxel_size=-1.0,
                             opacity_threshold=0.005,
                             max_initial_cells=50000,
                             foreground_mask=None,
                             min_foreground_points=32):
        """Create motion cells for dynamic-candidate Gaussians.

        If --filter_sem is enabled, `foreground_mask` is the semantic foreground
        mask used by dynamic training. Non-candidate Gaussians receive cell id -1
        and are never queried by the dynamic transport path. If --filter_sem is
        not enabled, the mask is all-True and all Gaussians are candidates.
        """
        with torch.no_grad():
            xyz_det = xyz.detach()
            device = xyz_det.device
            n = xyz_det.shape[0]
            if n == 0:
                centers = torch.zeros(1, 3, device=device)
                ids = torch.zeros(0, dtype=torch.long, device=device)
                return cls(centers, ids, feature_dim=feature_dim, voxel_size=1.0)

            occ_mask = torch.ones(n, dtype=torch.bool, device=device)
            if opacity is not None and opacity.numel() == n:
                occ_mask = occ_mask & (opacity.detach().reshape(-1) > float(opacity_threshold))

            if foreground_mask is not None:
                fg = foreground_mask.detach().reshape(-1).to(device=device).bool()
                if fg.numel() != n:
                    raise ValueError(f"foreground_mask must have length {n}, got {fg.numel()}")
                occ_mask = occ_mask & fg

            min_pts = int(max(1, min_foreground_points))
            if int(occ_mask.sum().item()) < min_pts:
                if opacity is not None and opacity.numel() == n:
                    score = opacity.detach().reshape(-1)
                    k = min(n, min_pts)
                    top = torch.argsort(score, descending=True)[:k]
                    occ_mask = torch.zeros(n, dtype=torch.bool, device=device)
                    occ_mask[top] = True
                else:
                    occ_mask = torch.ones(n, dtype=torch.bool, device=device)

            pts = xyz_det[occ_mask]
            extent = torch.clamp(pts.max(dim=0).values - pts.min(dim=0).values, min=1e-6)
            if voxel_size is None or float(voxel_size) <= 0:
                voxel_size = float(extent.max().item() / 64.0)
                voxel_size = max(voxel_size, 1e-5)

            origin = pts.min(dim=0).values
            coords_occ = torch.floor((pts - origin) / float(voxel_size)).long()
            unique_coords, inverse_occ = torch.unique(coords_occ, dim=0, return_inverse=True)

            if unique_coords.shape[0] > max_initial_cells:
                factor = int(torch.ceil(torch.tensor(unique_coords.shape[0] / float(max_initial_cells))).item())
                factor = max(2, factor)
                voxel_size = float(voxel_size) * float(factor)
                coords_occ = torch.floor((pts - origin) / float(voxel_size)).long()
                unique_coords, inverse_occ = torch.unique(coords_occ, dim=0, return_inverse=True)

            ids = -torch.ones(n, dtype=torch.long, device=device)
            ids[occ_mask] = inverse_occ.long()

            centers = origin + (unique_coords.float() + 0.5) * float(voxel_size)
            return cls(centers, ids, feature_dim=feature_dim, voxel_size=voxel_size)

    @property
    def num_cells(self):
        return int(self.cell_centers.shape[0])

    @property
    def device(self):
        return self.cell_centers.device

    def query_features(self, x, gaussian_cell_id=None):
        if gaussian_cell_id is None:
            gaussian_cell_id = self.gaussian_cell_id
        if gaussian_cell_id.numel() != x.shape[0]:
            gaussian_cell_id = gaussian_cell_id[:x.shape[0]]
        ids = gaussian_cell_id.long()
        out = torch.zeros(x.shape[0], self.feature_dim, device=x.device, dtype=self.features.dtype)
        valid = ids >= 0
        if valid.any():
            safe = ids[valid].clamp(min=0, max=self.features.shape[0] - 1)
            out[valid] = self.features[safe]
        return out

    def active_cell_ids(self):
        return torch.where(self.active_mask)[0]

    def topology_state_dict(self):
        return {
            "cell_centers": self.cell_centers,
            "cell_level": self.cell_level,
            "active_mask": self.active_mask,
            "parent_id": self.parent_id,
            "gaussian_cell_id": self.gaussian_cell_id,
            "features": self.features.detach(),
            "score_ema": self.score_ema.detach(),
            "score_ema_updates": self.score_ema_updates.detach(),
            "feature_dim": self.feature_dim,
            "voxel_size": self.voxel_size,
        }

    @classmethod
    def from_topology_state_dict(cls, state):
        grid = cls(state["cell_centers"], state["gaussian_cell_id"],
                   feature_dim=int(state.get("feature_dim", state["features"].shape[-1])),
                   voxel_size=float(state.get("voxel_size", 0.01)))
        grid.cell_level = state["cell_level"].long()
        grid.active_mask = state["active_mask"].bool()
        grid.parent_id = state["parent_id"].long()
        grid.features = nn.Parameter(state["features"].float().contiguous())
        if "score_ema" in state:
            grid.score_ema = state["score_ema"].float().to(grid.device).contiguous()
        if "score_ema_updates" in state:
            grid.score_ema_updates = state["score_ema_updates"].long().to(grid.device).reshape(1)
        grid._ensure_score_ema_size()
        return grid

    def _ensure_score_ema_size(self):
        """Keep EMA buffers aligned with the current number of cells."""
        device = self.device
        n = int(self.num_cells)
        if not hasattr(self, "score_ema") or self.score_ema.numel() != n:
            old = getattr(self, "score_ema", None)
            new = torch.zeros(n, device=device)
            if old is not None and old.numel() > 0:
                k = min(int(old.numel()), n)
                new[:k] = old.detach().to(device)[:k]
            self.score_ema = new
        if not hasattr(self, "score_ema_updates") or self.score_ema_updates.numel() != 1:
            self.score_ema_updates = torch.zeros(1, dtype=torch.long, device=device)

    def _update_score_ema(self, scores, decay=0.95):
        """Update persistent per-cell score EMA and return decision scores."""
        self._ensure_score_ema_size()
        scores_det = scores.detach().float().to(self.device)
        if scores_det.numel() < self.num_cells:
            padded = torch.zeros(self.num_cells, device=self.device)
            padded[: scores_det.numel()] = scores_det
            scores_det = padded
        elif scores_det.numel() > self.num_cells:
            scores_det = scores_det[: self.num_cells]

        scores_det = torch.where(torch.isfinite(scores_det), scores_det, torch.zeros_like(scores_det))
        valid = self.active_mask[: self.num_cells]
        decay = float(decay)
        use_ema = 0.0 <= decay < 1.0
        if use_ema:
            if int(self.score_ema_updates.item()) == 0:
                self.score_ema[valid] = scores_det[valid]
            else:
                self.score_ema[valid] = decay * self.score_ema[valid] + (1.0 - decay) * scores_det[valid]
            self.score_ema_updates += 1
            return self.score_ema.detach().clone(), scores_det
        return scores_det.detach().clone(), scores_det

    def update_topology(self,
                        scores,
                        xyz0,
                        split_threshold=1e-3,
                        split_percentile=-1.0,
                        score_ema_decay=0.95,
                        merge_threshold=1e-5,
                        max_splits=16,
                        min_points=32,
                        noise_std=1e-4,
                        topk=5):
        """Split/merge motion cells using detached diagnostic scores.

        Returns True when the feature tensor shape changed and the optimizer must
        be rebuilt by the caller.
        """
        with torch.no_grad():
            decision_scores, current_scores = self._update_score_ema(scores, decay=score_ema_decay)
            active_scores = decision_scores.clone()
            valid = self.active_mask[: active_scores.numel()]

            top_cells = []
            if active_scores.numel() > 0 and valid.any() and int(topk) > 0:
                masked = torch.where(valid, active_scores, torch.full_like(active_scores, -float("inf")))
                k = min(int(topk), int(valid.sum().item()))
                vals, ids = torch.topk(masked, k=k)
                for cid, val in zip(ids.tolist(), vals.tolist()):
                    if not torch.isfinite(torch.tensor(val)):
                        continue
                    pts = int((self.gaussian_cell_id == int(cid)).sum().item())
                    cur = float(current_scores[int(cid)].item()) if int(cid) < current_scores.numel() else 0.0
                    top_cells.append({"cell_id": int(cid), "score": float(val), "current_score": cur, "points": pts})

            old_active = int(self.active_mask.sum().item())
            old_cells = int(self.num_cells)
            changed_merge, merge_count = self._merge_low_score_children(decision_scores, merge_threshold)
            changed_split, split_count, child_count, cutoff = self._split_high_score_cells(
                decision_scores, xyz0.detach(), split_threshold, split_percentile, max_splits, min_points, noise_std
            )
            self.last_update_info = {
                "old_active_cells": old_active,
                "old_total_cells": old_cells,
                "new_active_cells": int(self.active_mask.sum().item()),
                "new_total_cells": int(self.num_cells),
                "split_count": int(split_count),
                "new_child_count": int(child_count),
                "merge_count": int(merge_count),
                "score_ema_decay": float(score_ema_decay),
                "score_ema_updates": int(self.score_ema_updates.item()),
                "split_percentile": float(split_percentile),
                "split_cutoff": float(cutoff),
                "decision_score_max": float(active_scores[valid].max().item()) if active_scores.numel() and valid.any() else 0.0,
                "decision_score_mean": float(active_scores[valid].mean().item()) if active_scores.numel() and valid.any() else 0.0,
                "current_score_max": float(current_scores[valid].max().item()) if current_scores.numel() and valid.any() else 0.0,
                "current_score_mean": float(current_scores[valid].mean().item()) if current_scores.numel() and valid.any() else 0.0,
                "top_cells": top_cells,
            }
        return bool(changed_merge or changed_split)

    def _split_high_score_cells(self, scores, xyz0, split_threshold, split_percentile, max_splits, min_points, noise_std):
        if scores.numel() == 0 or self.gaussian_cell_id.numel() == 0:
            return False, 0, 0, 0.0

        active = self.active_mask[:scores.shape[0]]
        finite = torch.isfinite(scores)
        valid_base = active & finite
        if not valid_base.any():
            return False, 0, 0, 0.0

        active_vals = scores[valid_base]
        cutoff = float(split_threshold)
        if float(split_percentile) >= 0.0 and active_vals.numel() > 0:
            q = float(split_percentile) / 100.0 if float(split_percentile) > 1.0 else float(split_percentile)
            q = min(max(q, 0.0), 1.0)
            cutoff = max(cutoff, float(torch.quantile(active_vals, q).item()))

        valid_ids = torch.where(valid_base & (scores >= cutoff))[0]
        if valid_ids.numel() == 0:
            return False, 0, 0, cutoff
        vals = scores[valid_ids]
        order = torch.argsort(vals, descending=True)
        valid_ids = valid_ids[order[:int(max_splits)]]

        new_centers = []
        new_levels = []
        new_parents = []
        new_features = []
        changed = False
        split_count = 0
        ids = self.gaussian_cell_id
        for cid in valid_ids.tolist():
            pts_idx = torch.where(ids == cid)[0]
            if pts_idx.numel() < int(min_points):
                continue
            pts = xyz0[pts_idx]
            center = pts.mean(dim=0)
            signs = (pts > center).long()
            child_keys = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]
            unique_child_keys = torch.unique(child_keys)
            if unique_child_keys.numel() <= 1:
                continue
            self.active_mask[cid] = False
            parent_feat = self.features[cid].detach()
            for key in unique_child_keys.tolist():
                local = pts_idx[child_keys == key]
                if local.numel() == 0:
                    continue
                new_id = self.cell_centers.shape[0] + len(new_centers)
                self.gaussian_cell_id[local] = new_id
                child_center = xyz0[local].mean(dim=0)
                new_centers.append(child_center)
                new_levels.append(self.cell_level[cid] + 1)
                new_parents.append(torch.tensor(cid, device=self.device, dtype=torch.long))
                feat = parent_feat.clone()
                if noise_std > 0:
                    feat = feat + torch.randn_like(feat) * float(noise_std)
                new_features.append(feat)
            changed = True
            split_count += 1

        if not changed:
            return False, 0, 0, cutoff

        self.cell_centers = torch.cat([self.cell_centers, torch.stack(new_centers, dim=0)], dim=0)
        self.cell_level = torch.cat([self.cell_level, torch.stack(new_levels, dim=0).long()], dim=0)
        self.parent_id = torch.cat([self.parent_id, torch.stack(new_parents, dim=0).long()], dim=0)
        self.active_mask = torch.cat([self.active_mask, torch.ones(len(new_centers), dtype=torch.bool, device=self.device)], dim=0)
        self.features = nn.Parameter(torch.cat([self.features.detach(), torch.stack(new_features, dim=0)], dim=0))
        self._ensure_score_ema_size()
        return True, int(split_count), int(len(new_centers)), cutoff

    def _merge_low_score_children(self, scores, merge_threshold):
        if float(merge_threshold) <= 0:
            return False, 0
        if scores.numel() == 0 or self.parent_id.numel() == 0:
            return False, 0
        changed = False
        merge_count = 0
        inactive_parents = torch.where(~self.active_mask & (torch.arange(self.active_mask.shape[0], device=self.device) < scores.shape[0]))[0]
        for pid in inactive_parents.tolist():
            child_ids = torch.where((self.parent_id == pid) & self.active_mask)[0]
            if child_ids.numel() <= 1:
                continue
            if child_ids.max() >= scores.shape[0]:
                continue
            if torch.all(scores[child_ids] < float(merge_threshold)):
                mask = torch.zeros_like(self.gaussian_cell_id, dtype=torch.bool)
                for child in child_ids.tolist():
                    mask |= (self.gaussian_cell_id == child)
                self.gaussian_cell_id[mask] = pid
                self.features.data[pid] = self.features.data[child_ids].mean(dim=0)
                self.active_mask[pid] = True
                self.active_mask[child_ids] = False
                changed = True
                merge_count += 1
        return changed, int(merge_count)
