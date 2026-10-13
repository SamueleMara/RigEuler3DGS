"""The full 12-DOF free-motion state, exported and read back.

Section 2 of CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md.

`phase1_centers/` stores centers and nothing else. Everything downstream then
had to invent the rest of the state: the single-source CLI took `R` from the
endpoint model and held it constant, differenced the centers for `v`, and set
`omega = 0`. That is not the state the PDF defines, and for an articulated
rotation it can start outside the image optimiser's capture region -- the
orientations are simply wrong by the whole sweep.

The filter has the state all along. `tools/kpart_pipeline.bootstrap_positions`
carries a full `GaussianMotionState` from frame to frame and appends only
`state.mu`; it already offers an `observer(ti, state, prior)` hook and calls
it with that state. So nothing in the filter's mathematics has to change: this
module is the observer, and a reader for what it writes.

Two conventions are carried explicitly rather than assumed, because getting
either wrong is invisible in the array shapes:

  ORDER  the filter runs in PROCESSING order. For a backward direction that
         is original frame T-1 down to 0. Everything here is written in
         ORIGINAL CHRONOLOGICAL order.
  RATES  a reverse walk's `v` is the rate at which the state moves as the
         PROCESSING index advances, and that index runs backwards through
         physical time. Physical rates are therefore negated as well as
         reordered -- `free_dynamics.reverse_to_physical_state` is the one
         implementation of that and is called here, not reimplemented.

What is NOT exported is anything that does not exist. `iekf_frame_step`'s
information observer is handed `Lambda_prior[:, :3, :3]` and
`fisher_pose[:, :3, :3]` -- the TRANSLATION blocks only -- so that is what is
saved, under names that say so. There is no full 12x12 posterior covariance
in the filter's output to export, and inventing one would be worse than the
gap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "unified_free_full_state/1"
CENTER_ONLY = "CENTER_ONLY_APPROXIMATE"
FULL_12D = "FULL_12D_FILTER"


def _sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.shape}:{contiguous.dtype.str}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class FullStateRecorder:
    """An output-only observer for the existing free IEKF.

    Attach it to `bootstrap_positions(..., observer=recorder)`. It reads the
    state the filter already computed and writes nothing back, so the run it
    records is the run that would have happened without it.
    """

    def __init__(self, *, expected_frames: int, keep_information: bool = True):
        self.expected_frames = int(expected_frames)
        self.keep_information = bool(keep_information)
        self.mu: dict = {}
        self.quat: dict = {}
        self.v: dict = {}
        self.omega: dict = {}
        self.prior_translation: dict = {}
        self.fisher_translation: dict = {}
        self.appearance: dict | None = None

    def __call__(self, processing_index, state, prior=None, fisher=None):
        from estimator.state import matrix_to_quaternion_wxyz

        t = int(processing_index)
        self.mu[t] = state.mu.detach().to("cpu", torch.float64).clone()
        self.quat[t] = matrix_to_quaternion_wxyz(
            state.R.detach()).to("cpu", torch.float64).clone()
        self.v[t] = state.v.detach().to("cpu", torch.float64).clone()
        self.omega[t] = state.omega.detach().to("cpu", torch.float64).clone()
        if self.appearance is None:
            # Frozen by the estimator's own contract; stored once so a reader
            # can rebuild a renderable state without reopening the model.
            self.appearance = {
                "colors": state.colors.detach().to("cpu", torch.float32).clone(),
                "opacity": state.opacity.detach().to("cpu", torch.float32).clone(),
                "scales": state.scales.detach().to("cpu", torch.float32).clone()}
        if self.keep_information and prior is not None:
            self.prior_translation[t] = torch.as_tensor(prior).to(
                "cpu", torch.float32).clone()
        if self.keep_information and fisher is not None:
            self.fisher_translation[t] = torch.as_tensor(fisher).to(
                "cpu", torch.float32).clone()

    # -- writing ----------------------------------------------------------
    def _stack(self, store, order):
        missing = [t for t in range(len(order)) if t not in store]
        if missing:
            raise ValueError(
                f"the recorder never saw processing frames {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}; a partial trajectory "
                "must not be written as a complete one")
        return torch.stack([store[t] for t in range(len(order))])

    def write(self, out_dir, *, direction: str, canonical_endpoint: str,
              processing_order_original_indices, dt_seconds: float,
              dt_is_measured: bool, model_path: str, source_path: str,
              ply_sha256: str = "", scene: str = "", extra: dict | None = None
              ) -> dict:
        """Write the state in ORIGINAL chronological order."""
        from bidirectional_iekf.free_dynamics import reverse_to_physical_state
        from estimator.state import quaternion_wxyz_to_matrix

        order = [int(i) for i in processing_order_original_indices]
        if len(order) != self.expected_frames:
            raise ValueError(
                f"{len(order)} original indices but {self.expected_frames} "
                "frames were expected")
        mu = self._stack(self.mu, order)
        quat = self._stack(self.quat, order)
        v = self._stack(self.v, order)
        omega = self._stack(self.omega, order)
        R = quaternion_wxyz_to_matrix(quat.reshape(-1, 4)).reshape(
            quat.shape[0], quat.shape[1], 3, 3)

        backward = order[0] > order[-1]
        if backward:
            if order != sorted(order, reverse=True):
                raise ValueError(
                    f"a backward run's processing order must be strictly "
                    f"decreasing; got {order[:4]}...{order[-4:]}")
            mu, R, v, omega = reverse_to_physical_state(mu, R, v, omega)
            rate_convention = ("reordered AND negated by "
                               "free_dynamics.reverse_to_physical_state")
        else:
            if order != sorted(order):
                raise ValueError(
                    f"a forward run's processing order must be strictly "
                    f"increasing; got {order[:4]}...{order[-4:]}")
            rate_convention = "already physical; neither reordered nor negated"
        from estimator.state import matrix_to_quaternion_wxyz
        quat = matrix_to_quaternion_wxyz(R.reshape(-1, 3, 3)).reshape(
            R.shape[0], R.shape[1], 4)
        original_frames = sorted(order)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "mu_original_order.npy": mu.numpy(),
            "quat_wxyz_original_order.npy": quat.numpy(),
            "v_physical_original_order.npy": v.numpy(),
            "omega_physical_original_order.npy": omega.numpy(),
            "original_frame_ids.npy": np.asarray(original_frames, dtype=np.int64),
        }
        if self.appearance is not None:
            for name, tensor in self.appearance.items():
                arrays[f"appearance_{name}.npy"] = tensor.numpy()
        information = {}
        for label, store in (("prior_information_translation", self.prior_translation),
                             ("measurement_fisher_translation", self.fisher_translation)):
            if not store:
                continue
            # Frame 0 of the processing order has no prediction step, so the
            # store may be short by one; that is recorded, not padded.
            present = sorted(store)
            stacked = torch.stack([store[t] for t in present])
            if backward:
                stacked = stacked.flip(0)
                frames_for = [order[t] for t in present][::-1]
            else:
                frames_for = [order[t] for t in present]
            arrays[f"{label}_original_order.npy"] = stacked.numpy()
            arrays[f"{label}_frames.npy"] = np.asarray(frames_for, dtype=np.int64)
            information[label] = {
                "shape": list(stacked.shape),
                "frames_covered": frames_for,
                "what_it_is": ("the TRANSLATION 3x3 block only. "
                               "iekf_frame_step's information observer is "
                               "handed Lambda_prior[:, :3, :3] and "
                               "fisher_pose[:, :3, :3]; there is no full "
                               "12x12 posterior covariance in the filter's "
                               "output, and none is invented here"),
            }
        for name, array in arrays.items():
            np.save(out_dir / name, array)

        manifest = {
            "schema": SCHEMA,
            "warm_start_mode": FULL_12D,
            "scene": scene,
            "direction": direction,
            "canonical_endpoint": canonical_endpoint,
            "model_path": str(model_path),
            "source_path": str(source_path),
            "ply_sha256": ply_sha256,
            "num_frames": int(mu.shape[0]),
            "num_gaussians": int(mu.shape[1]),
            "original_frame_ids": original_frames,
            "processing_order_original_indices": order,
            "trajectory_order": "original_chronological",
            "rotation_storage": "unit quaternion wxyz, world-from-local",
            "rotation_convention": ("R+ = Exp(hat(dphi)) @ R, the left/spatial "
                                    "SO(3) convention used throughout"),
            "rate_convention": rate_convention,
            "rates_are_per": "second, at the dt recorded below",
            "dt_seconds": float(dt_seconds),
            "dt_is_measured": bool(dt_is_measured),
            "dt_note": ("the dataset's job_config_resolved.yaml records "
                        "fps: null and duration_sec: null, so dt is a "
                        "DECLARED convention; what the dataset spaces evenly "
                        "is the joint coordinate q, not a clock. A constant "
                        "dt rescales v, omega and Q together and no reported "
                        "quantity depends on its value"
                        if not dt_is_measured else "measured from the dataset"),
            "appearance": ("colors, opacity and scales as the estimator "
                           "received them; frozen by its own contract and "
                           "copied, never updated"),
            "information": information or {
                "none": "the run was recorded without the information observer"},
            "sha256": {name: _sha256(array) for name, array in arrays.items()},
        }
        manifest.update(extra or {})
        (out_dir / "full_state_manifest.json").write_text(
            json.dumps(manifest, indent=1))
        return manifest


def load_full_state(state_dir, *, frames=None, device="cuda",
                    dtype=torch.float32) -> dict:
    """Read a full-state export back, in original chronological order."""
    from estimator.state import quaternion_wxyz_to_matrix

    state_dir = Path(state_dir)
    manifest_path = state_dir / "full_state_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{state_dir} has no full_state_manifest.json. A directory of "
            "arrays carries no evidence of its order or its rate convention, "
            "and guessing either is how a backward trajectory gets consumed "
            "as a forward one with every shape correct.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unknown schema {manifest.get('schema')!r}")
    if manifest.get("trajectory_order") != "original_chronological":
        raise ValueError(
            f"trajectory_order={manifest.get('trajectory_order')!r}")

    ids = np.load(state_dir / "original_frame_ids.npy").tolist()
    index = {int(f): i for i, f in enumerate(ids)}
    want = [int(f) for f in (ids if frames is None else frames)]
    missing = [f for f in want if f not in index]
    if missing:
        raise KeyError(f"frames {missing} are not in this export ({ids[0]}..{ids[-1]})")
    rows = [index[f] for f in want]

    def read(name):
        return torch.from_numpy(np.load(state_dir / name)[rows]).to(device, dtype)

    quat = read("quat_wxyz_original_order.npy")
    R = quaternion_wxyz_to_matrix(quat.reshape(-1, 4)).reshape(
        len(rows), -1, 3, 3)
    out = {"manifest": manifest, "original_frame_ids": want,
           "mu": read("mu_original_order.npy"), "R": R.contiguous(),
           "v": read("v_physical_original_order.npy"),
           "omega": read("omega_physical_original_order.npy")}
    for name in ("colors", "opacity", "scales"):
        path = state_dir / f"appearance_{name}.npy"
        if path.is_file():
            out[name] = torch.from_numpy(np.load(path)).to(device, dtype)
    return out


def trajectory_from_full_state(loaded: dict, *, dt: float, appearance=None):
    """A `Trajectory` of `GaussianMotionState`, ready for the unified IEKS.

    `appearance` overrides the stored colors/opacity/scales; pass the trained
    model's own when the estimator should see exactly what it was fitted to.
    """
    from bidirectional_iekf.ieks import Trajectory
    from estimator.state import GaussianMotionState

    payload = appearance or {k: loaded[k] for k in ("colors", "opacity", "scales")
                             if k in loaded}
    for key in ("colors", "opacity", "scales"):
        if key not in payload:
            raise KeyError(
                f"no {key} in the export and none supplied; the estimator "
                "renders with the model's frozen appearance and must not "
                "substitute a default")
    states = []
    for t in range(loaded["mu"].shape[0]):
        states.append(GaussianMotionState(
            mu=loaded["mu"][t].contiguous(), R=loaded["R"][t].contiguous(),
            v=loaded["v"][t].contiguous(), omega=loaded["omega"][t].contiguous(),
            colors=payload["colors"], opacity=payload["opacity"],
            scales=payload["scales"]))
    return Trajectory(states=states,
                      frame_ids=list(loaded["original_frame_ids"]), dt=float(dt))
