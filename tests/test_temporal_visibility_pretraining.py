from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from utils.temporal_visibility_pretraining import (
    OCCLUDED,
    PRESENT,
    UNKNOWN,
    VACATED,
    VisibilityConfig,
    _atomic_torch_save,
    build_flow_identity_mask,
    compute_depth_state_likelihoods,
    smooth_visibility_sequence,
)


def _states(depth_values):
    depth = torch.tensor(depth_values, dtype=torch.float32)[:, None, None]
    emissions = compute_depth_state_likelihoods(
        torch.tensor([[1.0]]), depth, torch.tensor([[1.0]]),
        VisibilityConfig(depth_sigma=0.02, occlusion_margin=2.0, vacated_margin=2.0),
    )
    return smooth_visibility_sequence(emissions)["state"][:, 0, 0]


def test_static_visible():
    assert _states([1.0] * 6).tolist() == [PRESENT] * 6


def test_temporary_occlusion_does_not_vote_flow():
    states = _states([1.0, 1.0, 0.75, 0.75, 1.0, 1.0])
    assert states.tolist() == [PRESENT, PRESENT, OCCLUDED, OCCLUDED, PRESENT, PRESENT]
    assert build_flow_identity_mask(states[:, None, None])[:, 0, 0].tolist() == [1, 0, 0, 0, 1]


def test_vacated_and_return_are_allowed():
    states = _states([1.0, 1.0, 1.30, 1.30, 1.0, 1.0])
    assert states.tolist() == [PRESENT, PRESENT, VACATED, VACATED, PRESENT, PRESENT]


def test_invalid_depth_is_unknown():
    states = _states([1.0, 0.0, 1.0])
    assert int(states[1]) == UNKNOWN
    assert not bool(build_flow_identity_mask(states[:, None, None]).any())


def test_atomic_save_retries_zip_writer_failure():
    real_save = torch.save
    calls = []

    def fail_zip_once(value, path, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            Path(path).write_bytes(b"partial")
            raise RuntimeError("synthetic ZIP writer failure")
        return real_save(value, path, **kwargs)

    with TemporaryDirectory() as directory:
        destination = Path(directory) / "artifact.pt"
        with patch("utils.temporal_visibility_pretraining.torch.save", fail_zip_once):
            _atomic_torch_save({"value": torch.tensor([1.0])}, destination)
        assert torch.load(destination)["value"].item() == 1.0
        assert not destination.with_suffix(".pt.tmp").exists()
    assert calls == [{}, {"_use_new_zipfile_serialization": False}]
