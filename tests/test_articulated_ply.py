"""The final Phase-5 PLY carries part_id / lineage / the frozen joint, with the 3DGS fields unchanged."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.articulated_ply import read_articulated_ply, write_articulated_ply  # noqa: E402
from recovery.contracts import Joint  # noqa: E402
from recovery.gaussians import GaussianTensors  # noqa: E402


def _model(n=50, seed=0):
    r = np.random.default_rng(seed)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    g = GaussianTensors(t(r.normal(size=(n, 3))), t(r.normal(size=(n, 4))), t(r.normal(size=(n, 3))),
                        t(r.normal(size=(n, 1))), t(r.normal(size=(n, 1, 3))), t(r.normal(size=(n, 15, 3))), 3)
    g.lineage["material"][n // 2:] = 1
    g.lineage["parent_uid"][-3:] = [0, 47, 48]                  # 48 <- 47 <- 0: a chain of densified children
    return g


def test_articulated_ply_fields():
    g = _model()
    fr = (59, 58, 57, 56)
    j = Joint("revolute", (0.0, 0.0, 1.0), (0.1, 0.2, 0.3), fr, (0.0, -0.2, -0.5, -0.9), (1.0, 0.01, 0.02, 0.03),
              "backward", "g", "t", "gs", "ts")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "point_cloud.ply"
        g.save_ply(p)
        before = GaussianTensors.load_ply(p, device="cpu")
        write_articulated_ply(p, g.lineage, j, p, unresolved_uids=[0])
        r = read_articulated_ply(p)
        after = GaussianTensors.load_ply(p, device="cpu")       # the 3DGS loader still reads it
        for k in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
            assert torch.equal(getattr(before, k), getattr(after, k)), k
        assert np.array_equal(r["part_id"], g.lineage["material"])
        assert np.nonzero(r["unresolved"])[0].tolist() == [0, 47, 48, 49]
        assert r["joint_type"] == "revolute" and np.allclose(r["axis"], [0, 0, 1]) and np.allclose(r["origin"], [.1, .2, .3])
        assert r["q_range"] == (-0.9, 0.0) and r["anchor_frame"] == 59
        assert r["frames"].tolist() == [56, 57, 58, 59] and np.allclose(r["q"], [-0.9, -0.5, -0.2, 0.0])
        assert any(c.startswith("joint_axis 0 0 1") for c in r["comments"])
        try:
            write_articulated_ply(p, g.lineage, j, p)
            raise AssertionError("a second write must be refused")
        except ValueError:
            pass
    print("  articulated PLY: 3DGS fields identical, part_id / unresolved chain / joint / per-frame q read back")


if __name__ == "__main__":
    test_articulated_ply_fields()
    print("PASS")
