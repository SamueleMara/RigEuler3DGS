"""Gate 7: the optional final joint refinement exists, is tested, and is never part of the default workflow.

  * nothing in recovery/ (except its own module), tools/run_phase4_recovery.py, tools/train_phase5_rgbd_gaussians.py
    or run_bidirectional_all.sh imports or calls it
  * the CLI without --enable is a dry run that writes nothing; --enable without a Phase-5 proof is refused
  * parameterisation: zero correction == the Phase-3 transform exactly; axis stays unit; type preserved;
    prismatic has no origin DOF; revolute point moves only perpendicular to e0; anchor q fixed; gradients
    of a rendered-like loss w.r.t. the minimal coordinates match finite differences
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery.articulated_scene import posed  # noqa: E402
from recovery.joint_refine_final import MinimalJoint, RefineConfig  # noqa: E402
from test_phase45_direction_anchor import toy_joint  # noqa: E402


def test_not_wired_into_default_workflow():
    files = [p for p in (ROOT / "recovery").glob("*.py") if p.name != "joint_refine_final.py"]
    files += [ROOT / "tools" / "run_phase4_recovery.py", ROOT / "tools" / "train_phase5_rgbd_gaussians.py",
              ROOT / "run_bidirectional_all.sh"]
    for f in files:
        if f.is_file():
            t = f.read_text()
            assert "joint_refine_final" not in t and "refine_joint_geometry_final" not in t, f


def test_cli_dry_run_and_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # a minimal real-format Phase-3 directory
        p3 = tmp / "forward_joint" / "revolute"
        p3.mkdir(parents=True)
        (p3 / "geometry.json").write_text('{"status": "FITTED", "joint_type": "revolute", "geometry_unchanged_bitwise": '
                                          'true, "geometry_fixed": {"axis": [0, 1, 0], "axis_point": [0, 0, 2]}}')
        (p3 / "trajectory.csv").write_text("processing_index,original_frame,q\n0,10,0.0\n1,11,0.1\n2,12,0.2\n")
        p5 = tmp / "phase5"
        p5.mkdir()
        cmd = [sys.executable, str(ROOT / "tools" / "refine_joint_geometry_final.py"), "--phase5-dir", str(p5),
               "--phase3-dir", str(tmp / "forward_joint"), "--joint-type", "revolute"]
        before = sorted(str(p) for p in tmp.rglob("*"))
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0 and "DRY_RUN" in r.stdout and "NOT RUN" in r.stdout, r.stderr
        assert sorted(str(p) for p in tmp.rglob("*")) == before, "the dry run wrote files"
        r = subprocess.run(cmd + ["--enable"], capture_output=True, text=True)
        assert r.returncode != 0 and "refusing" in r.stderr


def test_minimal_parameterisation():
    for jt in ("revolute", "prismatic"):
        j = toy_joint(jt)
        mj = MinimalJoint(j, allow_q_correction=True)
        for k, f in enumerate(j.frames):
            ft = mj.transform(f)
            Q, p = j.transform(f)
            assert np.allclose(ft.Q.detach().numpy(), Q, atol=1e-12) and np.allclose(ft.p.detach().numpy(), p, atol=1e-12)
        assert mj.dof() == (4 if jt == "revolute" else 2) + len(j.q) - 1
        assert (mj.point() is None) == (jt == "prismatic")
        with torch.no_grad():
            mj.d_e += torch.tensor([0.01, -0.02], dtype=torch.float64)
            if mj.d_a is not None:
                mj.d_a += torch.tensor([0.003, 0.004], dtype=torch.float64)
        assert abs(float(mj.axis().norm()) - 1) < 1e-12
        if mj.d_a is not None:
            assert abs(float((mj.point() - mj.a0) @ mj.e0)) < 1e-12, "axis point slid along the axis"
        assert float(mj.q(0)) == 0.0, "the anchor coordinate must stay 0"
        assert mj.within_trust(RefineConfig()) and not mj.within_trust(RefineConfig(trust_axis_deg=0.1))
        # gradients vs finite differences
        torch.manual_seed(0)
        X = torch.randn(9, 3, dtype=torch.float64)
        R = torch.nn.functional.normalize(torch.randn(9, 4, dtype=torch.float64), dim=1)
        mv = torch.arange(9) < 5
        w = torch.randn(9, 3, dtype=torch.float64)

        def L():
            mu, _ = posed(X, R, mv, mj.transform(j.frames[-1]))
            return (w * mu).sum()
        for prm in [mj.d_e] + ([mj.d_a] if mj.d_a is not None else []) + [mj.dq]:
            mj.zero_grad()
            L().backward()
            g = prm.grad.clone()
            for i in range(prm.numel()):
                with torch.no_grad():
                    prm.view(-1)[i] += 1e-6
                lp = float(L())
                with torch.no_grad():
                    prm.view(-1)[i] -= 2e-6
                lm = float(L())
                with torch.no_grad():
                    prm.view(-1)[i] += 1e-6
                assert abs((lp - lm) / 2e-6 - float(g.view(-1)[i])) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
