"""4A on every view, 4B / 4C on every k-th frame (--recovery-frame-stride), and the Phase-5 input preflight.

  stride     4A runs on the full view set and records per-view hashes (view_hashes.json, inside its hashed artifact);
             4B / 4C on every 2nd frame complete and declare recovery_views {frame_stride 2, full_sequence false};
             a recovery view that is not byte-identical to 4A's copy is refused; 4C refuses a stride different from 4B's
  preflight  Phase 5 accepts the complete 4A -> 4C chain and refuses a tampered 4C or a 4C built from another 4A
  ssim       the CUDA fused_ssim (3DGS train.py) equals the 3DGS python utils/loss_utils.ssim; view_loss uses
             utils/loss_utils.l1_loss and the declared SSIM implementation
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import phase4_artifacts as AR  # noqa: E402
from recovery import phase4a_exact as X  # noqa: E402
from recovery import phase4b_surface as B  # noqa: E402
from recovery import phase4c_validate as C  # noqa: E402
from recovery.exact_objective import ExactLossConfig  # noqa: E402
from recovery.phase4_stages import stage_4a, stage_4b, stage_4c  # noqa: E402
from phase4_ctx import QUIET, ctx_from_scene  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_phase5_rgbd_gaussians import preflight  # noqa: E402


def _refused(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def test_stride_recovery_and_phase5_preflight():
    s = make_scene("revolute", "forward", withhold_frac=0.4, seed=0)
    full = ctx_from_scene(s)
    rec = ctx_from_scene(s, stride=2)
    loss = ExactLossConfig()
    with tempfile.TemporaryDirectory() as tmp:
        ra = stage_4a(full, tmp, X.Policy4A(sensitivity=False), loss, log=QUIET)
        assert ra["complete"], ra.get("reasons")
        a_dir = Path(tmp) / AR.STAGE_A
        vh = json.loads((a_dir / "view_hashes.json").read_text())["per_view"]
        assert len(vh) == len(full.scene.views)
        # a recovery view that differs from 4A's decoded copy is refused
        bad = ctx_from_scene(s, stride=2)
        bad.pv.checksums[0] = "0" * 64
        bad.pv.manifest = (lambda m=bad.pv.manifest: lambda: {**m(), "per_view": [
            {**v, "sha256": "0" * 64} if i == 0 else v for i, v in enumerate(m()["per_view"])]})()
        assert _refused(lambda: stage_4b(bad, tmp, a_dir, B.Surface4B(), log=QUIET))
        rb = stage_4b(rec, tmp, a_dir, B.Surface4B(), log=QUIET)
        assert rb["complete"]
        b_dir = Path(tmp) / AR.STAGE_B
        rv = AR.require_complete(b_dir, AR.STAGE_B)["recovery_views"]
        assert rv["frame_stride"] == 2 and not rv["full_sequence"] and rv["views"] == len(rec.scene.views) < len(vh)
        # 4C must use the same stride as 4B
        assert _refused(lambda: stage_4c(ctx_from_scene(s), tmp, a_dir, b_dir, C.Validate4C(), loss, log=QUIET))
        rc = stage_4c(rec, tmp, a_dir, b_dir, C.Validate4C(), loss, log=QUIET)
        assert rc["complete"], rc.get("reasons")
        c_dir = Path(tmp) / AR.STAGE_C
        assert AR.require_complete(c_dir, AR.STAGE_C)["recovery_views"]["frame_stride"] == 2
        # the 4A artifact is still intact after both recovery stages
        AR.require_complete(a_dir, AR.STAGE_A)

        # Phase-5 preflight: needs the manifest setup == complete setup (procedural here) and a final canonical PLY
        pf = preflight(Path(tmp))
        assert pf["4c"]["N_final"] == len(s.model) + pf["4c"]["N_accepted_births"]
        # tampered 4C is refused
        t2 = Path(tmp) / "tampered"
        shutil.copytree(tmp, t2, ignore=shutil.ignore_patterns("tampered"))
        with open(t2 / AR.STAGE_C / "final_canonical" / "point_cloud.ply", "ab") as fh:
            fh.write(b"x")
        assert _refused(lambda: preflight(t2))
        # a 4C from another 4A run is refused
        t3 = Path(tmp) / "other4a"
        shutil.copytree(tmp, t3, ignore=shutil.ignore_patterns("tampered", "other4a"))
        shutil.rmtree(t3 / AR.STAGE_A)
        ra2 = stage_4a(ctx_from_scene(s), t3, X.Policy4A(sensitivity=False, sens_groups=2), loss, log=QUIET)
        assert ra2["complete"]
        assert _refused(lambda: preflight(t3))
    print(f"  stride: 4A on {len(full.scene.views)} views, 4B/4C on {rv['views']} (stride 2, declared); bad view and "
          f"stride mismatch refused; Phase-5 preflight accepts the chain, refuses tampered 4C and a foreign 4A")


def test_fused_ssim_matches_3dgs_python():
    from fused_ssim import fused_ssim
    from utils.loss_utils import ssim
    from recovery.rgbd_loss import LossConfig, view_loss
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.rand(3, 96, 128, device="cuda", generator=g)
    b = (a + 0.1 * torch.randn(3, 96, 128, device="cuda", generator=g)).clamp(0, 1)
    f, p = float(fused_ssim(a[None], b[None])), float(ssim(a[None], b[None]))
    assert abs(f - p) < 1e-4, (f, p)

    class V:
        rgb, mask = b, torch.ones(96, 128, dtype=torch.bool, device="cuda")
        depth = torch.ones(96, 128, device="cuda")
        depth_valid = mask
    out = {"color": a, "alpha": torch.ones(1, 96, 128, device="cuda"), "depth": torch.ones(1, 96, 128, device="cuda")}
    t1 = view_loss(out, V, LossConfig(ssim_impl="fused_ssim"))
    t2 = view_loss(out, V, LossConfig(ssim_impl="3dgs_python"))
    assert abs(t1["rgb_ssim_loss"] - (1 - f)) < 1e-6 and abs(t2["rgb_ssim_loss"] - (1 - p)) < 1e-6
    assert abs(t1["rgb_l1"] - float((a - b).abs().mean())) < 1e-7
    print(f"  ssim: fused {f:.6f} vs 3DGS python {p:.6f} (|diff| {abs(f - p):.1e}); L1 = utils/loss_utils.l1_loss")


if __name__ == "__main__":
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed.append(name)
    raise SystemExit(1 if failed else 0)
