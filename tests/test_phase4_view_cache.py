"""Gate A (directive 2026-11-11, section 3): one decode per source view.

A 378-view sequence (2 cameras x 189 frames, small images, written to disk in the dataset layout) is
prepared once and then swept sequentially three more times:
  * PreparedViews: exactly 378 RGB decodes, 378 npy loads, 378 mask decodes, 1134 files opened -- and no
    further decode or file read during the sweeps; uploads counted per access
  * the pre-correction ViewStore(cache=64) on the same sweeps misses on EVERY access (the thrash)
  * the memory-mapped fallback (host budget below the need) returns bit-identical arrays and checksums
  * lazy access before prepare() is refused (no silent decode path)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recovery  # noqa: E402,F401
from recovery.view_cache import PreparedViews, ViewKey  # noqa: E402
from recovery.views import RenderCamera, ViewStore  # noqa: E402

W, H, CAMS, FRAMES = 16, 12, ("cam_001", "cam_002"), 189


def _dataset(root: Path):
    rng = np.random.default_rng(0)
    for c in CAMS:
        for sub in ("rgb", "depth_npy", "mask"):
            (root / sub / c).mkdir(parents=True, exist_ok=True)
        for f in range(FRAMES):
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(root / "rgb" / c / f"frame_{f:06d}.png")
            d = rng.uniform(0.5, 2.0, (H, W)).astype(np.float32)
            d[0, 0] = np.nan
            np.save(root / "depth_npy" / c / f"frame_{f:06d}.npy", d)
            m = np.zeros((H, W), np.uint8); m[2:10, 3:12] = 255
            Image.fromarray(m).save(root / "mask" / c / f"frame_{f:06d}.png")


def test_one_decode_per_view_and_thrash_gone():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _dataset(root)
        keys = [ViewKey(c, f, W, H) for c in CAMS for f in range(FRAMES)]
        pv = PreparedViews(root, keys, log=lambda *a, **k: None)
        try:
            pv.gpu_view(0)
            raise AssertionError("access before prepare() must be refused")
        except RuntimeError:
            pass
        pv.prepare()
        c0 = dict(pv.counters.to_dict())
        assert c0["image_decodes"] == c0["npy_loads"] == c0["mask_decodes"] == 378 and c0["files_opened"] == 1134
        for _ in range(3):
            for i in range(len(pv)):
                v = pv.gpu_view(i)
        c1 = pv.counters.to_dict()
        for k in ("image_decodes", "npy_loads", "mask_decodes", "files_opened", "bytes_read"):
            assert c1[k] == c0[k], (k, c0[k], c1[k])
        assert c1["gpu_uploads"] == 3 * 378
        assert not bool(v.depth_valid[0, 0]) and v.depth.dtype == torch.float32
        cams = {c: RenderCamera(c, W, H, torch.eye(4), torch.eye(4), torch.zeros(3), 1.0, 1.0) for c in CAMS}
        st = ViewStore(root, cams, cache=64)
        for _ in range(3):
            for k in keys:
                st.get(k.camera, k.frame)
        assert st.counters["hits"] == 0 and st.counters["decodes"] == 3 * 378, st.counters
        # memory-mapped fallback: identical arrays and checksums
        pm = PreparedViews(root, keys, max_host_gb=1e-9, spill_dir=root / "spill", log=lambda *a, **k: None)
        pm.prepare()
        assert pm.storage == "memmap" and pm.checksums == pv.checksums
        for i in range(len(pv)):
            for a, b in ((pm.rgb, pv.rgb), (pm.depth, pv.depth), (pm.mask, pv.mask), (pm.valid, pv.valid)):
                assert np.array_equal(np.asarray(a[i]), b[i]), i
        spill_files = sorted(p.name for p in (root / "spill").iterdir())
        assert len(spill_files) == 4, spill_files      # one file per array kind, never one per view
        del pm
        print(f"  378 views: PreparedViews {c0['image_decodes']} decodes / {c0['bytes_read']} B once, 0 more over 3 "
              f"sweeps; old ViewStore(64): {st.counters['decodes']} decodes, {st.counters['hits']} hits; memmap identical")


if __name__ == "__main__":
    test_one_decode_per_view_and_thrash_gone()
    print("PASS test_one_decode_per_view_and_thrash_gone")
