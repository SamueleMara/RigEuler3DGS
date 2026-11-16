"""One decode per source view for Phase 4 (directive 2026-11-11, section 3).

The pre-correction ViewStore kept a 64-entry LRU in front of a 378-view round-robin sweep, so every access
missed and re-decoded PNG + npy + mask (measured on red_fridge: 3 TB page-cache reads in 13 h, 11.4 ms per
decode against a 0.95 ms render). PreparedViews reads every source file exactly ONCE (raw bytes -> sha256 ->
decode from memory) into compact host arrays:

    rgb    uint8   [H,W,3]
    depth  float32 [H,W]   metres, camera axial; NaN / inf / <= 0 stay invalid (never reduced to float16)
    mask   bool    [H,W]   strict binary object mask (estimator.masked_observation rules)
    valid  bool    [H,W]   mask & finite & depth > 0  (the ONLY depth observations; M = 0 is no depth)

Memory: 378 x 640 x 480 x (3 + 4 + 1 + 1) B = 1.04 GiB for the red_fridge sequence. The store refuses to
hold more than `max_host_gb` in RAM and then unpacks into a hash-named memory map under `spill_dir`
instead (same arrays, same checksums). Views go to the GPU one at a time (`gpu_view`); nothing keeps 378
full-resolution float tensors on the 8-GB card.

Counters (asserted by tests/test_phase4_view_cache.py, written to cache_manifest.json): view_get,
gpu_uploads, files_opened, image_decodes, npy_loads, mask_decodes, bytes_read, decode_seconds.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .views import View


@dataclass
class DecodeCounters:
    view_get: int = 0
    gpu_uploads: int = 0
    gpu_upload_bytes: int = 0
    files_opened: int = 0
    image_decodes: int = 0
    npy_loads: int = 0
    mask_decodes: int = 0
    bytes_read: int = 0
    decode_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


def _find(source: Path, sub: str, cam: str, frame: int, exts) -> Path:
    hits = [source / sub / cam / f"frame_{int(frame):06d}{e}" for e in exts]
    hits = [h for h in hits if h.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(f"{sub}/{cam}/frame_{int(frame):06d}: {len(hits)} candidates")
    return hits[0]


def _decode_mask(buf: bytes, path: Path, W: int, H: int) -> np.ndarray:
    """estimator.masked_observation.load_object_mask rules, from bytes already read."""
    im = Image.open(io.BytesIO(buf))
    if im.size != (W, H):
        raise ValueError(f"{path}: mask {im.size} but the camera image is {(W, H)}")
    a = np.asarray(im)
    if a.ndim == 3:
        if not all(np.array_equal(a[..., 0], a[..., k]) for k in range(1, min(a.shape[2], 3))):
            raise ValueError(f"{path}: colour mask with differing channels is not a binary mask")
        a = a[..., 0]
    vals = set(np.unique(a).tolist())
    if not (vals <= {0, 255} or vals <= {0, 1}):
        raise ValueError(f"{path}: non-binary mask values -- refusing to threshold")
    return a > 0


def _rgb_float(t: torch.Tensor) -> torch.Tensor:
    """[H,W,3] uint8 (0..255) or float32 (0..1) -> [3,H,W] float32 in [0,1]."""
    t = t.permute(2, 0, 1)
    return t.float() / 255.0 if t.dtype == torch.uint8 else t.float()


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


def rss_gb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


@dataclass
class ViewKey:
    camera: str
    frame: int
    W: int
    H: int


class PreparedViews:
    """All (camera, original frame) views of one Phase-4 run, decoded once.

    keys: ordered list of ViewKey (the processing order of the stage). Index i is the view id everywhere in
    Phase 4 (scores, checkpoints, logs)."""

    def __init__(self, source, keys, max_host_gb: float = 8.0, spill_dir=None, log=print):
        self.source = Path(source)
        self.keys = list(keys)
        self.counters = DecodeCounters()
        self.max_host_gb, self.spill_dir, self.log = float(max_host_gb), spill_dir, log
        self.rgb, self.depth, self.mask, self.valid = [], [], [], []
        self.checksums = []
        self.prepared = False
        self.storage = None
        self._n_valid = None

    # ---------------------------------------------------------------- preparation
    def bytes_required(self) -> int:
        return sum(k.W * k.H * (3 + 4 + 1 + 1) for k in self.keys)

    def _read(self, path: Path) -> bytes:
        with open(path, "rb") as fh:
            b = fh.read()
        self.counters.files_opened += 1
        self.counters.bytes_read += len(b)
        return b

    def _decode_one(self, k: ViewKey):
        t0 = time.perf_counter()
        prgb = _find(self.source, "rgb", k.camera, k.frame, (".png", ".jpg", ".jpeg"))
        pdep = _find(self.source, "depth_npy", k.camera, k.frame, (".npy",))
        pmsk = _find(self.source, "mask", k.camera, k.frame, (".png", ".jpg", ".jpeg"))
        brgb, bdep, bmsk = self._read(prgb), self._read(pdep), self._read(pmsk)
        h = hashlib.sha256()
        for b in (brgb, bdep, bmsk):
            h.update(hashlib.sha256(b).digest())
        im = Image.open(io.BytesIO(brgb)).convert("RGB")
        self.counters.image_decodes += 1
        if im.size != (k.W, k.H):
            raise ValueError(f"{k.camera} frame {k.frame}: RGB {im.size} vs camera {k.W}x{k.H}")
        rgb = np.array(im, dtype=np.uint8)           # writable copy
        d = np.load(io.BytesIO(bdep))
        self.counters.npy_loads += 1
        if d.shape != (k.H, k.W):
            raise ValueError(f"{k.camera} frame {k.frame}: depth {d.shape} vs camera {k.H}x{k.W}")
        d = d.astype(np.float32, copy=False)
        m = _decode_mask(bmsk, pmsk, k.W, k.H)
        self.counters.mask_decodes += 1
        valid = m & np.isfinite(d) & (d > 0)
        d = np.where(np.isfinite(d), d, 0.0).astype(np.float32)
        self.counters.decode_seconds += time.perf_counter() - t0
        return rgb, d, m, valid, h.hexdigest()

    def prepare(self):
        """Decode every view once. Idempotent: a second call decodes nothing."""
        if self.prepared:
            return self.manifest()
        need_gb = self.bytes_required() / 2 ** 30
        avail = mem_available_gb()
        in_ram = need_gb <= self.max_host_gb and (not np.isfinite(avail) or need_gb <= 0.5 * avail)
        self.storage = "host_ram" if in_ram else "memmap"
        self.log(f"  views: preparing {len(self.keys)} views, {need_gb:.2f} GiB unpacked, MemAvailable "
                 f"{avail:.1f} GiB -> {self.storage}", flush=True)
        if not in_ram:
            if self.spill_dir is None:
                raise MemoryError(f"{need_gb:.2f} GiB of views exceed max_host_gb={self.max_host_gb} and no "
                                  "spill_dir was given for the memory-mapped store")
            Path(self.spill_dir).mkdir(parents=True, exist_ok=True)
        # Spill: ONE flat file per array kind (4 file descriptors in total, whatever the view count). Python's
        # mmap dup()s the descriptor of every map it opens, so one map per view and kind exhausted the default
        # `ulimit -n` (1024) at ~256 views.
        spill = None
        if not in_ram:
            spill = {}
            kinds = (("rgb", np.uint8, lambda k: (k.H, k.W, 3)), ("depth", np.float32, lambda k: (k.H, k.W)),
                     ("mask", np.bool_, lambda k: (k.H, k.W)), ("valid", np.bool_, lambda k: (k.H, k.W)))
            for name, dt, shp in kinds:
                offs, n = [], 0
                for k in self.keys:
                    offs.append(n)
                    n += int(np.prod(shp(k)))
                path = Path(self.spill_dir) / f"views_{len(self.keys)}_{name}.bin"
                spill[name] = (np.memmap(path, dtype=dt, mode="w+", shape=(max(n, 1),)), offs, shp)
        t0 = time.time()
        last = t0
        for i, k in enumerate(self.keys):
            rgb, d, m, valid, cs = self._decode_one(k)
            if spill is not None:
                arrs = []
                for name, a in (("rgb", rgb), ("depth", d), ("mask", m), ("valid", valid)):
                    mm, offs, shp = spill[name]
                    sl = mm[offs[i]:offs[i] + a.size].reshape(shp(k))
                    sl[...] = a
                    arrs.append(sl)
                rgb, d, m, valid = arrs
            self.rgb.append(rgb); self.depth.append(d); self.mask.append(m); self.valid.append(valid)
            self.checksums.append(cs)
            if time.time() - last > 30:
                last = time.time()
                self.log(f"    decoded {i + 1}/{len(self.keys)} views ({(i + 1) / (last - t0):.1f}/s, "
                         f"RSS {rss_gb():.2f} GiB)", flush=True)
        if spill is not None:
            for mm, _, _ in spill.values():
                mm.flush()
        self.prepared = True
        self._n_valid = np.array([int(v.sum()) for v in self.valid])
        return self.manifest()

    @classmethod
    def from_memory(cls, keys, views, log=print):
        """Test / procedural scenes: views already in memory (views.View on the CPU, float RGB kept as float32 so
        procedural observations are not quantised). Counted as one 'decode' per view."""
        self = cls(source="<memory>", keys=keys, log=log)
        for k, v in zip(keys, views):
            self.rgb.append(v.rgb.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32))
            self.depth.append(v.depth.cpu().numpy().astype(np.float32))
            self.mask.append(v.mask.cpu().numpy().astype(bool))
            self.valid.append(v.depth_valid.cpu().numpy().astype(bool))
            self.checksums.append(hashlib.sha256(self.rgb[-1].tobytes() + self.depth[-1].tobytes() +
                                                 self.mask[-1].tobytes()).hexdigest())
            self.counters.image_decodes += 1
        self.prepared, self.storage = True, "memory"
        self._n_valid = np.array([int(v.sum()) for v in self.valid])
        return self

    def manifest(self) -> dict:
        allh = hashlib.sha256("".join(self.checksums).encode()).hexdigest() if self.checksums else None
        return {"source": str(self.source), "views": len(self.keys), "storage": self.storage,
                "bytes_unpacked": self.bytes_required(), "rss_gb_after_prepare": rss_gb(),
                "counters": self.counters.to_dict(), "combined_sha256": allh,
                "per_view": [{"camera": k.camera, "frame": k.frame, "sha256": c}
                             for k, c in zip(self.keys, self.checksums)],
                "valid_depth_pixels_total": int(self._n_valid.sum()) if self._n_valid is not None else None}

    # ---------------------------------------------------------------- access (never decodes)
    def __len__(self):
        return len(self.keys)

    def _check(self, i):
        if not self.prepared:
            raise RuntimeError("PreparedViews.prepare() was not called; refusing to decode lazily")
        self.counters.view_get += 1

    def cpu_view(self, i) -> View:
        """CPU View with float RGB (for the few CPU consumers: 4B back-projection)."""
        self._check(i)
        k = self.keys[i]
        return View(k.camera, k.frame, _rgb_float(torch.from_numpy(np.ascontiguousarray(self.rgb[i]))),
                    torch.from_numpy(np.ascontiguousarray(self.depth[i])), torch.from_numpy(np.ascontiguousarray(self.mask[i])),
                    torch.from_numpy(np.ascontiguousarray(self.valid[i])))

    def gpu_view(self, i, device="cuda") -> View:
        """Upload view i (compact uint8 / float32 / bool -> float on the device)."""
        self._check(i)
        k = self.keys[i]
        rgb = torch.from_numpy(np.ascontiguousarray(self.rgb[i])).to(device, non_blocking=True)
        dep = torch.from_numpy(np.ascontiguousarray(self.depth[i])).to(device, non_blocking=True)
        msk = torch.from_numpy(np.ascontiguousarray(self.mask[i])).to(device, non_blocking=True)
        val = torch.from_numpy(np.ascontiguousarray(self.valid[i])).to(device, non_blocking=True)
        self.counters.gpu_uploads += 1
        self.counters.gpu_upload_bytes += k.W * k.H * 9
        return View(k.camera, k.frame, _rgb_float(rgb), dep, msk, val)

    def n_valid(self, i) -> int:
        return int(self._n_valid[i])


def write_cache_manifest(path, views: PreparedViews, extra=None):
    d = views.manifest() | (extra or {})
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, path)
    return d
