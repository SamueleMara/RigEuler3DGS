"""Line-buffered Phase-4 logging and resource counters (directive 2026-11-11, sections 1 and 7).

Every message goes to stdout (flushed) and to <run>/phase4.log (line-buffered, appended), stamped with the
wall time since start, process RSS and CUDA allocated / peak memory. No stage may run an hour without a
progress record: the 4A sweeps call `progress()` at most every `every_s` seconds.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from .view_cache import rss_gb


def cuda_mem():
    if not torch.cuda.is_available():
        return {}
    return {"alloc_mb": torch.cuda.memory_allocated() / 2 ** 20, "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "reserved_mb": torch.cuda.memory_reserved() / 2 ** 20, "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20}


class RunLog:
    def __init__(self, path=None, quiet=False):
        self.t0 = time.time()
        self.cpu0 = time.process_time()
        self.fh = open(path, "a", buffering=1) if path else None
        self.quiet = quiet

    def __call__(self, msg="", **_):
        m = cuda_mem()
        stamp = f"[{(time.time() - self.t0) / 60:7.2f} min | RSS {rss_gb():5.2f} GiB | CUDA {m.get('alloc_mb', 0):6.0f}/" \
                f"{m.get('peak_alloc_mb', 0):6.0f} MiB]"
        line = f"{stamp} {msg}"
        if not self.quiet:
            print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")

    def resources(self) -> dict:
        return {"wall_s": time.time() - self.t0, "cpu_s": time.process_time() - self.cpu0, "rss_gb": rss_gb(),
                **cuda_mem()}

    def close(self):
        if self.fh:
            self.fh.close()


def quiet_log(*a, **k):
    pass


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    os.replace(tmp, path)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:          # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def stdout_line_buffered():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:       # pragma: no cover
        pass
