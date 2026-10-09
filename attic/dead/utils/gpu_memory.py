"""Keep peak GPU memory down so the large scenes are runnable at all.

The prepared scenes differ by 5x in size -- 7.5k Gaussians for blade_103706,
40.6k for storage_45135 -- so code sized for the small ones simply dies on
the large ones.  Everything here is about PEAK, not total: the pipeline holds
several large intermediates alive at once (two KNN graphs, two descriptor
sets, a scene with 480 cameras, a rasteriser), and it is the overlap that
blows the budget rather than any single tensor.

None of this may change a numerical result.  A run with the scope guards
removed must produce identical output, only slower or with a higher peak.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any

import torch


def available_bytes() -> tuple[int, int]:
    """``(free, total)`` device memory, or ``(0, 0)`` on CPU."""
    if not torch.cuda.is_available():
        return 0, 0
    free, total = torch.cuda.mem_get_info()
    return int(free), int(total)


def release(*tensors: Any) -> None:
    """Drop references and return the cached blocks to the driver.

    ``empty_cache`` is what actually matters when another process shares the
    GPU: PyTorch's allocator holds freed blocks indefinitely, so without this
    a finished stage keeps memory that a concurrent run cannot use and the
    next stage OOMs against its own cache.
    """
    for tensor in tensors:
        del tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def scope(name: str = "", *, report: bool = False):
    """Free everything the block allocated when it exits.

    Used around stages that build large throwaway intermediates -- KNN
    graphs, descriptor blocks, per-camera renders -- so their peak does not
    accumulate into the next stage's budget.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        release()
        if report and torch.cuda.is_available():
            free, total = available_bytes()
            from utils.progress import write

            write(
                f"[mem] {name}: peak {peak / 2**30:.2f} GiB | "
                f"free {free / 2**30:.2f}/{total / 2**30:.2f} GiB"
            )


def suggest_chunk(
    element_bytes: int,
    *,
    fraction: float = 0.25,
    minimum: int = 64,
    maximum: int = 8192,
) -> int:
    """Rows that fit in a fraction of free memory, clamped to a sane range.

    A runtime control (§45): it changes how work is split, never what is
    computed. The fraction is deliberately conservative because the GPU is
    routinely shared with another run of this same pipeline.
    """
    free, _total = available_bytes()
    if free <= 0 or element_bytes <= 0:
        return int(maximum)
    rows = int(free * float(fraction) / int(element_bytes))
    return max(int(minimum), min(int(maximum), rows))


@contextmanager
def on_device(tensor: torch.Tensor, device: torch.device | str):
    """Move a tensor to ``device`` for the block, then put it back.

    For tensors that are large and needed only briefly: the moved copy is
    freed on exit rather than living on the GPU for the rest of the run.
    """
    original = tensor.device
    moved = tensor.to(device)
    try:
        yield moved
    finally:
        del moved
        if str(original) != str(device):
            release()


def summarise() -> dict[str, float]:
    """Current allocator state, for the probe's memory diagnostics."""
    if not torch.cuda.is_available():
        return {}
    free, total = available_bytes()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
    }
