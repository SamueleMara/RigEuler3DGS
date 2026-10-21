"""Progress bars that survive output redirection.

The long runs here are redirected to a log file (`> run.log 2>&1`), which is
exactly when a progress bar matters most and exactly when the usual
`sys.stdout.isatty()` guard turns it off. Worse, writing a bar INTO the log
fills it with carriage returns.

So the bar is written to `/dev/tty` when one is attached: it appears live on
the terminal whatever stdout is doing, and the log stays clean. Without a
controlling terminal -- cron, nohup, a captured subprocess -- it degrades to a
plain iterator and costs nothing.

Set `RIGEULER_PROGRESS=0` to disable, or `=stderr` to send bars to stderr
instead (useful when the caller is already teeing stderr somewhere).

Heartbeat: when `RIGEULER_HEARTBEAT=<path>` is set (run_bidirectional_all.sh
does it per stage), every progress event -- a bar step, a note, a stage line,
an item of a wrapped loop -- touches that file, with or without a terminal.
The driver's watchdog reads its age to tell a slow stage from a hung one.
"""

from __future__ import annotations

import os
import sys

_TTY = None
_TTY_TRIED = False
_HB_LAST = 0.0


def beat() -> None:
    """Touch $RIGEULER_HEARTBEAT (at most every 2 s). Never raises."""
    global _HB_LAST
    path = os.environ.get("RIGEULER_HEARTBEAT")
    if not path:
        return
    import time
    now = time.time()
    if now - _HB_LAST < 2.0:
        return
    _HB_LAST = now
    try:
        with open(path, "a"):
            os.utime(path, None)
    except Exception:          # noqa: BLE001
        pass


def _beating(seq):
    for item in seq:
        beat()
        yield item


def _tty():
    """A handle on the controlling terminal, or None. Opened once."""
    global _TTY, _TTY_TRIED
    if _TTY_TRIED:
        return _TTY
    _TTY_TRIED = True
    try:
        _TTY = open("/dev/tty", "w")
    except Exception:          # noqa: BLE001 - no controlling terminal
        _TTY = None
    return _TTY


def progress_stream():
    mode = os.environ.get("RIGEULER_PROGRESS", "").strip().lower()
    if mode in ("0", "off", "none", "false"):
        return None
    if mode == "stderr":
        return sys.stderr
    t = _tty()
    if t is not None:
        return t
    return sys.stderr if sys.stderr.isatty() else None


def pbar(iterable, desc: str, total: int | None = None, min_items: int = 4,
         leave: bool = False):
    """tqdm when it helps, a plain iterator otherwise."""
    seq = iterable if total is not None else list(iterable)
    n = total if total is not None else len(seq)
    stream = progress_stream()
    if stream is None or n is None or n < min_items:
        return _beating(seq) if os.environ.get("RIGEULER_HEARTBEAT") else seq
    try:
        from tqdm import tqdm
    except Exception:          # noqa: BLE001
        return seq
    bar = tqdm(seq, desc=desc, total=n, leave=leave, dynamic_ncols=True,
               file=stream, mininterval=0.4, ascii=False,
               bar_format="  {desc:<28} {percentage:3.0f}%|{bar}| "
                          "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    return _beating(bar) if os.environ.get("RIGEULER_HEARTBEAT") else bar


def stage(desc: str):
    """A one-line note on the terminal for a stage with no natural loop."""
    beat()
    stream = progress_stream()
    if stream is None:
        return
    try:
        stream.write(f"  {desc}\n")
        stream.flush()
    except Exception:          # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# A bar you drive yourself.
#
# `pbar` wraps an iterable, which covers the per-frame loops. The solver loops
# inside Phase 1 and Phase 2 are different: they `break` on convergence, and
# the interesting number (energy, rho, accept/reject) only exists halfway
# through the body. Those need a handle, so they get one.
#
# Nesting is left to tqdm: a bar created while another is alive takes the next
# free line, and `leave=False` erases it on close. So the Phase-2 window bar
# stays put while each frame's solve bar appears underneath it and vanishes.
# ---------------------------------------------------------------------------

class _NullBar:
    """The no-tty case. Every method is a no-op, so callers never branch."""

    __slots__ = ()

    def to(self, n): beat()
    def update(self, n=1): beat()
    def note(self, text): beat()
    def close(self): pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Bar:
    __slots__ = ("_t",)

    def __init__(self, t):
        self._t = t

    def to(self, n):
        """Absolute position. Survives `continue` and `break`."""
        beat()
        try:
            self._t.n = int(n)
            self._t.refresh()
        except Exception:      # noqa: BLE001
            pass

    def update(self, n=1):
        beat()
        try:
            self._t.update(n)
        except Exception:      # noqa: BLE001
            pass

    def note(self, text: str):
        """The right-hand annotation: energy, rho, accept/reject.

        Truncated, because tqdm does not truncate the postfix itself: one
        caller passing a repr instead of a number wraps every redraw across
        several lines and destroys the display for every other bar.
        """
        beat()
        try:
            t = str(text)
            self._t.set_postfix_str(
                t if len(t) <= 56 else t[:53] + "...", refresh=True)
        except Exception:      # noqa: BLE001
            pass

    def close(self):
        try:
            self._t.close()
        except Exception:      # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def ibar(total: int, desc: str, min_items: int = 2, leave: bool = False):
    """A manually driven bar, or a silent stand-in with the same methods."""
    stream = progress_stream()
    if stream is None or total is None or total < min_items:
        return _NullBar()
    try:
        from tqdm import tqdm
    except Exception:          # noqa: BLE001
        return _NullBar()
    return _Bar(tqdm(total=int(total), desc=desc, leave=leave,
                     dynamic_ncols=True, file=stream, mininterval=0.2,
                     bar_format="  {desc:<28} {percentage:3.0f}%|{bar}| "
                                "{n_fmt}/{total_fmt} [{elapsed}]{postfix}"))
