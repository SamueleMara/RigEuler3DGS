"""tqdm bars that also work through pipes / tee / grep: every refresh is written as a COMPLETE line (newline-
terminated) instead of a \\r redraw, at most every `mininterval` seconds. Use for long loops whose output goes to
log files (the Phase-5 training stages)."""
import sys

from tqdm import tqdm


class _Lines:
    def __init__(self, f):
        self.f = f

    def write(self, s):
        s = s.replace("\r", "")
        if s.strip():
            self.f.write(s.rstrip() + "\n")
        self.f.flush()

    def flush(self):
        self.f.flush()


def line_tqdm(total, desc, mininterval=30.0, file=None, **kw):
    return tqdm(total=total, desc=desc, mininterval=mininterval, file=_Lines(file or sys.stdout), ncols=120,
                ascii=True, dynamic_ncols=False, leave=True, **kw)
