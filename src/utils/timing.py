from __future__ import annotations

import time
from contextlib import contextmanager

@contextmanager
def timer(label: str = ""):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"{label}: {elapsed:.3f}s" if label else f"{elapsed:.3f}s")

def format_seconds(s: float) -> str:
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"
