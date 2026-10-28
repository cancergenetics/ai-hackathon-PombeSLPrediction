"""Timing harness (DESIGN.md §7): wall clock, CPU time, peak RSS, threads, hardware string.

    from src.timing import Timer, set_threads
    set_threads(8)
    with Timer("train") as t:
        ...
    t.result  # {'stage': 'train', 'wall_s':..., 'cpu_s':..., 'peak_rss_mb':..., 'threads':..., 'hardware':...}

Several stages can be collected with `Timings()` and passed to src.evaluate.evaluate(runtime=...).
Peak RSS is the process high-water mark (getrusage), i.e. the peak up to the end of the stage.
"""
from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
import time
from functools import lru_cache

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def set_threads(n: int) -> None:
    """Fix thread counts for BLAS/OpenMP (call before heavy imports where possible) and threadpoolctl."""
    for v in THREAD_VARS:
        os.environ[v] = str(n)
    os.environ["PSL_N_JOBS"] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception:  # pragma: no cover
        pass


def threads() -> int:
    return int(os.environ.get("PSL_N_JOBS") or os.environ.get("OMP_NUM_THREADS") or (os.cpu_count() or 1))


@lru_cache(maxsize=1)
def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    mem = ""
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True).stdout.strip() or cpu
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        else:
            with open("/proc/meminfo") as fh:
                mem = int(fh.readline().split()[1]) * 1024
    except Exception:  # pragma: no cover
        pass
    gb = f", {mem / 2**30:.0f} GB" if mem else ""
    return f"{cpu}, {os.cpu_count()} cores{gb}, {platform.system()} {platform.release()}, py{platform.python_version()}"


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10   # bytes on macOS, KiB on Linux


def _cpu() -> float:
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


class Timer:
    def __init__(self, stage: str = "total"):
        self.stage = stage
        self.result: dict = {}

    def __enter__(self):
        self._w, self._c = time.perf_counter(), _cpu()
        return self

    def __exit__(self, *exc):
        self.result = {"stage": self.stage, "wall_s": round(time.perf_counter() - self._w, 3),
                       "cpu_s": round(_cpu() - self._c, 3), "peak_rss_mb": round(peak_rss_mb(), 1),
                       "threads": threads(), "hardware": hardware()}
        return False


class Timings:
    """Collects several stages: `with T.stage('train'): ...`; `T.as_dict()` for evaluate(runtime=...)."""

    def __init__(self):
        self.stages: list[dict] = []

    def stage(self, name: str) -> Timer:
        t = Timer(name)
        outer = self

        class _Ctx:
            def __enter__(self_):
                return t.__enter__()

            def __exit__(self_, *exc):
                t.__exit__(*exc)
                outer.stages.append(t.result)
                return False
        return _Ctx()

    def as_dict(self) -> dict:
        return {"stages": self.stages,
                "wall_s": round(sum(s["wall_s"] for s in self.stages), 3),
                "cpu_s": round(sum(s["cpu_s"] for s in self.stages), 3),
                "peak_rss_mb": max([s["peak_rss_mb"] for s in self.stages], default=round(peak_rss_mb(), 1)),
                "threads": threads(), "hardware": hardware()}
