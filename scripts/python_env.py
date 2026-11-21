"""Launch project Python using the virtualenv's bundled OpenMP on macOS.

Usage: .venv/bin/python scripts/python_env.py -m pytest -q
No system libraries are installed or modified. Other platforms execute normally.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Supply Python arguments, e.g. -m pytest -q")
    environment = os.environ.copy()
    if sys.platform == "darwin":
        spec = importlib.util.find_spec("sklearn")
        if spec and spec.origin:
            directory = Path(spec.origin).parent / ".dylibs"
            if (directory / "libomp.dylib").is_file():
                current = environment.get("DYLD_LIBRARY_PATH", "")
                environment["DYLD_LIBRARY_PATH"] = str(directory) + (os.pathsep + current if current else "")
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("OPENBLAS_NUM_THREADS", "2")
    environment.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    environment.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results/.matplotlib"))
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()
