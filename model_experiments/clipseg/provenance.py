"""
Run-provenance recorder.

Writes ``run_meta.json`` (and ``code.patch`` for any uncommitted edits) into an
experiment's output directory, so a results CSV can always be traced back to the
exact code / args / environment / data that produced it -- the lab-notebook entry
for a run, and the only provenance trail when the code isn't being committed.

Standalone: stdlib only. Nothing here raises out of ``RunMeta`` -- capturing
provenance must never be the thing that fails an experiment.

Usage
-----
    from provenance import RunMeta

    meta = RunMeta(args.out_dir, args, __file__)      # partial run_meta.json now
    ... run the experiment ...
    meta.finish(detail_csv=path, summary_csv=path, n_detail_rows=n)
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _run(cmd, cwd):
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        return ""


def _git_info(repo_dir, out_dir):
    info = {"available": False}
    head = _run(["git", "rev-parse", "HEAD"], repo_dir)
    if not head:
        return info
    status = _run(["git", "status", "--porcelain"], repo_dir)
    info.update(
        available=True,
        commit=head,
        branch=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir),
        describe=_run(["git", "describe", "--always", "--dirty", "--tags"], repo_dir),
        dirty=bool(status),
        untracked=[ln[3:] for ln in status.splitlines() if ln.startswith("?? ")],
    )
    # Capture every uncommitted edit (staged + unstaged vs HEAD) as a patch file
    # -- this is what makes the run reproducible without a commit.
    diff = _run(["git", "diff", "HEAD"], repo_dir)
    if diff:
        try:
            with open(os.path.join(out_dir, "code.patch"), "w") as f:
                f.write(diff + "\n")
            info["code_patch"] = "code.patch"
        except Exception:
            pass
    return info


def _pkg_versions():
    out = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
        for p in ["torch", "transformers", "sentence-transformers", "numpy",
                  "datasets", "pycocotools", "nltk", "pillow", "requests",
                  "mmcv", "mmsegmentation", "mmengine", "open-clip-torch"]:
            try:
                out[p] = version(p)
            except PackageNotFoundError:
                pass
    except Exception:
        pass
    return out


def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "cuda": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())],
            }
        return {"cuda": None}
    except Exception:
        return {}


def _slurm_info():
    keys = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_NODELIST", "SLURMD_NODENAME", "SLURM_JOB_PARTITION",
            "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS"]
    return {k: os.environ[k] for k in keys if k in os.environ}


class RunMeta:
    def __init__(self, out_dir, args, script_file, repo_dir=None):
        self.start = time.time()
        self.path = os.path.join(out_dir, "run_meta.json")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        repo_dir = repo_dir or os.path.dirname(os.path.abspath(script_file))
        try:
            self.meta = {
                "script": os.path.abspath(script_file),
                "argv": sys.argv,
                "args": {k: _jsonable(v) for k, v in vars(args).items()},
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "cwd": os.getcwd(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "git": _git_info(repo_dir, out_dir),
                "packages": _pkg_versions(),
                "gpu": _gpu_info(),
                "slurm": _slurm_info(),
                "status": "running",
            }
        except Exception as e:  # pragma: no cover - provenance must not fail a run
            self.meta = {"provenance_error": repr(e), "status": "running"}
        self._write()

    def finish(self, status="completed", **extra):
        try:
            self.meta["status"] = status
            self.meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
            self.meta["wall_seconds"] = round(time.time() - self.start, 1)
            self.meta.update({k: _jsonable(v) for k, v in extra.items()})
        except Exception as e:  # pragma: no cover
            self.meta["provenance_finish_error"] = repr(e)
        self._write()

    def _write(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.meta, f, indent=2, default=str)
        except Exception:
            pass


def count_csv_rows(path):
    """Data rows in a CSV (excludes the header). 0 if unreadable."""
    try:
        with open(path) as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0
