"""Lightweight memory probe for debugging dataloader / training memory pressure.

Stdout prints (no wandb dependency) so they appear in slurm ``.out`` logs even
when wandb is disabled or crashes. Each line prints process RSS (host RAM)
plus current torch-allocated VRAM for the calling process.

Usage::

    from pillar.utils.memdebug import print_mem
    print_mem("dataset[ct] worker pid=12345 idx=0")

Output::

    [mem] dataset[ct] worker pid=12345 idx=0 RAM=4.21 GiB VRAM=0.00 GiB

The probe is process-local — call it inside workers to see per-worker RAM,
and from the main process to see the pin-memory queue + model + activations.
"""

from __future__ import annotations

import os

import torch

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def print_mem(tag: str = "", flush: bool = True) -> None:
    """Print current host RAM (RSS) and CUDA VRAM allocated for this process."""
    parts = ["[mem]"]
    if tag:
        parts.append(tag)
    if _HAS_PSUTIL:
        try:
            rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            parts.append(f"RAM={rss_gb:.2f} GiB")
        except Exception:
            parts.append("RAM=?")
    else:
        parts.append("RAM=? (psutil missing)")
    if torch.cuda.is_available():
        try:
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            parts.append(f"VRAM={vram_gb:.2f} GiB")
        except Exception:
            pass
    print(" ".join(parts), flush=flush)
