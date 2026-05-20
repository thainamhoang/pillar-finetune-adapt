#!/usr/bin/env python3
"""Strip head + pool from a MultiStage training checkpoint, save backbone only.

Produces the encoder-only ``.pt`` consumed by
``MultimodalAtlas.pretrained_backbone_ckpt`` for the PET-from-CT path.

Usage::

    python scripts/extract_encoder.py \
        --in  logs/dual-stream-pillar/ct/dual-stream-pillar-ct/checkpoints/best.pt \
        --out logs/dual-stream-pillar/ct/dual-stream-pillar-ct/checkpoints/encoder_only.pt

Key reshaping
~~~~~~~~~~~~~

A ``MultiStage`` checkpoint stores parameters under prefixes like::

    backbone_model.<...>        <-- backbone (MultimodalAtlas)
    pool.<...>                  <-- pool head
    head_models.<name>.<...>    <-- task heads

We keep only the ``backbone_model.`` keys and strip the prefix so the
output ``.pt`` loads cleanly into a fresh ``MultimodalAtlas`` instance
via ``load_state_dict(strict=False)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _unwrap_state_dict(raw):
    """Return the parameter dict from common checkpoint formats."""
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    if isinstance(raw, dict):
        # Already a flat state dict (or close to it).
        return raw
    raise TypeError(f"Unsupported checkpoint type: {type(raw)}")


def extract(in_path: Path, out_path: Path, prefix: str = "backbone_model.") -> None:
    print(f"[extract] loading {in_path}")
    raw = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = _unwrap_state_dict(raw)
    print(f"[extract] full state_dict has {len(sd)} keys")

    backbone = {
        k[len(prefix):]: v
        for k, v in sd.items()
        if k.startswith(prefix)
    }
    other = {k: None for k in sd if not k.startswith(prefix)}

    if not backbone:
        raise RuntimeError(
            f"No keys starting with {prefix!r} found in {in_path}. "
            "Was this checkpoint produced by a MultiStage training run?"
        )

    n_params = sum(v.numel() for v in backbone.values() if torch.is_tensor(v))
    n_bytes = sum(v.numel() * v.element_size() for v in backbone.values() if torch.is_tensor(v))
    print(
        f"[extract] kept {len(backbone)} backbone keys "
        f"({n_params/1e6:.1f}M params, {n_bytes/1024**2:.1f} MiB)"
    )
    print(f"[extract] dropped {len(other)} non-backbone keys "
          f"(head/pool/etc.); first few: {list(other)[:5]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone, out_path)
    print(f"[extract] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, type=Path,
                    help="MultiStage training checkpoint (best.pt, last.pt, ...)")
    ap.add_argument("--out", dest="out_path", required=True, type=Path,
                    help="Where to write the encoder-only state_dict")
    ap.add_argument("--prefix", default="backbone_model.",
                    help="Key prefix that identifies backbone params "
                         "(default: backbone_model. — MultiStage convention)")
    args = ap.parse_args()
    extract(args.in_path, args.out_path, prefix=args.prefix)


if __name__ == "__main__":
    main()
