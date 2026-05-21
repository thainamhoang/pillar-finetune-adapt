#!/usr/bin/env python3
"""Generate a single report from a trained Phase B checkpoint.

Loads a ``DualStreamReportGenerator`` from a Lightning-style checkpoint
(``best.ckpt`` / ``epoch=N.ckpt``), pulls one sample from
``ViMedChestReportDataset`` by ``study_id``, and prints the reference and
generated English text side by side.

Usage::

    uv run python scripts/generate_report_demo.py \\
        --ckpt logs/dual-stream-pillar/report/<exp>/checkpoints/best.ckpt \\
        --manifest /scratch/thahoa/PET/ViMed_prep_v2/manifest_splits.csv \\
        --study-id <one-from-val>

If ``--study-id`` is omitted, the first val sample with a non-empty
report is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_device(spec: str) -> torch.device:
    if spec == "cuda" and not torch.cuda.is_available():
        print("[demo] CUDA unavailable; using CPU (will be slow)")
        return torch.device("cpu")
    return torch.device(spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="Phase B training checkpoint (best.ckpt / epoch=N.ckpt)")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="ViMED manifest CSV")
    ap.add_argument("--split", default="val")
    ap.add_argument("--region", default="chest")
    ap.add_argument("--study-id", default=None,
                    help="Pick this specific study_id; default = first non-empty val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"[demo] ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 2
    if not args.manifest.exists():
        print(f"[demo] ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    print(f"[demo] loading {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = state["model"] if "model" in state else state.get("state_dict", state)
    saved_args = state.get("args", None)
    if saved_args is None or not hasattr(saved_args, "model"):
        print("[demo] ERROR: checkpoint has no `args` -- can't rebuild model",
              file=sys.stderr)
        return 2

    # Rebuild model from saved args.
    from pillar.models import DualStreamReportGenerator

    model_kwargs = dict(saved_args.model.kwargs)
    # Avoid re-loading the Phase A encoder file -- the saved ckpt already
    # contains the encoder state. Set encoder_ckpt=None so the constructor
    # doesn't try to overlay from disk again.
    model_kwargs["encoder_ckpt"] = None
    model = DualStreamReportGenerator(args=saved_args, **model_kwargs)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"[demo] loaded state_dict: {len(missing)} missing, {len(unexpected)} unexpected")

    model = model.to(device)
    model.eval()

    # Load one sample.
    from pillar.datasets.vimed_chest_report import ViMedChestReportDataset

    ds_kwargs = dict(saved_args.dataset.shared_dataset_kwargs)
    ds_kwargs["csv_path"] = str(args.manifest)  # respect CLI override
    ds_kwargs["split_group"] = args.split
    ds_kwargs["region"] = args.region
    ds = ViMedChestReportDataset(**ds_kwargs)

    if len(ds) == 0:
        print(f"[demo] ERROR: empty dataset for split={args.split} region={args.region}",
              file=sys.stderr)
        return 2

    if args.study_id is None:
        for i in range(len(ds)):
            item = ds[i]
            if item is not None:
                args.study_id = item["study_id"]
                break
        if args.study_id is None:
            print("[demo] ERROR: no non-empty samples found", file=sys.stderr)
            return 2

    # Find the matching row.
    target_item = None
    for i in range(len(ds)):
        item = ds[i]
        if item is not None and item["study_id"] == args.study_id:
            target_item = item
            break
    if target_item is None:
        print(f"[demo] ERROR: study_id {args.study_id!r} not found", file=sys.stderr)
        return 2

    # Build a batch of size 1.
    batch = {}
    for k, v in target_item.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = [v]

    with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                            enabled=device.type == "cuda", dtype=torch.bfloat16):
        gens = model.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

    print()
    print("=" * 70)
    print(f"Study ID: {args.study_id}")
    print("=" * 70)
    print("\n--- Reference (Gemma-4 translated English) ---\n")
    print(target_item["report_text"] or "[empty]")
    print("\n--- Generated ---\n")
    print(gens[0])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
