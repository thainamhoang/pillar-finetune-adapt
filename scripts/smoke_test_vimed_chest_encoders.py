#!/usr/bin/env python3
"""Smoke-test Pillar-initialized CT/PET Atlas encoder wrappers on ViMED chest data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pillar.datasets.vimed_chest_report import ViMedChestReportDataset
from pillar.models.backbones.dual_stream_atlas import ViMedChestDualStreamEncoders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-repo-id", default="YalaLab/Pillar0-ChestCT")
    p.add_argument("--model-revision", default="main")
    return p.parse_args()


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float().cpu()
    print(f"{name}_shape={tuple(tensor.shape)}")
    print(f"{name}_min={float(tensor.min()):.6f}")
    print(f"{name}_max={float(tensor.max()):.6f}")


def main() -> None:
    args = parse_args()

    ds = ViMedChestReportDataset(args.manifest, split=args.split, include_raw=True)
    if len(ds) == 0:
        raise SystemExit(f"No chest rows found for split={args.split!r} in {args.manifest}")

    sample = ds[args.index]
    ct_windows = sample["ct_windows"].unsqueeze(0).to(args.device)
    pet_windows = sample["pet_windows"].unsqueeze(0).to(args.device)

    print(f"dataset_len={len(ds)}")
    print(f"study_id={sample['study_id']}")
    print(f"region={sample['region']}")
    summarize_tensor("ct_windows", sample["ct_windows"])
    summarize_tensor("pet_windows", sample["pet_windows"])
    summarize_tensor("x_raw", sample["x_raw"])

    model = ViMedChestDualStreamEncoders(
        ct_channels=ct_windows.shape[1],
        pet_channels=pet_windows.shape[1],
        device=args.device,
        model_repo_id=args.model_repo_id,
        model_revision=args.model_revision,
    ).to(args.device)
    model.eval()

    with torch.no_grad():
        outputs = model(ct_windows, pet_windows)

    print(f"ct_input_adapter={model.ct_encoder.input_adapter_name}")
    print(f"pet_input_adapter={model.pet_encoder.input_adapter_name}")
    summarize_tensor("ct_activ", outputs["ct_activ"])
    summarize_tensor("ct_pooled", outputs["ct_pooled"])
    summarize_tensor("pet_activ", outputs["pet_activ"])
    summarize_tensor("pet_pooled", outputs["pet_pooled"])
    print("forward_ok=1")


if __name__ == "__main__":
    main()
