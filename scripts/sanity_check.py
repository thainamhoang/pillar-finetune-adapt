#!/usr/bin/env python3
"""Sanity-check the pillar-finetune-adapt training pipeline.

Verifies, in order, with PASS / WARN / FAIL signals:

1. Manifest CSV health — splits, label prevalences, patient leakage
2. Sample tensor file integrity — shape, dtype, value ranges, labels
3. Dataset class output — both ``channels_mode=ct`` and ``channels_mode=pet``
4. Model forward pass (optional, requires GPU) — input/output shapes
5. D-axis identification — confirms which axis of ``activ`` is depth
6. Conv3d patch-embed adapter — first conv ``in_channels`` matches modality

Usage::

    # CPU-only checks (1-3, 6 static)
    python scripts/sanity_check.py configs/vimed_chest_ct_only.yaml

    # All checks including GPU forward pass
    python scripts/sanity_check.py configs/vimed_chest_ct_only.yaml --with-forward

Returns nonzero exit code if any FAIL is recorded.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# Add project root to path so `pillar` imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml


# ----- pretty printing -----

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

_results: list[tuple[str, str, str]] = []  # (level, section, message)


def _emit(level: str, color: str, section: str, message: str) -> None:
    tag = f"{color}[{level}]{RESET}"
    print(f"{tag} {BOLD}{section}{RESET} — {message}")
    _results.append((level, section, message))


def passed(section: str, message: str) -> None:
    _emit("PASS", GREEN, section, message)


def warned(section: str, message: str) -> None:
    _emit("WARN", YELLOW, section, message)


def failed(section: str, message: str) -> None:
    _emit("FAIL", RED, section, message)


def info(section: str, message: str) -> None:
    print(f"{CYAN}[INFO]{RESET} {BOLD}{section}{RESET} — {message}")


def section_header(title: str) -> None:
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")


# ----- config helpers -----

def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def get(d: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----- 1. Manifest CSV -----

def check_manifest(config: dict) -> None:
    section_header("1. Manifest CSV health")

    csv_path = Path(get(config, "dataset", "shared_dataset_kwargs", "csv_path", default=""))
    region = get(config, "dataset", "shared_dataset_kwargs", "region", default="chest")
    label_columns = get(config, "dataset", "shared_dataset_kwargs", "label_columns", default=[])

    if not csv_path.exists():
        failed("manifest", f"CSV not found: {csv_path}")
        return
    passed("manifest", f"Found {csv_path}")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    info("manifest", f"Total rows in CSV: {len(rows)}")

    region_rows = [r for r in rows if r.get("region") == region]
    info("manifest", f"Rows with region={region!r}: {len(region_rows)}")
    if not region_rows:
        failed("manifest", f"No rows match region={region!r}")
        return

    # Split counts
    split_counts = Counter(r.get("split", "<missing>") for r in region_rows)
    info("manifest", f"Split counts: {dict(split_counts)}")
    if "train" not in split_counts or split_counts["train"] == 0:
        failed("manifest", "No train rows in this region")
    if "val" not in split_counts or split_counts["val"] == 0:
        warned("manifest", "No val rows — best.pt tracking won't work")

    # Patient-level leakage between train and val
    def _patient_id(study_id: str) -> str:
        # study_id pattern observed: e.g. "000016_patient_18_2017"
        parts = study_id.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else study_id

    train_patients = {_patient_id(r["study_id"]) for r in region_rows if r.get("split") == "train"}
    val_patients = {_patient_id(r["study_id"]) for r in region_rows if r.get("split") == "val"}
    test_patients = {_patient_id(r["study_id"]) for r in region_rows if r.get("split") == "test"}

    tv_leak = train_patients & val_patients
    tt_leak = train_patients & test_patients
    if tv_leak:
        failed("manifest", f"{len(tv_leak)} patient(s) appear in BOTH train and val — split leakage")
        for pid in list(tv_leak)[:5]:
            print(f"        - {pid}")
    else:
        passed("manifest", f"No train/val patient leakage ({len(train_patients)} train pts, {len(val_patients)} val pts)")

    if tt_leak:
        warned("manifest", f"{len(tt_leak)} patient(s) appear in BOTH train and test")

    # Label prevalence per split
    if label_columns:
        info("manifest", f"Checking {len(label_columns)} label columns for prevalence skew")
        prevalences: dict[str, dict[str, float]] = defaultdict(dict)
        for split in ("train", "val", "test"):
            sub = [r for r in region_rows if r.get("split") == split]
            if not sub:
                continue
            for col in label_columns:
                if col not in sub[0]:
                    continue
                vals = []
                for r in sub:
                    raw = r.get(col, "")
                    try:
                        vals.append(float(raw))
                    except (ValueError, TypeError):
                        pass
                if vals:
                    prevalences[col][split] = sum(vals) / len(vals)

        # Pretty print
        print(f"\n        {'Label':<35} {'train':>8} {'val':>8} {'test':>8}  Δ(train-val)")
        max_delta = 0.0
        for col in label_columns:
            tr = prevalences.get(col, {}).get("train")
            va = prevalences.get(col, {}).get("val")
            te = prevalences.get(col, {}).get("test")
            if tr is None or va is None:
                print(f"        {col:<35} {'-':>8} {'-':>8} {'-':>8}")
                continue
            delta = abs(tr - va)
            max_delta = max(max_delta, delta)
            print(
                f"        {col:<35} {tr:>8.3f} {va:>8.3f} "
                f"{(te if te is not None else float('nan')):>8.3f}  {delta:+.3f}"
            )

        if max_delta > 0.05:
            warned("manifest", f"Max train↔val prevalence Δ = {max_delta:.3f} (>5%) — val may not represent train")
        else:
            passed("manifest", f"All train↔val prevalence Δ ≤ 5% (max {max_delta:.3f})")


# ----- 2. Sample tensor file -----

def check_sample_tensor(config: dict) -> Optional[dict]:
    section_header("2. Sample tensor file integrity")

    csv_path = Path(get(config, "dataset", "shared_dataset_kwargs", "csv_path", default=""))
    region = get(config, "dataset", "shared_dataset_kwargs", "region", default="chest")
    label_columns = get(config, "dataset", "shared_dataset_kwargs", "label_columns", default=[])

    if not csv_path.exists():
        failed("tensor", "Cannot check — manifest missing")
        return None

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r.get("region") == region and r.get("split") == "train"]
    if not train_rows:
        failed("tensor", "No train rows to sample from")
        return None

    sample_row = train_rows[0]
    tensor_path = Path(sample_row.get("tensor_path", ""))
    if not tensor_path.exists():
        failed("tensor", f"First train sample .pt missing: {tensor_path}")
        return None
    passed("tensor", f"Loading {tensor_path}")

    try:
        item = torch.load(tensor_path, map_location="cpu", weights_only=False)
    except Exception as e:
        failed("tensor", f"torch.load failed: {e}")
        return None

    if not isinstance(item, dict) or "x_raw" not in item:
        failed("tensor", f"Sample has no 'x_raw' key (got: {list(item.keys()) if isinstance(item, dict) else type(item)})")
        return None

    x_raw = item["x_raw"]
    info("tensor", f"x_raw shape={tuple(x_raw.shape)} dtype={x_raw.dtype}")

    if x_raw.ndim != 4 or x_raw.shape[0] != 2:
        failed("tensor", f"x_raw expected shape (2, D, H, W), got {tuple(x_raw.shape)}")
    elif x_raw.shape[2:] != (256, 256):
        warned("tensor", f"x_raw H,W = {tuple(x_raw.shape[2:])}, expected (256, 256)")
    else:
        passed("tensor", f"x_raw shape is (2, {x_raw.shape[1]}, 256, 256)")

    ct = x_raw[0].float()
    pet = x_raw[1].float()
    info("tensor", f"CT  (ch 0) range: [{ct.min():.1f}, {ct.max():.1f}]  mean={ct.mean():.1f}")
    info("tensor", f"PET (ch 1) range: [{pet.min():.3f}, {pet.max():.3f}]  mean={pet.mean():.3f}")

    # CT should be HU-like (-1024 to ~3000)
    if ct.min() < -1500 or ct.max() > 5000:
        warned("tensor", f"CT range [{ct.min():.1f}, {ct.max():.1f}] is outside expected HU range")
    elif ct.max() <= 1.0 and ct.min() >= 0.0:
        failed("tensor", "CT appears already normalized to [0,1] — windowing would be applied to wrong scale")
    else:
        passed("tensor", "CT values look like HU (Hounsfield Units)")

    # PET should be normalized [0, 1]
    if pet.min() < -0.05 or pet.max() > 1.05:
        warned("tensor", f"PET range [{pet.min():.3f}, {pet.max():.3f}] is outside expected [0, 1]")
    else:
        passed("tensor", "PET values are in expected [0, 1] range")

    # Labels
    if "labels" in item:
        labels = item["labels"]
        info("tensor", f"labels shape={tuple(labels.shape)} dtype={labels.dtype}")
        if label_columns and labels.numel() != len(label_columns):
            warned("tensor", f"labels has {labels.numel()} entries, config expects {len(label_columns)}")
        else:
            passed("tensor", "labels tensor present with expected length")
    elif label_columns:
        warned("tensor", "No 'labels' key inside .pt — dataset will need to source them from CSV columns")

    return item


# ----- 3. Dataset class output -----

def check_dataset(config: dict) -> None:
    section_header("3. Dataset class output (CT + PET modes)")

    try:
        from pillar.datasets import ViMedChestSingleModalityDataset
    except Exception as e:
        failed("dataset", f"Import failed: {e}")
        return

    shared = dict(get(config, "dataset", "shared_dataset_kwargs", default={}) or {})
    expected_channels = {"ct": 6, "pet": 4}
    expected_label_len = len(get(config, "dataset", "shared_dataset_kwargs", "label_columns", default=[]) or [])

    for mode, n_channels in expected_channels.items():
        kwargs = dict(shared)
        kwargs["channels_mode"] = mode
        try:
            ds = ViMedChestSingleModalityDataset(
                args=None,
                augmentations=None,
                split_group="train",
                **kwargs,
            )
        except Exception as e:
            failed(f"dataset[{mode}]", f"Instantiation failed: {e}")
            continue

        if len(ds) == 0:
            failed(f"dataset[{mode}]", "Dataset is empty after split/region filter")
            continue
        info(f"dataset[{mode}]", f"len={len(ds)}")

        try:
            item = ds[0]
        except Exception as e:
            failed(f"dataset[{mode}]", f"__getitem__ raised: {e}")
            traceback.print_exc()
            continue

        # Required keys
        required = {"x", "y", "mask", "image_annotations", "has_annotation",
                    "accession", "anatomy"}
        missing = required - set(item.keys())
        if missing:
            failed(f"dataset[{mode}]", f"Output missing keys: {missing}")
        else:
            passed(f"dataset[{mode}]", "Output has all engine-required keys")

        # x shape and range
        x = item["x"]
        info(f"dataset[{mode}]", f"x shape={tuple(x.shape)} dtype={x.dtype}")
        if x.shape[0] != n_channels:
            failed(f"dataset[{mode}]",
                   f"x channels = {x.shape[0]}, expected {n_channels}")
        elif x.ndim != 4:
            failed(f"dataset[{mode}]", f"x ndim = {x.ndim}, expected 4")
        else:
            passed(f"dataset[{mode}]", f"x shape correct ({n_channels}, D, H, W)")

        xmin, xmax = float(x.min()), float(x.max())
        info(f"dataset[{mode}]", f"x range: [{xmin:.4f}, {xmax:.4f}]")
        if xmin < -1e-4 or xmax > 1.0 + 1e-4:
            failed(f"dataset[{mode}]",
                   f"x out of [0,1] — windowing produced values outside expected range")
        else:
            passed(f"dataset[{mode}]", "x values are in [0, 1]")

        # Labels
        y = item["y"]
        if isinstance(y, torch.Tensor):
            info(f"dataset[{mode}]", f"y shape={tuple(y.shape)} dtype={y.dtype} values={y.tolist()}")
            if expected_label_len and y.numel() != expected_label_len:
                warned(f"dataset[{mode}]",
                       f"y has {y.numel()} elements, expected {expected_label_len}")
            else:
                passed(f"dataset[{mode}]", "y length matches label_columns")
        else:
            warned(f"dataset[{mode}]", f"y is not a tensor (got {type(y).__name__})")

        # Mask shape parity
        mask = item["mask"]
        if mask.shape[-3:] != x.shape[-3:]:
            warned(f"dataset[{mode}]",
                   f"mask spatial shape {tuple(mask.shape[-3:])} != x {tuple(x.shape[-3:])}")
        else:
            passed(f"dataset[{mode}]", "mask spatial shape matches x")


# ----- 4. Model forward pass (optional, GPU) -----

def check_forward(config: dict, device: str) -> None:
    section_header("4. Model forward pass (GPU)")

    try:
        from pillar.models.backbones.dual_stream_atlas import (
            PillarInitializedAtlasEncoder,
            EncoderConfig,
        )
    except Exception as e:
        failed("forward", f"Import failed: {e}")
        return

    channels_mode = get(config, "dataset", "shared_dataset_kwargs", "channels_mode", default="ct")
    n_channels = 6 if channels_mode == "ct" else 4

    try:
        encoder = PillarInitializedAtlasEncoder(
            EncoderConfig(
                input_channels=n_channels,
                anatomy="chest_ct",
                model_repo_id="YalaLab/Pillar0-ChestCT",
                model_revision="main",
                device=device,
                patch_embed_init="kaiming",
            )
        ).to(device)
        encoder.eval()
        passed("forward", f"Encoder built with input_channels={n_channels} on {device}")
    except Exception as e:
        failed("forward", f"Encoder build failed: {e}")
        traceback.print_exc()
        return

    # Conv3d adapter check
    first_conv = None
    for m in encoder.modules():
        if isinstance(m, torch.nn.Conv3d):
            first_conv = m
            break
    if first_conv is None:
        failed("forward", "Could not find first Conv3d in encoder")
    elif first_conv.in_channels != n_channels:
        failed("forward",
               f"First Conv3d in_channels={first_conv.in_channels}, expected {n_channels}")
    else:
        passed("forward",
               f"First Conv3d adapted to in_channels={n_channels} (kernel={tuple(first_conv.kernel_size)})")

    # Synthetic input
    x = torch.rand(1, n_channels, 64, 256, 256, device=device, dtype=torch.float32)
    info("forward", f"Input shape: {tuple(x.shape)}, range [{x.min():.3f}, {x.max():.3f}]")

    try:
        with torch.no_grad():
            out = encoder(x)
    except Exception as e:
        failed("forward", f"Forward pass raised: {e}")
        traceback.print_exc()
        return

    if "activ" not in out or "pooled" not in out:
        failed("forward", f"Output missing 'activ' or 'pooled'. Got keys: {list(out.keys())}")
        return
    passed("forward", "Forward returned 'activ' and 'pooled'")

    activ = out["activ"]
    pooled = out["pooled"]
    info("forward", f"activ shape: {tuple(activ.shape)}")
    info("forward", f"pooled shape: {tuple(pooled.shape)}")

    expected_activ = (1, 1152, 32, 32, 16)
    if tuple(activ.shape) != expected_activ:
        warned("forward",
               f"activ shape {tuple(activ.shape)} ≠ expected {expected_activ}")
    else:
        passed("forward", f"activ shape matches expected {expected_activ}")

    if tuple(pooled.shape) != (1, 1152):
        warned("forward", f"pooled shape {tuple(pooled.shape)} ≠ (1, 1152)")
    else:
        passed("forward", "pooled shape matches (1, 1152)")

    if torch.isnan(activ).any() or torch.isinf(activ).any():
        failed("forward", "activ contains NaN or Inf")
    else:
        info("forward", f"activ stats: min={activ.min():.3f} max={activ.max():.3f} mean={activ.mean():.3f}")
        passed("forward", "activ is finite")

    # ----- 5. D-axis identification -----
    section_header("5. D-axis identification in activ")

    # The dataset emits (C, D, H, W); unsqueezed to (B, C, D, H, W), so the
    # depth axis is INPUT axis 2 (size 64). After the (B,C,D,H,W)→(B,C,H,W,D)
    # permute inside MultimodalAtlas.forward and the Conv3d patch_embed
    # with kernel [8, 8, 4], depth becomes the last (smallest) spatial axis
    # of activ. Expected mapping:
    #   input axis 2 (D=64)   -> activ axis 4 (D_tok=16)
    #   input axis 3 (H=256)  -> activ axis 2 (H_tok=32)
    #   input axis 4 (W=256)  -> activ axis 3 (W_tok=32)
    # Zero the last slice along input axis 2 (D) and verify the diff
    # concentrates on activ axis 4.
    x_modified = x.clone()
    x_modified[:, :, -1, :, :] = 0.0  # zero last depth slice (input axis 2)

    try:
        with torch.no_grad():
            out_modified = encoder(x_modified)
        diff = (out_modified["activ"] - activ).abs()
    except Exception as e:
        failed("d-axis", f"Comparison forward raised: {e}")
        return

    per_axis_diff = []
    for axis in (2, 3, 4):
        other_axes = tuple(a for a in (2, 3, 4) if a != axis)
        # Mean-reduce across batch, channels, and the OTHER two spatial axes;
        # keep the target axis to measure how much variance the perturbation
        # induced along it.
        d = diff.mean(dim=(0, 1) + other_axes)
        per_axis_diff.append(d)

    h_var, w_var, d_var = (a.var().item() for a in per_axis_diff)
    info("d-axis", f"variance along activ axis 2 (H_tok=32): {h_var:.4e}")
    info("d-axis", f"variance along activ axis 3 (W_tok=32): {w_var:.4e}")
    info("d-axis", f"variance along activ axis 4 (D_tok=16): {d_var:.4e}")
    likely_d_axis = 2 + int(max(range(3), key=lambda i: [h_var, w_var, d_var][i]))
    if likely_d_axis == 4:
        passed("d-axis",
               "Input axis 2 (D) maps to activ axis 4 (D_tok) — permute is correct")
    else:
        warned("d-axis",
               f"Input axis 2 (D) appears to map to activ axis {likely_d_axis}, not 4 — "
               "check permute / convention assumptions")


# ----- main -----

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to training config YAML")
    parser.add_argument("--with-forward", action="store_true",
                        help="Also run GPU forward-pass and D-axis checks (requires CUDA + HF cache)")
    parser.add_argument("--device", default="cuda",
                        help="Device for --with-forward (default: cuda)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"{RED}Config not found: {args.config}{RESET}")
        return 2
    config = load_yaml(args.config)
    info("config", f"Loaded {args.config}")

    check_manifest(config)
    check_sample_tensor(config)
    check_dataset(config)
    if args.with_forward:
        if args.device == "cuda" and not torch.cuda.is_available():
            warned("forward", "CUDA unavailable; falling back to CPU")
            args.device = "cpu"
        check_forward(config, args.device)
    else:
        info("forward", "Skipped (use --with-forward to enable; needs GPU)")

    # Summary
    section_header("Summary")
    counts = Counter(level for level, _, _ in _results)
    print(f"  {GREEN}PASS{RESET}: {counts['PASS']}")
    print(f"  {YELLOW}WARN{RESET}: {counts['WARN']}")
    print(f"  {RED}FAIL{RESET}: {counts['FAIL']}")

    if counts["FAIL"] > 0:
        print(f"\n{RED}{BOLD}❌ Pipeline has issues. Fix FAILs before training.{RESET}")
        return 1
    if counts["WARN"] > 0:
        print(f"\n{YELLOW}{BOLD}⚠  Pipeline runnable but check WARNs above.{RESET}")
        return 0
    print(f"\n{GREEN}{BOLD}✅ All checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
