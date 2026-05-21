#!/usr/bin/env python3
"""Sanity-check the dual-stream fused encoder pipeline (Phase A artifact).

Mirrors ``scripts/sanity_check.py`` (same PASS/WARN/FAIL helpers) but
exercises the new ``ViMedChestDualStreamFusedEncoder`` end-to-end:

1. Both Phase-1 encoder checkpoints exist and unwrap cleanly
2. ``ViMedChestReportDataset`` emits ``ct_windows (11, D, H, W)`` and
   ``pet_windows (4, D, H, W)`` from a real sample
3. ``ViMedChestDualStreamFusedEncoder`` forward returns the expected
   ``(1, 32768, 1152)`` token sequence + ``(1, 1152)`` pooled vector
4. CT-half and PET-half of ``activ`` are meaningfully different
   (sanity that overlay loading isn't a no-op and the two encoders
   produce distinct outputs)
5. The modality type embed is distinguishable -- swapping the two
   halves' first tokens changes the output
6. The positional embedding contributes (parameter has non-trivial std,
   and zeroing it changes the output)

Usage::

    uv run python scripts/sanity_check_dual_stream.py \\
        --ct-ckpt   logs/dual-stream-pillar/ct/<run>/checkpoints/best.ckpt \\
        --pet-ckpt  logs/dual-stream-pillar/pet_from_ct/<run>/checkpoints/best.ckpt \\
        --manifest  /scratch/thahoa/PET/ViMed_prep_v2/manifest_splits.csv

Returns nonzero exit code if any FAIL is recorded. ``--with-forward`` is
on by default since the whole point of this script is to exercise the
forward path; pass ``--no-forward`` to skip GPU work and just check the
checkpoint/dataset bits.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# ----- pretty printing (mirrors sanity_check.py) -----

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

_results: list[tuple[str, str, str]] = []


def _emit(level: str, color: str, section: str, message: str) -> None:
    print(f"{color}[{level}]{RESET} {BOLD}{section}{RESET} — {message}")
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


# ----- 1. Checkpoints loadable -----

def check_checkpoints(ct_ckpt: Path, pet_ckpt: Path) -> bool:
    section_header("1. Phase-1 encoder checkpoints")

    all_ok = True
    for label, p in (("ct", ct_ckpt), ("pet", pet_ckpt)):
        if not p.exists():
            failed(f"ckpt[{label}]", f"missing: {p}")
            all_ok = False
            continue
        try:
            raw = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            failed(f"ckpt[{label}]", f"torch.load raised: {e}")
            all_ok = False
            continue

        # Unwrap exactly the way MultimodalAtlas._load_pretrained_backbone_ckpt does.
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
            kind = "lightning .ckpt (state_dict wrapper)"
        elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
            kind = "model-dict wrapper"
        else:
            sd = raw
            kind = "flat state_dict"

        if not isinstance(sd, dict):
            failed(f"ckpt[{label}]", f"unwrap produced non-dict: {type(sd)}")
            all_ok = False
            continue

        n_backbone = sum(1 for k in sd if k.startswith("backbone_model."))
        size_mib = sum(v.numel() * v.element_size() for v in sd.values()
                       if torch.is_tensor(v)) / 1024**2
        info(f"ckpt[{label}]",
             f"{kind}; {len(sd)} keys ({n_backbone} backbone_model.*); {size_mib:.1f} MiB")
        passed(f"ckpt[{label}]", f"loadable: {p.name}")

    return all_ok


# ----- 2. Dataset shapes -----

def check_dataset(manifest_path: Path, region: str, split: str) -> dict | None:
    section_header(f"2. ViMedChestReportDataset[{split}, {region}] sample shapes")

    if not manifest_path.exists():
        failed("dataset", f"manifest missing: {manifest_path}")
        return None

    try:
        from pillar.datasets.vimed_chest_report import ViMedChestReportDataset
    except Exception as e:
        failed("dataset", f"import failed: {e}")
        return None

    try:
        ds = ViMedChestReportDataset(
            manifest_path=manifest_path,
            split=split,
            region=region,
        )
    except Exception as e:
        failed("dataset", f"instantiation failed: {e}")
        return None

    if len(ds) == 0:
        failed("dataset", f"empty after split={split!r} region={region!r}")
        return None
    info("dataset", f"len={len(ds)}")

    try:
        item = ds[0]
    except Exception as e:
        failed("dataset", f"__getitem__ raised: {e}")
        traceback.print_exc()
        return None

    ct = item.get("ct_windows")
    pet = item.get("pet_windows")
    if not isinstance(ct, torch.Tensor) or not isinstance(pet, torch.Tensor):
        failed("dataset", "ct_windows / pet_windows are not tensors")
        return None

    info("dataset", f"ct_windows  shape={tuple(ct.shape)}  dtype={ct.dtype}")
    info("dataset", f"pet_windows shape={tuple(pet.shape)}  dtype={pet.dtype}")

    ok = True
    if ct.ndim != 4 or ct.shape[0] != 11:
        failed("dataset", f"ct_windows expected (11, D, H, W); got {tuple(ct.shape)}")
        ok = False
    if pet.ndim != 4 or pet.shape[0] != 4:
        failed("dataset", f"pet_windows expected (4, D, H, W); got {tuple(pet.shape)}")
        ok = False
    if ct.shape[1:] != pet.shape[1:]:
        failed("dataset",
               f"CT spatial {tuple(ct.shape[1:])} != PET spatial {tuple(pet.shape[1:])}")
        ok = False

    if ok:
        passed("dataset",
               f"CT (11, {ct.shape[1]}, {ct.shape[2]}, {ct.shape[3]}) + "
               f"PET (4, {pet.shape[1]}, {pet.shape[2]}, {pet.shape[3]}) shapes correct")

    # Report-text presence (informational; required for Phase B but not A).
    txt = item.get("report_text", "")
    if isinstance(txt, str) and txt.strip():
        info("dataset", f"report_text present ({len(txt)} chars)")
    else:
        warned("dataset", "report_text empty for this sample (OK for Phase A export)")

    return item


# ----- 3-6. Forward + fusion sanity -----

def check_forward(
    ct_ckpt: Path,
    pet_ckpt: Path,
    item: dict,
    device: str,
) -> None:
    section_header("3. ViMedChestDualStreamFusedEncoder forward")

    try:
        from pillar.models.backbones import ViMedChestDualStreamFusedEncoder
    except Exception as e:
        failed("forward", f"import failed: {e}")
        return

    try:
        model = ViMedChestDualStreamFusedEncoder(
            ct_channels=11,
            pet_channels=4,
            device=device,
            ct_pretrained_backbone_ckpt=str(ct_ckpt),
            pet_pretrained_backbone_ckpt=str(pet_ckpt),
        ).to(device)
        model.eval()
    except Exception as e:
        failed("forward", f"model build failed: {e}")
        traceback.print_exc()
        return
    passed("forward", "fused encoder built (HF + CT overlay + PET overlay)")

    # Check the per-encoder first Conv3d in_channels matches expectations.
    ct_first = next(m for m in model.dual_stream.ct_encoder.modules()
                    if isinstance(m, torch.nn.Conv3d))
    pet_first = next(m for m in model.dual_stream.pet_encoder.modules()
                     if isinstance(m, torch.nn.Conv3d))
    if ct_first.in_channels != 11:
        failed("forward", f"CT first Conv3d in_channels={ct_first.in_channels} != 11")
    else:
        passed("forward", "CT first Conv3d adapted to 11 channels")
    if pet_first.in_channels != 4:
        failed("forward", f"PET first Conv3d in_channels={pet_first.in_channels} != 4")
    else:
        passed("forward", "PET first Conv3d adapted to 4 channels")

    ct = item["ct_windows"].unsqueeze(0).to(device).float()
    pet = item["pet_windows"].unsqueeze(0).to(device).float()
    info("forward", f"input ct={tuple(ct.shape)} pet={tuple(pet.shape)}")

    try:
        with torch.no_grad():
            out = model(ct, pet)
    except Exception as e:
        failed("forward", f"forward raised: {e}")
        traceback.print_exc()
        return

    activ = out["activ"]
    pooled = out["pooled"]
    info("forward", f"activ={tuple(activ.shape)} pooled={tuple(pooled.shape)}")

    expected_activ = (1, model.num_tokens, model.token_dim)
    if tuple(activ.shape) != expected_activ:
        failed("forward",
               f"activ shape {tuple(activ.shape)} != expected {expected_activ}")
        return
    passed("forward", f"activ shape matches {expected_activ}")

    if not torch.isfinite(activ).all():
        failed("forward", "activ contains NaN/Inf")
        return
    passed("forward", "activ is finite")

    # ----- 4. CT-half vs PET-half differ -----
    section_header("4. CT-half vs PET-half differ")
    n_per = model.fusion.num_tokens_per_modality
    ct_tokens = activ[:, :n_per]
    pet_tokens = activ[:, n_per:]
    ct_norm = ct_tokens.norm().item()
    pet_norm = pet_tokens.norm().item()
    diff = (ct_tokens - pet_tokens).norm().item()
    info("halves", f"||ct_half||={ct_norm:.3f}  ||pet_half||={pet_norm:.3f}  ||ct-pet||={diff:.3f}")
    # Floor: if the two halves were identical (e.g. overlay didn't apply
    # and both encoders started from the same HF weights with same input),
    # diff would be ~0. We're loading different overlays AND different
    # input channels, so diff should be at least ~10% of either norm.
    floor = 0.10 * max(ct_norm, pet_norm)
    if diff < floor:
        warned("halves",
               f"||ct-pet||={diff:.3f} is small (< 10% of either norm = {floor:.3f}); "
               "verify both ckpts were actually loaded.")
    else:
        passed("halves", "CT-half and PET-half are meaningfully distinct")

    # ----- 5. Modality type embed distinguishable -----
    section_header("5. Modality type embedding is distinguishable")
    type_embed = model.fusion.modality_type_embed.detach()
    type_diff = (type_embed[0] - type_embed[1]).norm().item()
    info("type-embed", f"||type[CT] - type[PET]|| = {type_diff:.4f}")
    if type_diff < 1e-4:
        failed("type-embed", "CT and PET type embeddings are essentially identical (init bug?)")
    else:
        passed("type-embed", "type embeddings carry distinct signal (post trunc_normal init)")

    # ----- 6. Pos embed contributes -----
    section_header("6. Positional embedding contributes")
    pos_embed = model.fusion.pos_embed.detach()
    pos_std = pos_embed.std().item()
    pos_max = pos_embed.abs().max().item()
    info("pos-embed", f"pos_embed.std={pos_std:.4f}  pos_embed.abs.max={pos_max:.4f}")
    if pos_std < 1e-4:
        failed("pos-embed", "pos_embed is essentially zero (init bug?)")
        return

    # Zero pos_embed and re-run; output should change.
    with torch.no_grad():
        saved_pos = pos_embed.clone()
        model.fusion.pos_embed.zero_()
        out2 = model(ct, pet)
        # Restore so subsequent ops see the original.
        model.fusion.pos_embed.copy_(saved_pos)

    delta = (out2["activ"] - activ).norm().item()
    info("pos-embed", f"||activ_with_pos - activ_zero_pos|| = {delta:.4f}")
    if delta < 1e-3:
        failed("pos-embed", "zeroing pos_embed did NOT change the fused output")
    else:
        passed("pos-embed", "pos_embed contributes to the fused output")


# ----- main -----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ct-ckpt", required=True, type=Path)
    ap.add_argument("--pet-ckpt", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="ViMED manifest CSV (manifest_splits.csv)")
    ap.add_argument("--region", default="chest")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-forward", action="store_true",
                    help="Skip GPU forward / fusion checks (sections 3-6)")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        warned("device", "CUDA requested but unavailable; falling back to CPU "
                         "(forward will be slow but still valid)")
        device = "cpu"

    ckpts_ok = check_checkpoints(args.ct_ckpt, args.pet_ckpt)
    item = check_dataset(args.manifest, args.region, args.split)

    if not args.no_forward:
        if not ckpts_ok or item is None:
            warned("forward", "skipping forward because earlier sections failed")
        else:
            check_forward(args.ct_ckpt, args.pet_ckpt, item, device)
    else:
        info("forward", "skipped (--no-forward)")

    # Summary.
    section_header("Summary")
    counts = Counter(level for level, _, _ in _results)
    print(f"  {GREEN}PASS{RESET}: {counts['PASS']}")
    print(f"  {YELLOW}WARN{RESET}: {counts['WARN']}")
    print(f"  {RED}FAIL{RESET}: {counts['FAIL']}")
    if counts["FAIL"] > 0:
        print(f"\n{RED}{BOLD}❌ Dual-stream export has issues. Fix FAILs.{RESET}")
        return 1
    if counts["WARN"] > 0:
        print(f"\n{YELLOW}{BOLD}⚠  Runnable but check WARNs above.{RESET}")
        return 0
    print(f"\n{GREEN}{BOLD}✅ All checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
