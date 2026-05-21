#!/usr/bin/env python3
"""Export a fused CT+PET dual-stream encoder for the report-generation phase.

Loads two Phase-1 training checkpoints (CT and PET-from-CT), constructs a
:class:`ViMedChestDualStreamFusedEncoder` (= dual Pillar0-initialized Atlas
encoders + ``TokenConcatFusion``) with each backbone overlaid by its trained
weights, runs an optional smoke forward to verify the
``(B, 32768, 1152)`` output contract, and saves the combined ``state_dict``
to a single ``.pt`` artifact.

The output artifact has NO classification head, NO pool head, and NO
optimizer state — it is the pure encoder that the Phase B Q-Former + LLM
decoder will consume via ``state_dict`` loading.

Usage::

    uv run python scripts/export_dual_stream_encoder.py \\
        --ct-ckpt  logs/dual-stream-pillar/ct/<run>/checkpoints/best.ckpt \\
        --pet-ckpt logs/dual-stream-pillar/pet_from_ct/<run>/checkpoints/best.ckpt \\
        --out      logs/dual-stream-pillar/fusion/dual_stream_encoder.pt \\
        --smoke    /scratch/thahoa/PET/ViMed_prep_v2/manifest_splits.csv

Both ``--ct-ckpt`` and ``--pet-ckpt`` may be either:

- A Lightning training checkpoint (``.ckpt``) with ``state_dict`` /
  ``backbone_model.`` prefixed keys (the dual-stream finetune writes
  these). ``MultimodalAtlas._load_pretrained_backbone_ckpt`` handles the
  unwrap + prefix strip + shape-mismatch filtering internally; no manual
  extract step is needed.
- An encoder-only ``.pt`` produced by ``scripts/extract_encoder.py``.

The exported state_dict is keyed under :class:`ViMedChestDualStreamFusedEncoder`'s
module namespace, so re-loading is::

    model = ViMedChestDualStreamFusedEncoder(...)
    model.load_state_dict(torch.load(out_path)["state_dict"])
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
        print("[export] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(spec)


def build_model(
    ct_ckpt: Path,
    pet_ckpt: Path,
    device: torch.device,
    ct_channels: int,
    pet_channels: int,
    model_repo_id: str,
    model_revision: str,
):
    """Construct the fused encoder, overlaying both backbones in one shot."""
    # Local import so `--help` doesn't pull in transformers / hf hub.
    from pillar.models.backbones import ViMedChestDualStreamFusedEncoder

    print(f"[export] building ViMedChestDualStreamFusedEncoder on {device}")
    print(f"[export]   ct_channels={ct_channels}, pet_channels={pet_channels}")
    print(f"[export]   ct overlay  = {ct_ckpt}")
    print(f"[export]   pet overlay = {pet_ckpt}")
    model = ViMedChestDualStreamFusedEncoder(
        ct_channels=ct_channels,
        pet_channels=pet_channels,
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        device=str(device),
        ct_pretrained_backbone_ckpt=str(ct_ckpt),
        pet_pretrained_backbone_ckpt=str(pet_ckpt),
    )
    model = model.to(device)
    model.eval()

    # Report parameter / size summary.
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable_fusion = sum(p.numel() for p in model.fusion.parameters())
    print(
        f"[export] total params: {n_total/1e6:.1f}M "
        f"(fusion-only: {n_trainable_fusion/1e3:.1f}K — type+pos embeds)"
    )
    return model


def smoke_forward(
    model,
    manifest_path: Path,
    device: torch.device,
    region: str,
    split: str,
) -> None:
    """Forward one ViMedChestReportDataset sample, assert the output contract."""
    from pillar.datasets.vimed_chest_report import ViMedChestReportDataset

    print(f"[smoke] loading one {split}-split sample from {manifest_path}")
    ds = ViMedChestReportDataset(
        manifest_path=manifest_path,
        split=split,
        region=region,
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"No samples in split={split!r} region={region!r}; "
            "cannot run smoke forward."
        )
    item = ds[0]
    ct = item["ct_windows"].unsqueeze(0).to(device).float()    # (1, 11, D, H, W)
    pet = item["pet_windows"].unsqueeze(0).to(device).float()  # (1,  4, D, H, W)
    print(f"[smoke]   ct={tuple(ct.shape)}  pet={tuple(pet.shape)}")

    with torch.no_grad():
        out = model(ct, pet)
    activ = out["activ"]
    pooled = out["pooled"]
    print(f"[smoke]   activ={tuple(activ.shape)}  pooled={tuple(pooled.shape)}")
    expected_activ = (1, model.num_tokens, model.token_dim)
    if tuple(activ.shape) != expected_activ:
        raise RuntimeError(
            f"activ shape {tuple(activ.shape)} != expected {expected_activ}"
        )
    if not torch.isfinite(activ).all():
        raise RuntimeError("activ contains NaN/Inf")
    print(f"[smoke]   activ stats: min={activ.min():.3f} max={activ.max():.3f} mean={activ.mean():.3f}")
    print("[smoke] PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ct-ckpt", required=True, type=Path,
                    help="Phase-1 CT training checkpoint (.ckpt or encoder_only.pt)")
    ap.add_argument("--pet-ckpt", required=True, type=Path,
                    help="Phase-1 PET-from-CT training checkpoint")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write the fused encoder .pt")
    ap.add_argument("--smoke", type=Path, default=None,
                    help="Manifest CSV path; if set, run a forward pass on one sample")
    ap.add_argument("--device", default="cuda",
                    help="Device for model build / smoke forward (default: cuda)")
    ap.add_argument("--ct-channels", type=int, default=11)
    ap.add_argument("--pet-channels", type=int, default=4)
    ap.add_argument("--model-repo-id", default="YalaLab/Pillar0-ChestCT")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--smoke-split", default="val")
    ap.add_argument("--smoke-region", default="chest")
    args = ap.parse_args()

    if not args.ct_ckpt.exists():
        print(f"[export] ERROR: CT checkpoint not found: {args.ct_ckpt}", file=sys.stderr)
        return 2
    if not args.pet_ckpt.exists():
        print(f"[export] ERROR: PET checkpoint not found: {args.pet_ckpt}", file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    model = build_model(
        ct_ckpt=args.ct_ckpt,
        pet_ckpt=args.pet_ckpt,
        device=device,
        ct_channels=args.ct_channels,
        pet_channels=args.pet_channels,
        model_repo_id=args.model_repo_id,
        model_revision=args.model_revision,
    )

    if args.smoke is not None:
        if not args.smoke.exists():
            print(f"[smoke] ERROR: manifest not found: {args.smoke}", file=sys.stderr)
            return 2
        smoke_forward(
            model=model,
            manifest_path=args.smoke,
            device=device,
            region=args.smoke_region,
            split=args.smoke_split,
        )
    else:
        print("[smoke] skipped (no --smoke manifest passed)")

    # Move to CPU before saving so the artifact is portable.
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": cpu_state,
        "config": {
            "class": "ViMedChestDualStreamFusedEncoder",
            "ct_channels": args.ct_channels,
            "pet_channels": args.pet_channels,
            "model_repo_id": args.model_repo_id,
            "model_revision": args.model_revision,
            "token_dim": model.token_dim,
            "grid": list(model.grid),
            "num_tokens": model.num_tokens,
            "source": {
                "ct_ckpt": str(args.ct_ckpt),
                "pet_ckpt": str(args.pet_ckpt),
            },
        },
    }
    torch.save(artifact, args.out)
    n_bytes = sum(v.numel() * v.element_size() for v in cpu_state.values())
    print(
        f"[export] wrote {args.out}  "
        f"({len(cpu_state)} keys, {n_bytes / 1024**2:.1f} MiB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
