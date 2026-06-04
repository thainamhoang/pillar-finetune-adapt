#!/usr/bin/env python3
"""Preprocess PETWB-REP whole-body PET/CT into per-region Torch tensors.

Mirrors the structure of ``preprocess_vimed_petct.py`` (sibling repo) but
differs in three concrete ways that the cropping plan
(``ancient-sprouting-fern.md``) calls out:

1. **NIfTI sources, not ``.npz``.** PETWB-REP ships ``.nii.gz`` for CT
   (``rawdata/sub-X/ses-*/anat/*_ct.nii.gz``) and SUV-calibrated PET
   (``derivatives/sub-X/ses-*/*desc-suv_pet.nii.gz``). The two modalities
   live on different native grids but share DICOM Frame of Reference, so
   we crop in physical mm via each volume's own affine.

2. **Landmark-driven cropping, not proportional.** Boundaries come from
   ``landmarks.json`` produced by ``segment_landmarks_petwbrep.py``:
   ``z_apex`` (lung apex → H&N/chest boundary), ``z_base`` (lung base /
   diaphragm → chest/abd boundary), ``z_pelvis_floor`` (pelvis floor →
   abd-pelvis inferior bound). 25 mm of overlap each side covers the
   diaphragmatic dome and thoracic inlet (where regions genuinely share
   anatomy).

3. **Per-region encoder dims.** Pillar-0 has separate Head/Chest/Abd-pelvis
   CT checkpoints, each expecting a fixed input shape (Table 7 of the
   Pillar-0 paper). We resample each region crop to its target
   ``(D, H, W)`` rather than ViMED's fit-to-fixed-depth-via-pad approach.

The PET storage units flag (``--pet-units {suv,normalized}``) lets us
park the absolute-SUV-windowing decision separately: pick ``suv`` to keep
calibration through to the encoder once the SUV windowing change lands;
pick ``normalized`` for ViMED parity (log-normalize to [0,1]) until then.

Resumable on a per-``(patient, region)`` granularity. Shardable for
parallel cluster runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pillar.utils.nifti import (
    crop_physical_z,
    load_canonical,
    resample_volume,
    scan_extent_mm,
    voxel_array,
)


logger = logging.getLogger("preprocess_petwbrep")


REGION_KEYS = ("head_neck", "chest", "abdomen_pelvis")

# Pillar-0 per-region encoder input dims (Table 7). ``(D, H, W)``.
# Chest is the only region where we already have a Phase A dual-stream
# encoder; head and abd will be filled in once those Phase A runs are
# done. Preprocessing still writes all three so the dataset/manifest is
# ready when the encoders are.
REGION_TARGET_SHAPE = {
    "head_neck": (128, 256, 256),
    "chest": (256, 256, 256),
    "abdomen_pelvis": (384, 384, 384),
}

# Overlap per side at each region boundary, in mm. 25 mm covers the
# diaphragmatic dome excursion (lung-base / liver-dome ambiguity) and
# thoracic inlet (lung apex / supraclavicular nodes) so the boundary
# reader sees both sides. See plan §"Why these landmarks".
DEFAULT_OVERLAP_MM = 25.0

# CT HU range: matches ``preprocess_vimed_petct.normalize_ct`` so the
# multi-windowing tokenizer in ``pillar/utils/petct_windowing.py`` sees
# the same dynamic range across datasets.
CT_HU_MIN, CT_HU_MAX = -1024.0, 3071.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--petwbrep-root", type=Path, required=True)
    p.add_argument("--landmarks-dir", type=Path, required=True,
                   help="Output of segment_landmarks_petwbrep.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--regions", nargs="+", default=list(REGION_KEYS), choices=REGION_KEYS)
    p.add_argument("--overlap-mm", type=float, default=DEFAULT_OVERLAP_MM)
    p.add_argument("--pet-units", choices=["suv", "normalized"], default="suv",
                   help=(
                       "How PET values are stored in x_raw[1]. 'suv' preserves "
                       "absolute SUVbw (requires absolute-SUV windowing in the "
                       "dataset path); 'normalized' log-normalises to [0,1] via "
                       "the 99.5th percentile (matches ViMED preprocessor; "
                       "compatible with the legacy fractional PET windows). "
                       "Recorded in per-sample metadata['pet_units']."
                   ))
    p.add_argument("--report-segments-json", type=Path, default=None,
                   help=(
                       "Optional path to LLM-segmented reports "
                       "(``{patient_id: {head_neck, chest, abdomen_pelvis}}``). "
                       "If omitted, report_text is left empty and can be "
                       "backfilled later by a separate manifest-update step."
                   ))
    p.add_argument("--metadata-csv", default="metadata.csv",
                   help="Per-patient metadata file under petwbrep-root (sex/weight/dose/...)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-patients", type=int, default=None,
                   help="Debug limit after sharding")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def find_session_files(patient_dir: Path) -> tuple[Path | None, Path | None]:
    """Return ``(raw_ct_path, derivative_suv_pet_path)`` for one patient.

    PETWB-REP's BIDS layout is asymmetric (per the paper's Fig. 4):

    * Raw CT lives nested under a session dir::

          rawdata/sub-X/ses-01/anat/sub-X_ses-01_ct.nii.gz

    * SUV PET derivatives are flat under the subject dir::

          derivatives/sub-X/sub-X_ses-01_desc-suv_pet.nii.gz

    Using ``rglob`` on both sides handles either layout, so the same
    code works if a future release reorganises derivatives under ``ses-*/``
    or if a local copy uses a different intermediate level. The first
    match wins; cohorts with multiple sessions per patient (not present
    in the released 490) would need a session-aware variant.
    """
    raw_ct = sorted(patient_dir.rglob("*_ct.nii.gz"))
    # Filter out derivatives-style filenames that may accidentally live
    # under rawdata (e.g. ``*desc-resampled_ct.nii.gz``); rawdata CT
    # filenames don't carry a ``desc-`` qualifier.
    raw_ct = [p for p in raw_ct if "desc-" not in p.name]
    raw_ct_path = raw_ct[0] if raw_ct else None

    # Derivatives mirror the rawdata patient ID. patient_dir lives under
    # rawdata/; locate the parallel directory under derivatives/.
    derivatives_root = patient_dir.parent.parent / "derivatives" / patient_dir.name
    # The PETWB-REP release uses ``*_desc-pet_suv.nii.gz`` (BIDS-style:
    # suffix=pet, desc=suv). The paper's Fig. 4 documented it as
    # ``*_desc-suv_pet.nii.gz`` (the two halves swapped). Match either
    # so we're robust to both observed conventions.
    suv_pet = sorted(derivatives_root.rglob("*desc-pet_suv*.nii.gz"))
    if not suv_pet:
        suv_pet = sorted(derivatives_root.rglob("*desc-suv_pet*.nii.gz"))
    suv_pet_path = suv_pet[0] if suv_pet else None
    return raw_ct_path, suv_pet_path


def compute_region_bounds_mm(
    region: str,
    landmarks: dict[str, float],
    overlap_mm: float,
) -> tuple[float, float]:
    """Return ``(z_lo_mm, z_hi_mm)`` for the region's crop.

    ``z_lo`` is the inferior bound, ``z_hi`` the superior bound. Both
    inclusive when handed to ``crop_physical_z``. ``overlap_mm`` is added
    on each side at the boundaries that abut another region.
    """
    z_apex = landmarks["z_apex_mm"]
    z_base = landmarks["z_base_mm"]
    z_pelvis = landmarks["z_pelvis_floor_mm"]
    z_top = landmarks["scan_top_mm"]
    z_bot = landmarks["scan_bottom_mm"]

    if region == "head_neck":
        # Top of scan (vertex / skull base) down to lung apex, plus
        # overlap into chest. No overlap at the top — that's the scan
        # boundary, nothing to cover.
        return (z_apex - overlap_mm, z_top)
    if region == "chest":
        # Lung base (with overlap into abdomen) up to lung apex (with
        # overlap into head/neck).
        return (z_base - overlap_mm, z_apex + overlap_mm)
    if region == "abdomen_pelvis":
        # Pelvis floor (no overlap at the bottom of scan) up through the
        # diaphragm with overlap into chest. Use scan_bottom rather than
        # the segmenter's pelvis floor if the latter is missing — see
        # segment_landmarks_petwbrep.compute_landmarks fallback.
        return (max(z_bot, z_pelvis), z_base + overlap_mm)
    raise ValueError(f"Unknown region: {region}")


def normalize_ct_hu(arr: np.ndarray) -> np.ndarray:
    """Clip CT to the standard HU range. No further normalisation.

    Pillar-0's multi-windowing tokenizer expects raw HU and applies its
    own per-channel window/clip — see ``pillar/utils/petct_windowing.py``.
    Don't z-score, don't min-max.
    """
    return np.clip(arr.astype(np.float32, copy=False), CT_HU_MIN, CT_HU_MAX)


def normalize_pet_logp995(arr: np.ndarray) -> np.ndarray:
    """Match the ViMED preprocessor's PET normalisation.

    Log-compress and divide by ``log1p(99.5th percentile of positive)``
    so output sits in [0,1]. Compatible with the existing fractional PET
    windows in ``DUAL_STREAM_PET_WINDOWS``. Used when ``--pet-units
    normalized`` is selected; preserved as the legacy / ViMED-parity
    fallback while the absolute-SUV windowing decision is pending.
    """
    arr = arr.astype(np.float32, copy=False)
    positive = arr[arr > 0]
    if positive.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p995 = float(np.percentile(positive, 99.5))
    if not math.isfinite(p995) or p995 <= 0:
        p995 = float(positive.max()) if positive.max() > 0 else 1.0
    clipped = np.clip(arr, 0.0, p995)
    return np.clip(np.log1p(clipped) / np.log1p(p995), 0.0, 1.0).astype(np.float32)


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Index metadata.csv by ``Image ID`` (the BIDS subject identifier)."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("Image ID") or row.get("image_id") or row.get("patient_id")
            if key:
                # Normalise to the same form used by patient directory names
                # (``sub-XXXXXXX-XXXXXX``). Some manifests omit the "sub-" prefix.
                key = key if key.startswith("sub-") else f"sub-{key}"
                out[key] = row
    return out


def load_report_segments(path: Path | None) -> dict[str, dict[str, str]]:
    """Optional ``{patient_id: {region: text}}`` from the LLM pass.

    Returns empty dict if not supplied; the preprocessor then writes
    ``report_text=""`` and the manifest is backfilled later.
    """
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict[str, str]] = {}
    for pid, sections in data.items():
        key = pid if pid.startswith("sub-") else f"sub-{pid}"
        out[key] = sections
    return out


def resolve_gender(raw_row: dict[str, str]) -> str:
    """Map the metadata gender to ``"male"`` / ``"female"`` (SAMF keys)."""
    raw = (raw_row.get("Sex") or raw_row.get("gender") or "").strip().lower()
    if raw.startswith("f"):
        return "female"
    return "male"


def process_patient_region(
    patient_id: str,
    region: str,
    ct_path: Path,
    pet_path: Path,
    landmarks: dict[str, float],
    out_dir: Path,
    pet_units: Literal["suv", "normalized"],
    overlap_mm: float,
    metadata_row: dict[str, str],
    report_text: str,
    overwrite: bool,
) -> dict[str, Any] | None:
    """Run one ``(patient, region)`` pair. Returns manifest row or None on skip."""
    target_shape = REGION_TARGET_SHAPE[region]
    tensor_dir = out_dir / "tensors" / region
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensor_dir / f"{patient_id}_{region}.pt"

    if tensor_path.exists() and not overwrite:
        # Resume path: re-emit the manifest row from already-saved metadata
        # so the user gets a complete CSV even on a partial re-run.
        item = torch.load(tensor_path, map_location="cpu", weights_only=False)
        md = item.get("metadata", {})
        return {
            "study_id": patient_id,
            "split": "",
            "region": region,
            "tensor_path": str(tensor_path),
            "ct_path": str(ct_path),
            "pet_path": str(pet_path),
            "dataset": "petwbrep",
            "gender": md.get("gender", ""),
            "pet_units": md.get("pet_units", pet_units),
            "z_start_mm": md.get("z_start_mm", ""),
            "z_end_mm": md.get("z_end_mm", ""),
            "report_text": md.get("report_text", report_text or ""),
        }

    # Crop each modality independently to the same physical-mm window.
    z_lo, z_hi = compute_region_bounds_mm(region, landmarks, overlap_mm)

    ct_img = load_canonical(ct_path)
    pet_img = load_canonical(pet_path)
    ct_crop = crop_physical_z(ct_img, z_lo, z_hi)
    pet_crop = crop_physical_z(pet_img, z_lo, z_hi)

    # Arrays come out as (X, Y, Z) from get_fdata in canonical RAS. For
    # the Pillar-0 convention we want (D, H, W) = (Z, Y, X) — i.e. axial
    # slices as the leading axis, matching what x_raw consumers expect.
    # Transpose accordingly.
    ct_arr = np.transpose(voxel_array(ct_crop), (2, 1, 0))
    pet_arr = np.transpose(voxel_array(pet_crop), (2, 1, 0))

    # Resample to encoder target dims. CT uses area (anti-aliased
    # downsampling like the ViMED 512→256 path); PET uses trilinear
    # because we may need to upsample its already-coarse native z.
    ct_resampled = resample_volume(ct_arr, target_shape, mode="area")
    pet_resampled = resample_volume(pet_arr, target_shape, mode="trilinear")

    ct_resampled = normalize_ct_hu(ct_resampled)
    if pet_units == "normalized":
        pet_resampled = normalize_pet_logp995(pet_resampled)
    else:
        # SUV: keep absolute values; clamp to [0, +inf) to drop any
        # spurious negative artifacts from interpolation. No further
        # transform — the dataset path will apply absolute-SUV windowing.
        pet_resampled = np.clip(pet_resampled.astype(np.float32), 0.0, None)

    x_raw = torch.stack(
        [torch.from_numpy(ct_resampled), torch.from_numpy(pet_resampled)],
        dim=0,
    ).to(torch.float16)

    metadata = {
        "study_id": patient_id,
        "region": region,
        "dataset": "petwbrep",
        "gender": resolve_gender(metadata_row),
        "pet_units": pet_units,
        "z_start_mm": float(z_lo),
        "z_end_mm": float(z_hi),
        "scan_extent_mm": scan_extent_mm(ct_img),
        "report_text": report_text or "",
        # Carry through clinically useful metadata fields for downstream
        # filtering / stratification. Cast everything to str to keep the
        # .pt file safely serialisable across torch versions.
        "Disease": str(metadata_row.get("Disease", "")),
        "Age": str(metadata_row.get("Age", "")),
        "Weight_kg": str(metadata_row.get("Weight_kg", "")),
        "ct_path": str(ct_path),
        "pet_path": str(pet_path),
    }

    torch.save({"x_raw": x_raw, "metadata": metadata}, tensor_path)

    return {
        "study_id": patient_id,
        "split": "",
        "region": region,
        "tensor_path": str(tensor_path),
        "ct_path": str(ct_path),
        "pet_path": str(pet_path),
        "dataset": "petwbrep",
        "gender": metadata["gender"],
        "pet_units": pet_units,
        "z_start_mm": metadata["z_start_mm"],
        "z_end_mm": metadata["z_end_mm"],
        "report_text": metadata["report_text"],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rawdata = args.petwbrep_root / "rawdata"
    if not rawdata.is_dir():
        sys.exit(f"rawdata/ not found under {args.petwbrep_root}")

    metadata_index = load_metadata(args.petwbrep_root / args.metadata_csv)
    report_index = load_report_segments(args.report_segments_json)

    patient_dirs = sorted(p for p in rawdata.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    shard = [p for i, p in enumerate(patient_dirs) if i % args.num_shards == args.shard_index]
    if args.max_patients is not None:
        shard = shard[: args.max_patients]
    logger.info("shard %d/%d: %d patients", args.shard_index, args.num_shards, len(shard))

    manifest_path = args.out_dir / f"manifest_shard{args.shard_index:04d}_of_{args.num_shards:04d}.csv"
    skipped_path = args.out_dir / f"skipped_shard{args.shard_index:04d}_of_{args.num_shards:04d}.csv"

    manifest_fields = [
        "study_id", "split", "region", "tensor_path",
        "ct_path", "pet_path", "dataset", "gender", "pet_units",
        "z_start_mm", "z_end_mm", "report_text",
    ]
    skipped_fields = ["study_id", "region", "reason", "details"]

    with manifest_path.open("w", newline="") as mf, skipped_path.open("w", newline="") as sf:
        mw = csv.DictWriter(mf, fieldnames=manifest_fields)
        sw = csv.DictWriter(sf, fieldnames=skipped_fields)
        mw.writeheader()
        sw.writeheader()

        for i, patient_dir in enumerate(shard, start=1):
            patient_id = patient_dir.name

            # Per-patient lookups: landmarks (required), metadata (required-ish),
            # report segments (optional).
            landmarks_path = args.landmarks_dir / f"{patient_id}.json"
            if not landmarks_path.exists():
                for region in args.regions:
                    sw.writerow({
                        "study_id": patient_id, "region": region,
                        "reason": "no_landmarks", "details": str(landmarks_path),
                    })
                continue
            landmarks = json.loads(landmarks_path.read_text())

            ct_path, pet_path = find_session_files(patient_dir)
            if ct_path is None or pet_path is None:
                for region in args.regions:
                    sw.writerow({
                        "study_id": patient_id, "region": region,
                        "reason": "missing_modality",
                        "details": f"ct={ct_path}, pet={pet_path}",
                    })
                continue

            metadata_row = metadata_index.get(patient_id, {})
            segs = report_index.get(patient_id, {})

            for region in args.regions:
                try:
                    row = process_patient_region(
                        patient_id=patient_id,
                        region=region,
                        ct_path=ct_path,
                        pet_path=pet_path,
                        landmarks=landmarks,
                        out_dir=args.out_dir,
                        pet_units=args.pet_units,
                        overlap_mm=args.overlap_mm,
                        metadata_row=metadata_row,
                        report_text=segs.get(region, ""),
                        overwrite=args.overwrite,
                    )
                    if row is not None:
                        mw.writerow(row)
                except Exception as exc:
                    sw.writerow({
                        "study_id": patient_id, "region": region,
                        "reason": type(exc).__name__,
                        "details": f"{exc!r}\n{traceback.format_exc(limit=2)}",
                    })
                    logger.warning("FAIL %s/%s: %r", patient_id, region, exc)

            if i % 10 == 0:
                logger.info("[%d/%d] processed", i, len(shard))

    logger.info("wrote %s and %s", manifest_path, skipped_path)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
