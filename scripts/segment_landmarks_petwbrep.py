#!/usr/bin/env python3
"""Extract anatomical landmark z-coordinates for PETWB-REP via TotalSegmentator.

This is a one-time pass that produces a ``landmarks.json`` per patient
containing the physical-mm z-positions of the lung apex, lung base, and
pelvis floor — the three anatomical levels we need to crop whole-body
PET/CT into head-neck / chest / abdomen-pelvis regions (see
``ancient-sprouting-fern.md`` plan).

We could segment on the fly inside the main preprocessor, but caching the
landmarks separately:

* lets the preprocessor stay fast and deterministic on re-runs,
* makes the boundary decisions auditable (human-checkable JSON),
* decouples the GPU-bound segmentation pass from the CPU-bound cropping
  pass so they can scale independently across the cluster.

Why these landmarks (not, say, T1-T12 vertebrae): lung apex is the
classic chest superior boundary (thoracic inlet ≈ lung apex level), and
lung base + diaphragm dome co-locate (the lung's posteriorinferior extent
runs into the diaphragm). Pelvis floor (inferior extent of hip+sacrum)
marks the bottom of the abdomen-pelvis crop. TotalSegmentator gives all
three reliably from a single CT pass.

Usage::

    python scripts/segment_landmarks_petwbrep.py \\
        --petwbrep-root /path/to/PETWB-REP \\
        --out-dir       /path/to/landmarks_cache \\
        --device        gpu

Resumable: skips any patient whose ``landmarks.json`` already exists
unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

# Import nibabel-based helpers from pillar.utils.nifti for canonical RAS +
# world-z extraction. The masks TotalSegmentator emits are aligned to the
# input volume's grid, so canonicalising both input and masks consistently
# lets us read z-bounds the same way for every patient.
from pillar.utils.nifti import load_canonical, world_z_per_slice


logger = logging.getLogger("segment_landmarks_petwbrep")


# Class lists per landmark. We take the union over multiple ROIs per
# landmark and use the extreme z of the union, which is robust to a
# missing lobe (e.g. post-pneumonectomy) — if even one lobe segments, we
# still get a valid apex/base. The 5 lung classes cover both lungs; the
# 3 pelvis classes cover the bony pelvis floor (hip joints + sacrum
# inferior tip).
LUNG_ROIS = [
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_left",
    "lung_lower_lobe_right",
]
PELVIS_ROIS = ["hip_left", "hip_right", "sacrum"]
ALL_ROIS = LUNG_ROIS + PELVIS_ROIS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--petwbrep-root", type=Path, required=True,
                   help="PETWB-REP dataset root (containing rawdata/, derivatives/, metadata.csv)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write per-patient landmarks JSONs and the cohort summary")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"],
                   help="TotalSegmentator inference device")
    p.add_argument("--fast", action="store_true",
                   help="Use TotalSegmentator's --fast mode (3mm preset, ~2x faster, slightly less accurate). "
                        "Acceptable here because we only need z-bounds, not voxel-precise contours.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-segment patients whose landmarks.json already exists")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split work across shards for parallel cluster runs")
    p.add_argument("--max-patients", type=int, default=None,
                   help="Debug limit after sharding")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def find_ct_path(patient_dir: Path) -> Path | None:
    """Locate the raw CT NIfTI inside ``rawdata/sub-X/ses-*/anat/``.

    Returns ``None`` if no CT is found. PETWB-REP's BIDS layout puts one
    CT per session under ``anat/*_ct.nii.gz``; we take the first match.
    """
    matches = sorted(patient_dir.glob("ses-*/anat/*_ct.nii.gz"))
    if not matches:
        return None
    return matches[0]


def mask_z_extent_mm(mask_path: Path) -> tuple[float, float] | None:
    """Return (z_min_mm, z_max_mm) of non-zero voxels, or None if empty.

    We reload each mask via canonical RAS so that the z-axis sign is
    consistent across patients regardless of the original acquisition
    orientation. ``world_z_per_slice`` then maps slice index → physical
    mm directly off the affine.
    """
    img = load_canonical(mask_path)
    data = np.asarray(img.dataobj)
    if data.size == 0:
        return None
    # Reduce over x,y → 1-D per-slice "any voxel set" mask. Cheaper than
    # any/all per voxel for typical 200³ masks.
    per_slice = data.any(axis=(0, 1))
    if not per_slice.any():
        return None
    z = world_z_per_slice(img)
    z_present = z[per_slice]
    return float(z_present.min()), float(z_present.max())


def segment_one_patient(
    ct_path: Path,
    tmp_dir: Path,
    device: str,
    fast: bool,
) -> dict[str, Path]:
    """Run TotalSegmentator on one CT and return paths to the per-ROI masks.

    We import the python_api inside the function so the script can be
    imported (e.g. for ``--help``) on machines without TotalSegmentator
    installed; the actual call only happens at run time.
    """
    from totalsegmentator.python_api import totalsegmentator

    # Multi-label off → one file per ROI in ``tmp_dir``, named after the
    # ROI class (e.g. ``lung_upper_lobe_left.nii.gz``). Lighter than
    # multi-label parsing when we only need bounding boxes.
    totalsegmentator(
        input=str(ct_path),
        output=str(tmp_dir),
        task="total",
        roi_subset=ALL_ROIS,
        ml=False,
        fast=fast,
        device=device,
        quiet=True,
        skip_saving=False,
    )

    mask_paths: dict[str, Path] = {}
    for roi in ALL_ROIS:
        # TotalSegmentator names outputs ``<roi>.nii.gz``. If a class
        # didn't segment (e.g. the patient is missing that anatomy) we
        # tolerate it — see compute_landmarks for the union logic.
        path = tmp_dir / f"{roi}.nii.gz"
        if path.exists():
            mask_paths[roi] = path
        else:
            logger.warning("missing mask %s for %s", roi, ct_path.name)
    return mask_paths


def compute_landmarks(
    mask_paths: dict[str, Path],
    ct_path: Path,
) -> dict[str, Any]:
    """Reduce per-ROI masks to a {z_apex, z_base, z_pelvis_floor} dict.

    Returned mm values are in canonical-RAS world coordinates so they
    transfer directly to the PET volume (shared Frame of Reference) at
    cropping time. ``scan_top``/``scan_bottom`` are the CT's full
    z-extent; the preprocessor needs them to bound the head-neck top
    and abd-pelvis bottom crops, since TotalSegmentator doesn't tell us
    those (lungs don't reach the vertex; pelvis doesn't reach mid-thigh).
    """
    # Scan extent: read off the CT directly.
    ct_img = load_canonical(ct_path)
    z_axis = world_z_per_slice(ct_img)
    scan_bottom_mm = float(z_axis.min())
    scan_top_mm = float(z_axis.max())

    # Lung apex = max-z over the union of all lung lobes that segmented.
    # Lung base = min-z over the same union. Robust to single-lobe miss.
    lung_extents = [mask_z_extent_mm(mask_paths[r]) for r in LUNG_ROIS if r in mask_paths]
    lung_extents = [e for e in lung_extents if e is not None]
    if not lung_extents:
        raise RuntimeError(f"No lung mask voxels detected for {ct_path.name}")
    z_apex_mm = max(e[1] for e in lung_extents)
    z_base_mm = min(e[0] for e in lung_extents)

    # Pelvis floor = inferior-most z of hip/sacrum union.
    pelvis_extents = [mask_z_extent_mm(mask_paths[r]) for r in PELVIS_ROIS if r in mask_paths]
    pelvis_extents = [e for e in pelvis_extents if e is not None]
    if not pelvis_extents:
        # Some patients (extremity-skipped scans) may not include pelvis.
        # Fall back to scan_bottom so abd-pelvis still crops, just to the
        # full inferior extent. Log so it's discoverable in the cohort
        # summary.
        logger.warning("no pelvis mask for %s; using scan_bottom as floor", ct_path.name)
        z_pelvis_floor_mm = scan_bottom_mm
    else:
        z_pelvis_floor_mm = min(e[0] for e in pelvis_extents)

    return {
        "ct_path": str(ct_path),
        "scan_bottom_mm": scan_bottom_mm,
        "scan_top_mm": scan_top_mm,
        "z_apex_mm": z_apex_mm,
        "z_base_mm": z_base_mm,
        "z_pelvis_floor_mm": z_pelvis_floor_mm,
        # Convenience derived spans (for QC). H&N height should be
        # ~150-300mm, chest ~250-350, abd-pelvis ~300-500. Outliers
        # flagged here surface segmentation failures without re-deriving.
        "headneck_span_mm": scan_top_mm - z_apex_mm,
        "chest_span_mm": z_apex_mm - z_base_mm,
        "abdpelvis_span_mm": z_base_mm - z_pelvis_floor_mm,
    }


def process_patient(
    patient_dir: Path,
    out_dir: Path,
    device: str,
    fast: bool,
    overwrite: bool,
) -> tuple[bool, str]:
    """Return ``(success, message)``. Writes landmarks JSON on success."""
    patient_id = patient_dir.name
    out_path = out_dir / f"{patient_id}.json"
    if out_path.exists() and not overwrite:
        return True, "skipped (cached)"

    ct_path = find_ct_path(patient_dir)
    if ct_path is None:
        return False, "no CT found"

    # Run segmentation into a per-patient tempdir so concurrent shards
    # don't clobber each other's intermediates. The masks are large
    # enough that keeping them on disk after we extract z-bounds is
    # wasteful; tempfile auto-cleanup handles it.
    with tempfile.TemporaryDirectory(prefix=f"ts_{patient_id}_") as td:
        try:
            mask_paths = segment_one_patient(ct_path, Path(td), device=device, fast=fast)
        except Exception as exc:
            return False, f"TotalSegmentator failed: {exc!r}\n{traceback.format_exc(limit=2)}"

        try:
            landmarks = compute_landmarks(mask_paths, ct_path)
        except Exception as exc:
            return False, f"landmark extraction failed: {exc!r}"

    landmarks["patient_id"] = patient_id
    out_path.write_text(json.dumps(landmarks, indent=2))
    return True, "ok"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rawdata = args.petwbrep_root / "rawdata"
    if not rawdata.is_dir():
        sys.exit(f"rawdata/ not found under {args.petwbrep_root}")

    patient_dirs = sorted(p for p in rawdata.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    shard = [p for i, p in enumerate(patient_dirs) if i % args.num_shards == args.shard_index]
    if args.max_patients is not None:
        shard = shard[: args.max_patients]

    logger.info(
        "shard %d/%d: %d patients (cohort total %d)",
        args.shard_index, args.num_shards, len(shard), len(patient_dirs),
    )

    n_ok = n_skipped = n_fail = 0
    failures_log = out_dir / f"failures_shard{args.shard_index:04d}.log"
    # tqdm: one tick per patient. Per-patient TotalSegmentator takes ~10-30s
    # on GPU so progress is human-meaningful; the postfix surfaces the
    # running ok/skip/fail tallies so failures don't get lost between bars.
    # ``tqdm.write`` is used for failure log lines instead of ``print`` /
    # ``logger.warning`` so the progress bar stays intact at the bottom of
    # the terminal.
    pbar = tqdm(
        shard,
        total=len(shard),
        desc=f"shard {args.shard_index}",
        unit="patient",
        dynamic_ncols=True,
    )
    with failures_log.open("w") as flog:
        for patient_dir in pbar:
            ok, msg = process_patient(
                patient_dir, out_dir,
                device=args.device, fast=args.fast,
                overwrite=args.overwrite,
            )
            if ok:
                if "skipped" in msg:
                    n_skipped += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                flog.write(f"{patient_dir.name}\t{msg}\n")
                flog.flush()
                tqdm.write(f"FAILED {patient_dir.name}: {msg.splitlines()[0]}")
            pbar.set_postfix(ok=n_ok, skip=n_skipped, fail=n_fail)
    pbar.close()

    logger.info(
        "DONE shard %d: ok=%d skip=%d fail=%d (failures in %s)",
        args.shard_index, n_ok, n_skipped, n_fail, failures_log,
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
