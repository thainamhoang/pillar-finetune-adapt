#!/usr/bin/env python3
"""Cohort-level geometry diagnostic for PETWB-REP region crops.

After ``segment_landmarks_petwbrep.py`` has produced per-patient
``landmarks.json``, this script applies the same physical-mm crop logic
the main preprocessor would use, but **stops before resampling** and
just reports the native shape of each region crop.

Why a separate diagnostic: we want to know what target tensor shape to
resample to before committing the main preprocessor to 256³ (chest)
or 384³ (abd-pelvis). If the cohort's native chest crop is, say, 90
slices, then resampling to 256 is a 2.8x z upsample -- pure
interpolation, no new information. That argues for resampling to a
shape closer to the native count (or switching to an isotropic-mm
target with center-crop/pad). If the native count is closer to 160-200,
resample-to-256 interpolates modestly and is fine.

Output is a single CSV row per ``(patient, region)`` so the decision
can be made in pandas afterward (mean/median/percentiles/outliers).

This script is **fast** -- it reads only the NIfTI header (affine,
shape, pixdim) and computes index arithmetic. No voxel data is loaded,
no GPU needed. 490 patients × 3 regions should finish in ~30 seconds.

Usage::

    python scripts/scan_petwbrep_geometry.py \\
        --petwbrep-root /scratch/.../PETWB-REP \\
        --landmarks-dir /scratch/.../PETWB-REP_landmarks \\
        --out-csv       /scratch/.../petwbrep_geometry.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pillar.utils.nifti import load_canonical, world_z_per_slice


logger = logging.getLogger("scan_petwbrep_geometry")


REGION_KEYS = ("head_neck", "chest", "abdomen_pelvis")

# Pillar-0 per-region target shapes (Table 7) -- used only to compute
# resample factors for the diagnostic columns. Must match
# ``REGION_TARGET_SHAPE`` in preprocess_petwbrep.py.
REGION_TARGET_DEPTH = {
    "head_neck": 128,
    "chest": 256,
    "abdomen_pelvis": 384,
}

DEFAULT_OVERLAP_MM = 25.0


# NOTE: keep this function in sync with
# preprocess_petwbrep.compute_region_bounds_mm. Duplicated here rather
# than imported to avoid a script-to-script import in a code path that
# can't risk breaking. If the boundary logic ever drifts, both files
# must move together.
def region_bounds_mm(
    region: str,
    landmarks: dict,
    overlap_mm: float,
) -> tuple[float, float]:
    z_apex = landmarks["z_apex_mm"]
    z_base = landmarks["z_base_mm"]
    z_pelvis = landmarks["z_pelvis_floor_mm"]
    z_top = landmarks["scan_top_mm"]
    z_bot = landmarks["scan_bottom_mm"]
    if region == "head_neck":
        return (z_apex - overlap_mm, z_top)
    if region == "chest":
        return (z_base - overlap_mm, z_apex + overlap_mm)
    if region == "abdomen_pelvis":
        return (max(z_bot, z_pelvis), z_base + overlap_mm)
    raise ValueError(f"Unknown region: {region}")


def find_ct_path(patient_dir: Path) -> Path | None:
    matches = sorted(patient_dir.glob("ses-*/anat/*_ct.nii.gz"))
    return matches[0] if matches else None


def scan_patient(
    patient_dir: Path,
    landmarks_dir: Path,
    overlap_mm: float,
) -> list[dict] | None:
    """Return one row per region for the patient, or None if skipped."""
    patient_id = patient_dir.name
    landmarks_path = landmarks_dir / f"{patient_id}.json"
    if not landmarks_path.exists():
        logger.warning("%s: no landmarks JSON", patient_id)
        return None

    ct_path = find_ct_path(patient_dir)
    if ct_path is None:
        logger.warning("%s: no CT", patient_id)
        return None

    landmarks = json.loads(landmarks_path.read_text())

    # Header-only read: load_canonical() uses ``nib.load`` which lazy-
    # loads; we never touch dataobj, only the affine + shape.
    img = load_canonical(ct_path)
    nx, ny, nz = img.shape[:3]
    # In canonical RAS, affine diagonal magnitudes are the per-axis
    # voxel sizes in mm. Use the column norms in case there's any
    # rotation residue (there shouldn't be after canonicalisation, but
    # belt-and-braces).
    dx = float(np.linalg.norm(img.affine[:3, 0]))
    dy = float(np.linalg.norm(img.affine[:3, 1]))
    dz = float(np.linalg.norm(img.affine[:3, 2]))

    z_axis = world_z_per_slice(img)
    scan_z_lo_mm = float(z_axis.min())
    scan_z_hi_mm = float(z_axis.max())

    rows = []
    for region in REGION_KEYS:
        z_lo, z_hi = region_bounds_mm(region, landmarks, overlap_mm)
        mask = (z_axis >= z_lo) & (z_axis <= z_hi)
        crop_n_slices = int(mask.sum())
        crop_span_mm = float(z_hi - z_lo)
        target_depth = REGION_TARGET_DEPTH[region]
        # Resample factor on z = target_depth / crop_n_slices. Values >1
        # are upsampling (interpolation only, no new info); <1 is
        # downsampling. The diagnostic question is: how far from 1 is
        # this across the cohort?
        z_resample_factor = (
            float(target_depth) / float(crop_n_slices)
            if crop_n_slices > 0 else float("inf")
        )
        rows.append({
            "patient_id": patient_id,
            "region": region,
            # Raw CT geometry (same for all 3 region rows; repeated for
            # easy pandas aggregation).
            "raw_n_x": nx,
            "raw_n_y": ny,
            "raw_n_z": nz,
            "raw_dx_mm": round(dx, 4),
            "raw_dy_mm": round(dy, 4),
            "raw_dz_mm": round(dz, 4),
            "raw_extent_x_mm": round(nx * dx, 2),
            "raw_extent_y_mm": round(ny * dy, 2),
            "raw_extent_z_mm": round(nz * dz, 2),
            "scan_z_lo_mm": round(scan_z_lo_mm, 2),
            "scan_z_hi_mm": round(scan_z_hi_mm, 2),
            # Region-specific crop.
            "crop_z_lo_mm": round(z_lo, 2),
            "crop_z_hi_mm": round(z_hi, 2),
            "crop_span_mm": round(crop_span_mm, 2),
            "crop_n_slices": crop_n_slices,
            "crop_extent_x_mm": round(nx * dx, 2),  # in-plane unchanged
            "crop_extent_y_mm": round(ny * dy, 2),
            # Target encoder dims + implied resample factor on z.
            "target_depth": target_depth,
            "z_resample_factor": round(z_resample_factor, 4),
            # The factor we'd apply in-plane to reach 256/384 from 512.
            "xy_resample_factor": round(256 / nx, 4) if region != "abdomen_pelvis"
                                  else round(384 / nx, 4),
            # Landmark z-positions (for cross-reference / QC).
            "z_apex_mm": round(landmarks["z_apex_mm"], 2),
            "z_base_mm": round(landmarks["z_base_mm"], 2),
            "z_pelvis_floor_mm": round(landmarks["z_pelvis_floor_mm"], 2),
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--petwbrep-root", type=Path, required=True)
    p.add_argument("--landmarks-dir", type=Path, required=True,
                   help="Output of segment_landmarks_petwbrep.py")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--overlap-mm", type=float, default=DEFAULT_OVERLAP_MM,
                   help="Per-side overlap at region boundaries; must match "
                        "preprocess_petwbrep.py --overlap-mm.")
    p.add_argument("--max-patients", type=int, default=None,
                   help="Debug limit")
    p.add_argument("--print-summary", action="store_true",
                   help="After the CSV is written, print mean/median/p5/p95 "
                        "of crop_n_slices and z_resample_factor per region.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def print_summary(csv_path: Path) -> None:
    """Tiny per-region distribution summary, computed without pandas."""
    by_region: dict[str, list[dict]] = {r: [] for r in REGION_KEYS}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            by_region[row["region"]].append(row)

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"n": 0}
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    print("\n=== Cohort geometry summary ===")
    for region, rows in by_region.items():
        slices = stats([float(r["crop_n_slices"]) for r in rows])
        spans = stats([float(r["crop_span_mm"]) for r in rows])
        factors = stats([float(r["z_resample_factor"]) for r in rows])
        print(f"\n[{region}]  n={slices.get('n', 0)} patients")
        if slices.get("n", 0) == 0:
            continue
        print(f"  native z slices:    "
              f"mean={slices['mean']:.0f}  median={slices['median']:.0f}  "
              f"std={slices['std']:.0f}  p5={slices['p5']:.0f}  "
              f"p95={slices['p95']:.0f}  range=[{slices['min']:.0f}, {slices['max']:.0f}]")
        print(f"  span mm:            "
              f"mean={spans['mean']:.0f}  median={spans['median']:.0f}  "
              f"p5={spans['p5']:.0f}  p95={spans['p95']:.0f}")
        target = REGION_TARGET_DEPTH[region]
        print(f"  z resample factor to {target}: "
              f"mean={factors['mean']:.2f}x  median={factors['median']:.2f}x  "
              f"p5={factors['p5']:.2f}x  p95={factors['p95']:.2f}x")
        # Flag patients where the upsample factor is extreme.
        extreme = [r["patient_id"] for r in rows
                   if float(r["z_resample_factor"]) > 3.0]
        if extreme:
            print(f"  upsample > 3x ({len(extreme)} patients): "
                  f"{', '.join(extreme[:5])}{'...' if len(extreme) > 5 else ''}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rawdata = args.petwbrep_root / "rawdata"
    if not rawdata.is_dir():
        sys.exit(f"rawdata/ not found under {args.petwbrep_root}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in rawdata.iterdir()
                          if p.is_dir() and p.name.startswith("sub-"))
    if args.max_patients is not None:
        patient_dirs = patient_dirs[: args.max_patients]
    logger.info("scanning %d patients × 3 regions", len(patient_dirs))

    fieldnames = [
        "patient_id", "region",
        "raw_n_x", "raw_n_y", "raw_n_z",
        "raw_dx_mm", "raw_dy_mm", "raw_dz_mm",
        "raw_extent_x_mm", "raw_extent_y_mm", "raw_extent_z_mm",
        "scan_z_lo_mm", "scan_z_hi_mm",
        "crop_z_lo_mm", "crop_z_hi_mm", "crop_span_mm", "crop_n_slices",
        "crop_extent_x_mm", "crop_extent_y_mm",
        "target_depth", "z_resample_factor", "xy_resample_factor",
        "z_apex_mm", "z_base_mm", "z_pelvis_floor_mm",
    ]

    n_ok = n_skip = 0
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, patient_dir in enumerate(patient_dirs, start=1):
            rows = scan_patient(patient_dir, args.landmarks_dir, args.overlap_mm)
            if rows is None:
                n_skip += 1
            else:
                for r in rows:
                    writer.writerow(r)
                n_ok += 1
            if i % 50 == 0:
                logger.info("[%d/%d] ok=%d skip=%d", i, len(patient_dirs), n_ok, n_skip)

    logger.info("wrote %s (ok=%d skip=%d)", args.out_csv, n_ok, n_skip)

    if args.print_summary:
        print_summary(args.out_csv)


if __name__ == "__main__":
    main()
