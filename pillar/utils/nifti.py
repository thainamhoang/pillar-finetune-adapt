"""NIfTI loading + physical-coordinate cropping + resampling.

The other ViMED preprocessing path operates on pre-extracted ``.npz`` arrays
with slice-index cropping (see ``preprocess_vimed_petct.py`` in the sibling
ViMED repo). PETWB-REP ships ``.nii.gz`` directly with affines, and we crop
CT vs. PET on **different native grids** but the **same physical-mm
coordinate system** (DICOM Frame of Reference is shared per study). The
helpers in this module take that as the contract: define one crop in mm,
apply it to either modality via that modality's own affine.

All functions assume **canonical RAS** orientation after ``load_canonical``
so that axis 2 (k) is superior-inferior, increasing-k = superior. This
matches nibabel's ``as_closest_canonical`` and lets us read world z directly
off the affine without per-call reorientation arithmetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


__all__ = [
    "load_canonical",
    "world_z_per_slice",
    "scan_extent_mm",
    "crop_physical_z",
    "resample_volume",
    "voxel_array",
]


def load_canonical(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI and force canonical RAS orientation.

    In RAS, axis 2 (the third spatial axis) runs inferior→superior with
    larger index = more superior, so ``affine[2, 3]`` is the world-z of
    slice 0 and ``affine[2, 2]`` is the slice spacing (positive). Every
    downstream call assumes this convention.
    """
    img = nib.load(str(path))
    return nib.as_closest_canonical(img)


def world_z_per_slice(img: nib.Nifti1Image) -> np.ndarray:
    """Physical z (mm) of each axial slice, after canonical reorient.

    Returns a 1-D array of length ``img.shape[2]``. Use this for boundary
    lookups and for sanity-checking that a slice index maps where you
    expect anatomically.
    """
    nslices = int(img.shape[2])
    k = np.arange(nslices, dtype=np.float64)
    # After canonical RAS, voxel (0, 0, k) world-z = affine[2,3] + k * affine[2,2].
    return img.affine[2, 3] + k * img.affine[2, 2]


def scan_extent_mm(img: nib.Nifti1Image) -> tuple[float, float]:
    """Return ``(z_bottom_mm, z_top_mm)`` for a canonical-RAS volume.

    ``z_bottom`` is the inferior (smallest-z, e.g. mid-thigh) physical
    position; ``z_top`` is the superior (largest-z, e.g. vertex) position.
    """
    z = world_z_per_slice(img)
    return float(z.min()), float(z.max())


def crop_physical_z(
    img: nib.Nifti1Image,
    z_lo_mm: float,
    z_hi_mm: float,
) -> nib.Nifti1Image:
    """Crop a canonical-RAS volume to a physical z-range in mm.

    Selects slices whose world-z falls within ``[z_lo_mm, z_hi_mm]``
    (inclusive on both ends) and returns a new ``Nifti1Image`` whose
    affine origin is shifted so the cropped volume's geometry remains
    valid in world coordinates.

    Raises ``ValueError`` if the range selects zero slices, since that
    is almost certainly a logic bug at the caller (e.g. landmark JSON
    out of sync with the volume).
    """
    z = world_z_per_slice(img)
    mask = (z >= z_lo_mm) & (z <= z_hi_mm)
    if not mask.any():
        raise ValueError(
            f"crop_physical_z selected 0 slices for range [{z_lo_mm:.1f}, "
            f"{z_hi_mm:.1f}] mm; scan covers [{z.min():.1f}, {z.max():.1f}] mm."
        )

    # Slice the data array along k. Use the proxy so we don't materialise
    # the whole volume if we only need a sub-range — nibabel handles this
    # transparently via the array proxy.
    data = np.asarray(img.dataobj)[:, :, mask]

    new_aff = img.affine.copy()
    new_aff[2, 3] = float(z[mask].min())
    return nib.Nifti1Image(data, new_aff, img.header)


def voxel_array(img: nib.Nifti1Image, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get scaled voxel data as a contiguous numpy array.

    Uses ``get_fdata`` so that ``scl_slope``/``scl_inter`` (and any
    derivative-time scaling baked in by sitk/dcm2niix) are applied. The
    PETWB-REP derivative SUV files rely on this — their ``scl_slope`` of
    1.708... is what converts the on-disk uint16 to Bq/mL, which is then
    further multiplied during their preprocessing to land in SUVbw.
    """
    arr = img.get_fdata(dtype=np.float64).astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def resample_volume(
    volume_dhw: np.ndarray,
    dst_shape_dhw: tuple[int, int, int],
    mode: Literal["area", "trilinear"] = "trilinear",
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Resample a 3-D volume to a target ``(D, H, W)`` shape.

    Parameters
    ----------
    volume_dhw:
        ``(D, H, W)`` source volume. Float32 in / float32 out.
    dst_shape_dhw:
        Target ``(D, H, W)``. Pillar-0 chest expects ``(256, 256, 256)``,
        head ``(128, 256, 256)``, abdomen-pelvis ``(384, 384, 384)`` per
        Table 7 of the Pillar-0 paper.
    mode:
        ``"area"`` for pure downsampling (anti-aliased block average; the
        right choice for CT 512→256 in-plane like the existing ViMED path
        uses). ``"trilinear"`` for any other case, including any upsample
        leg. Use ``trilinear`` for PET because the native PET grid is
        already coarse and we may need to upsample along z.
    pad_value:
        Currently unused; kept in the signature to match
        ``preprocess_vimed_petct.fit_depth``'s pad-value convention so
        callers can pass ``-1024.0`` for CT and ``0.0`` for PET/SUV when
        we later want a pad-rather-than-resample path.

    Notes
    -----
    Unlike ``preprocess_vimed_petct.py`` which resamples in-plane only and
    then center-crops/zero-pads along depth, we resample **all three
    axes** at once. The two paths produce different geometry: depth
    resampling preserves the cropped anatomic span; depth-padding shifts
    the crop into a fixed-depth window. Pillar-0 expects a fixed
    ``(D, H, W)`` shape per modality and is invariant to which path you
    took to get there — we choose resampling because it preserves the
    full physical extent of the region crop without padding artifacts
    creeping into multi-window CT channels.
    """
    if volume_dhw.ndim != 3:
        raise ValueError(
            f"resample_volume expected 3-D (D,H,W), got shape {volume_dhw.shape}"
        )
    del pad_value  # not used in resample mode

    src = torch.from_numpy(np.ascontiguousarray(volume_dhw)).float()
    src = src.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    if mode == "area":
        out = F.interpolate(src, size=dst_shape_dhw, mode="area")
    else:
        out = F.interpolate(
            src, size=dst_shape_dhw, mode="trilinear", align_corners=False
        )
    return out.squeeze(0).squeeze(0).cpu().numpy()
