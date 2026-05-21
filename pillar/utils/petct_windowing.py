"""PET/CT windowing helpers for ViMED chest dual-stream experiments."""

from __future__ import annotations
from collections import OrderedDict
import torch


DUAL_STREAM_CT_WINDOWS = OrderedDict(
    [
        ("lung", {"center": -600.0, "width": 1500.0}),
        ("mediastinum", {"center": 50.0, "width": 400.0}),
        ("abdomen", {"center": 40.0, "width": 400.0}),
        ("liver", {"center": 80.0, "width": 150.0}),
        ("bone", {"center": 400.0, "width": 1800.0}),
        ("brain", {"center": 40.0, "width": 80.0}),
        ("subdural", {"center": 75.0, "width": 215.0}),
        ("stroke", {"center": 40.0, "width": 40.0}),
        ("temporal_bone", {"center": 600.0, "width": 2800.0}),
        ("soft_tissue", {"center": 50.0, "width": 350.0}),
        ("minmax", {"lo": -1024.0, "hi": 3071.0}),
    ]
)

DUAL_STREAM_PET_WINDOWS = OrderedDict(
    [
        ("pet_low", {"lo": 0.00, "hi": 0.25}),
        ("pet_mid", {"lo": 0.00, "hi": 0.50}),
        ("pet_high", {"lo": 0.00, "hi": 0.75}),
        ("pet_full", {"lo": 0.00, "hi": 1.00}),
    ]
)


def _ct_window_constants() -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-stacked (low, divisor) pairs for the 11 CT windows, shape (11, 1, 1, 1)."""
    lows, divs = [], []
    for spec in DUAL_STREAM_CT_WINDOWS.values():
        if "center" in spec:
            lows.append(spec["center"] - spec["width"] / 2.0)
            divs.append(spec["width"])
        else:
            lows.append(spec["lo"])
            divs.append(spec["hi"] - spec["lo"])
    return (
        torch.tensor(lows, dtype=torch.float32).view(-1, 1, 1, 1),
        torch.tensor(divs, dtype=torch.float32).view(-1, 1, 1, 1),
    )


def _pet_window_constants() -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-stacked (low, divisor) pairs for the 4 PET windows, shape (4, 1, 1, 1)."""
    lows = torch.tensor([s["lo"] for s in DUAL_STREAM_PET_WINDOWS.values()], dtype=torch.float32)
    divs = torch.tensor(
        [s["hi"] - s["lo"] + 1e-6 for s in DUAL_STREAM_PET_WINDOWS.values()],
        dtype=torch.float32,
    )
    return lows.view(-1, 1, 1, 1), divs.view(-1, 1, 1, 1)


_CT_LOWS, _CT_DIVS = _ct_window_constants()
_PET_LOWS, _PET_DIVS = _pet_window_constants()


def _broadcast_window_inplace(
    volume_dhw: torch.Tensor,
    lows: torch.Tensor,
    divs: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create window channels with one CxDxHxW allocation.

    The natural broadcast expression ``(x - lows) / divs`` materializes a
    large intermediate tensor before division. Persistent dataloader workers
    tend to keep those allocator arenas around, which makes ViMED runs look
    like a host-RAM leak. Expanding then cloning once lets the arithmetic stay
    in-place on the final output tensor.
    """
    volume = volume_dhw.to(dtype=dtype, copy=False)
    out = volume.unsqueeze(0).expand(lows.shape[0], *volume.shape).clone()
    lows = lows.to(device=out.device, dtype=out.dtype)
    divs = divs.to(device=out.device, dtype=out.dtype)
    out.sub_(lows).div_(divs).clamp_(0.0, 1.0)
    return out


def make_ct_windows_fast(ct_dhw: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Vectorized CT windowing.

    Input: ``(D, H, W)`` HU tensor (or ``(1, D, H, W)``; the leading 1 is
    squeezed). Output: ``(11, D, H, W)`` in ``[0, 1]`` using ``dtype``.

    Computes all 11 Pillar/RAVE-style CT windows in a single broadcasted op instead
    of looping over window specs and concatenating, removing Python
    iterations + a ``torch.cat`` from the per-sample CPU path.
    """
    if ct_dhw.ndim == 4:
        ct_dhw = ct_dhw.squeeze(0)
    return _broadcast_window_inplace(ct_dhw, _CT_LOWS, _CT_DIVS, dtype=dtype)


def make_pet_windows_fast(pet_dhw: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Vectorized PET windowing.

    Input: ``(D, H, W)`` PET tensor normalized to ``[0, 1]`` (or
    ``(1, D, H, W)``). Output: ``(4, D, H, W)`` using ``dtype``.
    """
    if pet_dhw.ndim == 4:
        pet_dhw = pet_dhw.squeeze(0)
    return _broadcast_window_inplace(pet_dhw, _PET_LOWS, _PET_DIVS, dtype=dtype)


def _window_ct(ct: torch.Tensor, center: float, width: float) -> torch.Tensor:
    low = center - width / 2.0
    return torch.clamp((ct - low) / width, 0.0, 1.0)


def _window_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _ensure_volume(x: torch.Tensor, expected_channels: int = 1) -> tuple[torch.Tensor, bool]:
    squeeze_batch = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim == 4:
        x = x.unsqueeze(0)
        squeeze_batch = True
    if x.ndim != 5 or x.shape[1] != expected_channels:
        raise ValueError(
            f"Expected shape (B,{expected_channels},D,H,W) or "
            f"({expected_channels},D,H,W) or (D,H,W), got {tuple(x.shape)}"
        )
    return x.float(), squeeze_batch


def make_ct_window_tensor(
    ct: torch.Tensor,
    window_specs: OrderedDict[str, dict] = DUAL_STREAM_CT_WINDOWS,
) -> torch.Tensor:
    """Create multi-window chest CT channels.

    Input may be (D,H,W), (1,D,H,W), or (B,1,D,H,W).
    Returns (C,D,H,W) or (B,C,D,H,W).
    """
    ct, squeeze_batch = _ensure_volume(ct, expected_channels=1)
    outputs = []
    for spec in window_specs.values():
        if "center" in spec:
            outputs.append(_window_ct(ct, spec["center"], spec["width"]))
        else:
            outputs.append(_window_range(ct, spec["lo"], spec["hi"]))
    out = torch.cat(outputs, dim=1)
    return out.squeeze(0) if squeeze_batch else out


def make_pet_window_tensor(
    pet: torch.Tensor,
    window_specs: OrderedDict[str, dict] = DUAL_STREAM_PET_WINDOWS,
) -> torch.Tensor:
    """Create PET uptake windows from normalized PET in [0,1]."""
    pet, squeeze_batch = _ensure_volume(torch.clamp(pet, 0.0, 1.0), expected_channels=1)
    outputs = [_window_range(pet, spec["lo"], spec["hi"]) for spec in window_specs.values()]
    out = torch.cat(outputs, dim=1)
    return out.squeeze(0) if squeeze_batch else out


def make_dual_stream_window_inputs(x_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert raw two-channel PET/CT into separate CT and PET window tensors."""
    squeeze_batch = False
    if x_raw.ndim == 4:
        x_raw = x_raw.unsqueeze(0)
        squeeze_batch = True
    if x_raw.ndim != 5 or x_raw.shape[1] != 2:
        raise ValueError(f"Expected x_raw shape (B,2,D,H,W) or (2,D,H,W), got {tuple(x_raw.shape)}")

    ct = x_raw[:, 0:1].float()
    pet = torch.clamp(x_raw[:, 1:2].float(), 0.0, 1.0)
    ct_windows = make_ct_window_tensor(ct)
    pet_windows = make_pet_window_tensor(pet)

    if squeeze_batch:
        ct_windows = ct_windows.squeeze(0)
        pet_windows = pet_windows.squeeze(0)

    return {"ct_windows": ct_windows, "pet_windows": pet_windows}
