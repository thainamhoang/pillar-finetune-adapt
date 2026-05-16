"""Pillar-initialized Atlas wrappers for chest CT/PET dual-stream experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from pillar.models.backbones.mmatlas import MultimodalAtlas


@dataclass
class EncoderConfig:
    input_channels: int
    anatomy: str = "chest_ct"
    model_repo_id: str = "YalaLab/Pillar0-ChestCT"
    model_revision: str = "main"
    device: Optional[str] = None
    patch_embed_init: str = "kaiming"


def _find_first_conv3d(module: nn.Module) -> tuple[str, nn.Conv3d]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv3d):
            return name, child
    raise ValueError("Could not find a Conv3d patch-embed layer in the Atlas visual backbone")


def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _adapt_first_conv3d(
    visual: nn.Module,
    target_in_channels: int,
    init: str = "kaiming",
) -> str:
    name, conv = _find_first_conv3d(visual)
    if conv.in_channels == target_in_channels:
        return name

    new_conv = nn.Conv3d(
        in_channels=target_in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )

    with torch.no_grad():
        if init == "mean" and target_in_channels <= conv.in_channels:
            pooled = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(pooled.repeat(1, target_in_channels, 1, 1, 1))
        elif init == "mean" and target_in_channels > conv.in_channels:
            new_conv.weight.zero_()
            new_conv.weight[:, : conv.in_channels].copy_(conv.weight)
            mean_weight = conv.weight.mean(dim=1, keepdim=True)
            repeat = target_in_channels - conv.in_channels
            new_conv.weight[:, conv.in_channels :].copy_(mean_weight.repeat(1, repeat, 1, 1, 1))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if new_conv.bias is not None:
            if conv.bias is not None and target_in_channels == conv.in_channels:
                new_conv.bias.copy_(conv.bias)
            else:
                new_conv.bias.zero_()

    _set_module_by_name(visual, name, new_conv)
    return name


def _resize_posemb_sequence(pe: torch.Tensor, target_tokens: int) -> torch.Tensor:
    squeeze_batch = False
    if pe.ndim == 2:
        pe = pe.unsqueeze(0)
        squeeze_batch = True
    if pe.ndim != 3:
        return pe
    if pe.shape[1] == target_tokens:
        return pe.squeeze(0) if squeeze_batch else pe

    pe = F.interpolate(
        pe.transpose(1, 2),
        size=target_tokens,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return pe.squeeze(0) if squeeze_batch else pe


def _build_relative_posemb(module: nn.Module, x: torch.Tensor, grid_size, modality: str) -> torch.Tensor:
    """Recreate RelativePosEmb generation without assuming token-length match."""
    if module.training:
        module.modality_grid_exists = {}
        module.posemb = {}

    if modality not in module.modality_grid_exists:
        module.modality_grid_exists[modality] = True
        h, w, d = grid_size

        relative_coords_h = torch.arange(0, h, device=x.device, dtype=x.dtype)
        relative_coords_w = torch.arange(0, w, device=x.device, dtype=x.dtype)
        relative_coords_d = torch.arange(0, d, device=x.device, dtype=x.dtype)

        relative_coords_table = (
            torch.stack(
                torch.meshgrid(
                    [relative_coords_h, relative_coords_w, relative_coords_d],
                    indexing="ij",
                )
            )
            .contiguous()
            .unsqueeze(0)
        )

        if h > 1:
            relative_coords_table[0, 0] -= h // 2
            relative_coords_table[0, 0] /= h // 2
        if w > 1:
            relative_coords_table[0, 1] -= w // 2
            relative_coords_table[0, 1] /= w // 2
        if d > 1:
            relative_coords_table[0, 2] -= d // 2
            relative_coords_table[0, 2] /= d // 2

        relative_coords_table = relative_coords_table.float()
        if not module.conv:
            posemb = module.cpb_mlp(
                relative_coords_table.permute(0, 2, 3, 4, 1).reshape(-1, h * w * d, 3)
            )
        else:
            posemb = module.cpb_mlp(relative_coords_table.squeeze(0).reshape(3, -1))
        module.posemb[modality] = posemb

    return module.posemb[modality]


def _patch_posemb_modules(root: nn.Module) -> None:
    for module in root.modules():
        posemb = getattr(module, "posemb", None)
        if posemb is None or getattr(module, "_vimed_posemb_patched", False):
            continue
        if module.__class__.__name__ != "RelativePosEmb":
            continue
        original_forward = module.forward

        def forward_with_interp(self, x: torch.Tensor, grid_size=(8, 8, 5), modality="chest_ct") -> torch.Tensor:
            target_tokens = x.shape[1]
            pe = _build_relative_posemb(self, x, grid_size, modality)
            pe_tokens = pe.shape[1]
            if pe_tokens != target_tokens:
                if target_tokens % pe_tokens == 0:
                    pe = pe.repeat(1, target_tokens // pe_tokens, 1)
                else:
                    pe = _resize_posemb_sequence(pe, target_tokens)
            return x + pe

        module.forward = MethodType(forward_with_interp, module)
        module._vimed_posemb_patched = True


class PillarInitializedAtlasEncoder(nn.Module):
    """Atlas encoder with Pillar-initialized body and reinitialized input layer.

    The body weights are loaded from the Pillar checkpoint. When the requested
    channel count differs from the pretrained CT tokenizer, the first Conv3d
    patch-embed layer is replaced and reinitialized.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.backbone = MultimodalAtlas(
            args=None,
            device=config.device,
            model_repo_id=config.model_repo_id,
            model_revision=config.model_revision,
            pretrained=True,
        )
        replaced_name = _adapt_first_conv3d(
            self.backbone.visual,
            target_in_channels=config.input_channels,
            init=config.patch_embed_init,
        )
        _patch_posemb_modules(self.backbone.visual)
        self.input_adapter_name = replaced_name
        self.hidden_dim = self.backbone.hidden_dim

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B,C,D,H,W) or (C,D,H,W), got {tuple(x.shape)}")

        modality_cfg = self.backbone.visual.model_config["modalities"][self.config.anatomy]
        modality_cfg["image_size"] = list(x.shape[-3:])
        batch = {"anatomy": [self.config.anatomy] * x.shape[0]}
        return self.backbone(x, batch=batch)


class ViMedChestDualStreamEncoders(nn.Module):
    """Convenience wrapper holding both CT and PET Atlas encoders."""

    def __init__(
        self,
        ct_channels: int = 6,
        pet_channels: int = 4,
        anatomy: str = "chest_ct",
        model_repo_id: str = "YalaLab/Pillar0-ChestCT",
        model_revision: str = "main",
        device: Optional[str] = None,
        patch_embed_init: str = "kaiming",
    ) -> None:
        super().__init__()
        common = dict(
            anatomy=anatomy,
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            device=device,
            patch_embed_init=patch_embed_init,
        )
        self.ct_encoder = PillarInitializedAtlasEncoder(
            EncoderConfig(input_channels=ct_channels, **common)
        )
        self.pet_encoder = PillarInitializedAtlasEncoder(
            EncoderConfig(input_channels=pet_channels, **common)
        )

    def forward(self, ct_windows: torch.Tensor, pet_windows: torch.Tensor) -> dict:
        ct_out = self.ct_encoder(ct_windows)
        pet_out = self.pet_encoder(pet_windows)
        return {
            "ct_activ": ct_out["activ"],
            "ct_pooled": ct_out["pooled"],
            "pet_activ": pet_out["activ"],
            "pet_pooled": pet_out["pooled"],
        }
