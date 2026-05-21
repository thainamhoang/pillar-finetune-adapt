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
    # Optional path to a previously-trained encoder state_dict to overlay on
    # top of the HF-pretrained weights. When set, channel adaptation AND
    # overlay loading are routed through ``MultimodalAtlas`` (rather than
    # this module's external ``_adapt_first_conv3d``) so the adapt-then-load
    # order is correct: the freshly target-channel Conv3d shape matches the
    # saved Conv3d (e.g. the trained PET 4ch conv is preserved instead of
    # being dropped by the shape-mismatch filter and re-Kaiming-init'd).
    pretrained_backbone_ckpt: Optional[str] = None


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
        if config.pretrained_backbone_ckpt is not None:
            # Route channel adaptation AND overlay loading through
            # MultimodalAtlas so the order is adapt-then-overlay. This is
            # critical for the dual-stream export: a trained PET 4-channel
            # Conv3d in the overlay checkpoint must hit a freshly-adapted
            # 4-channel layer (shape match -> load), not the HF 11-channel
            # layer (shape mismatch -> dropped, then re-Kaiming'd here,
            # losing the trained weights).
            self.backbone = MultimodalAtlas(
                args=None,
                device=config.device,
                model_repo_id=config.model_repo_id,
                model_revision=config.model_revision,
                pretrained=True,
                input_channels=config.input_channels,
                pretrained_backbone_ckpt=config.pretrained_backbone_ckpt,
            )
            # MultimodalAtlas._maybe_adapt_first_conv3d already swapped the
            # first Conv3d (when needed). Record its name for downstream use.
            first_conv_name = None
            for name, module in self.backbone.visual.named_modules():
                if isinstance(module, nn.Conv3d):
                    first_conv_name = name
                    break
            self.input_adapter_name = first_conv_name
        else:
            # Default path (no overlay): keep the original
            # build-HF-then-external-adapt sequence. Used by
            # sanity_check.py and any single-stream training config that
            # does NOT supply a pretrained_backbone_ckpt.
            self.backbone = MultimodalAtlas(
                args=None,
                device=config.device,
                model_repo_id=config.model_repo_id,
                model_revision=config.model_revision,
                pretrained=True,
            )
            self.input_adapter_name = _adapt_first_conv3d(
                self.backbone.visual,
                target_in_channels=config.input_channels,
                init=config.patch_embed_init,
            )
        _patch_posemb_modules(self.backbone.visual)
        self.hidden_dim = self.backbone.hidden_dim

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B,C,D,H,W) or (C,D,H,W), got {tuple(x.shape)}")

        batch = {"anatomy": [self.config.anatomy] * x.shape[0]}
        return self.backbone(x, batch=batch)


class ViMedChestDualStreamEncoders(nn.Module):
    """Convenience wrapper holding both CT and PET Atlas encoders."""

    def __init__(
        self,
        ct_channels: int = 11,
        pet_channels: int = 4,
        anatomy: str = "chest_ct",
        model_repo_id: str = "YalaLab/Pillar0-ChestCT",
        model_revision: str = "main",
        device: Optional[str] = None,
        patch_embed_init: str = "kaiming",
        ct_pretrained_backbone_ckpt: Optional[str] = None,
        pet_pretrained_backbone_ckpt: Optional[str] = None,
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
            EncoderConfig(
                input_channels=ct_channels,
                pretrained_backbone_ckpt=ct_pretrained_backbone_ckpt,
                **common,
            )
        )
        self.pet_encoder = PillarInitializedAtlasEncoder(
            EncoderConfig(
                input_channels=pet_channels,
                pretrained_backbone_ckpt=pet_pretrained_backbone_ckpt,
                **common,
            )
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


class TokenConcatFusion(nn.Module):
    """Fuse CT and PET encoder activations into a single token sequence.

    Each encoder emits ``activ`` of shape ``(B, token_dim, H, W, D)`` (default
    Pillar0 Atlas-small: ``(B, 1152, 32, 32, 16)``). This module flattens
    both to ``(B, H*W*D, token_dim)``, adds:

    - a **per-modality type embedding** (2 learnable vectors, CT=[0], PET=[1])
    - a **shared 3D positional embedding** -- the same parameter is added to
      both CT and PET tokens at matching grid positions.

    Then concatenates along the token axis to produce ``(B, 2*H*W*D, token_dim)``.
    For the default grid that's ``(B, 32768, 1152)`` -- the contract for the
    downstream Q-Former / LLM decoder.

    Why share the positional embedding across modalities? CT and PET are
    hardware-co-registered (same PET/CT scanner), so a CT voxel at ``(z,y,x)``
    and the PET voxel at the same ``(z,y,x)`` correspond physically. Sharing
    ``pos_embed`` makes that anatomical co-location explicit and lets the
    downstream attention layer (Q-Former cross-attn, or the LLM's own attn)
    bind tokens across modalities by position for free.

    Parameter count is tiny (~58K = grid*1152 + 2*1152). These are initialized
    but **untrained at export time**; they pick up signal in the Phase B
    Q-Former + LLM training run.
    """

    def __init__(
        self,
        token_dim: int = 1152,
        grid: tuple[int, int, int] = (32, 32, 16),
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grid = tuple(grid)
        H, W, D = self.grid
        self.num_tokens_per_modality = H * W * D

        # (2, C): index 0 = CT, index 1 = PET.
        self.modality_type_embed = nn.Parameter(torch.zeros(2, token_dim))
        # (1, N, C) shared across modalities.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens_per_modality, token_dim)
        )
        nn.init.trunc_normal_(self.modality_type_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _flatten_activ(self, activ: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W, D) -> (B, H*W*D, C)`` with shape sanity checks."""
        if activ.ndim != 5:
            raise ValueError(
                f"Expected activ shape (B, C, H, W, D); got {tuple(activ.shape)}"
            )
        B, C, H, W, D = activ.shape
        if (H, W, D) != self.grid:
            raise ValueError(
                f"activ spatial dims {(H, W, D)} != configured grid {self.grid}; "
                f"check that the encoder's H_tok/W_tok/D_tok output matches."
            )
        if C != self.token_dim:
            raise ValueError(
                f"activ channel dim {C} != configured token_dim {self.token_dim}"
            )
        return activ.flatten(start_dim=2).transpose(1, 2).contiguous()

    def forward(self, ct_activ: torch.Tensor, pet_activ: torch.Tensor) -> torch.Tensor:
        ct_tokens = self._flatten_activ(ct_activ)      # (B, N, C)
        pet_tokens = self._flatten_activ(pet_activ)    # (B, N, C)

        ct_tokens = ct_tokens + self.pos_embed + self.modality_type_embed[0]
        pet_tokens = pet_tokens + self.pos_embed + self.modality_type_embed[1]

        # (B, 2N, C). Layout convention: CT first, then PET. The downstream
        # Q-Former / LLM treats the whole sequence uniformly; modality-type
        # embed is what distinguishes them, not their index in the sequence.
        return torch.cat([ct_tokens, pet_tokens], dim=1)


class ViMedChestDualStreamFusedEncoder(nn.Module):
    """Single MultiStage-resolvable backbone: dual-stream encoders + concat fusion.

    Composes :class:`ViMedChestDualStreamEncoders` with
    :class:`TokenConcatFusion` to produce, from paired CT/PET windowed
    inputs, a single ``(B, 32768, 1152)`` token sequence and an averaged
    ``(B, 1152)`` pooled vector.

    Accepts either a positional ``(ct_windows, pet_windows)`` call or a
    dict with keys ``"ct_windows"`` and ``"pet_windows"``, so it slots into
    both ad-hoc scripts and ``MultiStage``-driven training (which passes the
    whole batch dict to the backbone).
    """

    def __init__(
        self,
        ct_channels: int = 11,
        pet_channels: int = 4,
        anatomy: str = "chest_ct",
        model_repo_id: str = "YalaLab/Pillar0-ChestCT",
        model_revision: str = "main",
        device: Optional[str] = None,
        patch_embed_init: str = "kaiming",
        ct_pretrained_backbone_ckpt: Optional[str] = None,
        pet_pretrained_backbone_ckpt: Optional[str] = None,
        token_dim: int = 1152,
        grid: tuple[int, int, int] = (32, 32, 16),
    ) -> None:
        super().__init__()
        self.dual_stream = ViMedChestDualStreamEncoders(
            ct_channels=ct_channels,
            pet_channels=pet_channels,
            anatomy=anatomy,
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            device=device,
            patch_embed_init=patch_embed_init,
            ct_pretrained_backbone_ckpt=ct_pretrained_backbone_ckpt,
            pet_pretrained_backbone_ckpt=pet_pretrained_backbone_ckpt,
        )
        self.fusion = TokenConcatFusion(token_dim=token_dim, grid=grid)
        self.token_dim = token_dim
        self.grid = tuple(grid)
        self.num_tokens = 2 * self.fusion.num_tokens_per_modality
        self.hidden_dim = token_dim  # mirrors MultimodalAtlas.hidden_dim contract

    def forward(self, batch_or_ct, pet_windows: Optional[torch.Tensor] = None) -> dict:
        if isinstance(batch_or_ct, dict):
            ct_windows = batch_or_ct["ct_windows"]
            pet_windows = batch_or_ct["pet_windows"]
        else:
            ct_windows = batch_or_ct
            if pet_windows is None:
                raise ValueError(
                    "ViMedChestDualStreamFusedEncoder.forward needs a dict with "
                    "'ct_windows' / 'pet_windows' OR two positional tensors."
                )
        encs = self.dual_stream(ct_windows, pet_windows)
        fused = self.fusion(encs["ct_activ"], encs["pet_activ"])
        return {
            "activ": fused,                # (B, 32768, 1152)
            "pooled": fused.mean(dim=1),   # (B, 1152)
            # Also expose the per-modality outputs for ablation / debugging.
            "ct_activ": encs["ct_activ"],
            "pet_activ": encs["pet_activ"],
            "ct_pooled": encs["ct_pooled"],
            "pet_pooled": encs["pet_pooled"],
        }
