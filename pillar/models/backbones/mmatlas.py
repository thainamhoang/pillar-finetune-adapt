"""Multimodal Atlas implementation."""

import torch
import numpy as np
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from torchvision.transforms import Compose, Normalize
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Any
from easydict import EasyDict
from pathlib import Path
import yaml
from pillar.utils.logging import logger


def ensure_transformers_tied_weight_compat() -> None:
    """
    Bridge minor API drift in Transformers tied-weight loading.

    Some custom remote-code models (e.g. the Pillar0-ChestCT remote code)
    still expose ``_tied_weights_keys`` in the older format, while newer
    Transformers loader paths look for ``all_tied_weights_keys``. The
    shim materializes ``all_tied_weights_keys`` on ``PreTrainedModel`` so
    old custom code becomes loadable.

    Compatibility wrinkle: in transformers >= ~4.56, ``PreTrainedModel``
    itself doesn't expose ``all_tied_weights_keys`` on the *class* but
    its ``post_init`` *writes* to ``self.all_tied_weights_keys`` (as an
    instance attribute, e.g. when constructing ``SiglipVisionModel``).
    A read-only ``@property`` shim makes that write fail with
    ``AttributeError: property of '...' object has no setter``.

    To support both API generations simultaneously, install the shim as
    a property *with* a setter that just stashes onto a private instance
    slot. Subsequent reads from old-style code path through the slot if
    set, else fall back to ``_tied_weights_keys``.
    """
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    _SLOT = "_pillar_all_tied_weights_keys"

    def _get(self):
        # Honor anything transformers' post_init wrote to us.
        cached = getattr(self, _SLOT, None)
        if cached is not None:
            return cached
        tied = getattr(self, "_tied_weights_keys", {}) or {}
        if isinstance(tied, dict):
            return tied
        if isinstance(tied, (list, tuple, set)):
            return {key: key for key in tied}
        return {}

    def _set(self, value):
        # Use object.__setattr__ to avoid recursing into nn.Module.__setattr__'s
        # parameter / buffer / submodule routing (the value is a plain dict,
        # not a torch object).
        object.__setattr__(self, _SLOT, value)

    PreTrainedModel.all_tied_weights_keys = property(_get, _set)


def setup_device(device_spec: str | None = None) -> torch.device:
    """
    Set up and validate device for model/data operations.

    Args:
        device_spec: Device specification ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        Torch device object
    """
    if device_spec is None:
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device


def load_model_config(model_name: str) -> dict[str, Any]:
    """
    Load model-specific configuration.

    Args:
        model_name: Name of the model (e.g., 'medgemma', 'medimageinsights')

    Returns:
        Model configuration dictionary
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "models" / f"{model_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    args = EasyDict(config)
    return args


def get_config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get value from config using dot notation (e.g., 'data.root_dir').

    Args:
        config: Configuration dictionary
        key: Dot-separated key path
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


class MultimodalAtlas(nn.Module):
    """Multimodal Atlas medical image analysis."""

    def __init__(
        self,
        args,
        device="cuda",
        model_repo_id=None,
        model_revision=None,
        pretrained=False,
        input_channels=None,
        extra_channel_init="zero",
        pretrained_backbone_ckpt: str | None = None,
    ):
        super().__init__()
        self.args = args

        # Load model-specific config and setup device
        self.device = setup_device(device)
        self.model_repo_id = model_repo_id
        self.model_revision = model_revision
        self.pretrained = pretrained
        self.input_channels = input_channels
        self.extra_channel_init = extra_channel_init
        # Optional second-stage init: overlay a previously-trained encoder
        # state_dict (e.g. W_B from Phase 1 CT) ON TOP of the HF-pretrained
        # weights, AFTER the first Conv3d has been adapted for the new
        # input_channels. Loaded with strict=False so the resized Conv3d
        # and any non-encoder keys (head, pool) are skipped.
        self.pretrained_backbone_ckpt = pretrained_backbone_ckpt

        # Setup model
        self.setup_model()

        logger.info(f"Initialized MultimodalAtlas on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""

        print(f"Loading {self.model_repo_id} with revision {self.model_revision} from HuggingFace")
        logger.info("Loading model from HuggingFace")
        ensure_transformers_tied_weight_compat()

        # self.model is CLIPMultimodalAtlas
        self.model = AutoModel.from_pretrained(
            self.model_repo_id,
            revision=self.model_revision,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        # self.visual is MultiModalAtlas
        if hasattr(self.model.model, "visual"):
            self.visual = self.model.model.visual
        else:
            self.visual = self.model.model

        self.model.to(self.device)

        if self.input_channels is not None:
            replaced = self._maybe_adapt_first_conv3d(self.input_channels)
            if replaced is not None:
                logger.info(f"Adapted first Conv3d patch embed for {self.input_channels} input channels at {replaced}")
            else:
                logger.warning(f"Requested input_channels={self.input_channels}, but no matching Conv3d was found")

        # Overlay a previously-trained encoder state_dict, e.g. from Phase 1
        # CT. Done AFTER the first-Conv3d adaptation so the resized layer
        # keeps its Kaiming init (channel semantics differ across modalities).
        # strict=False because head/pool keys live under a different prefix
        # in the upstream checkpoint and the resized Conv3d shape won't
        # match the saved one.
        if self.pretrained_backbone_ckpt:
            self._load_pretrained_backbone_ckpt(self.pretrained_backbone_ckpt)

        # Get hidden_dim from the model - check various possible attributes
        if hasattr(self.visual, "embed_dim"):
            self.hidden_dim = self.visual.embed_dim
        else:
            # Default fallback
            self.hidden_dim = 1152
            logger.warning(f"Could not determine hidden_dim from model, using default: {self.hidden_dim}")

        self.model.train()

        logger.info(f"Model loaded successfully on device: {self.device}")

    def _maybe_adapt_first_conv3d(self, target_in_channels: int) -> str | None:
        """Replace the first Conv3d so it accepts ``target_in_channels`` inputs.

        Handles both expansion (pretrained 11 -> 12 PET/CT, etc.) and
        reduction (pretrained 11 -> 11 CT windows or -> 4 PET windows).
        On expansion the pretrained weights are preserved and the new
        channels are initialized per ``self.extra_channel_init``. On
        reduction the channel semantics no longer match the pretrained
        layout, so the layer is reinitialized with Kaiming.
        """
        candidate_name = None
        candidate_module = None
        for name, module in self.visual.named_modules():
            if isinstance(module, nn.Conv3d):
                candidate_name = name
                candidate_module = module
                break

        if candidate_module is None:
            return None
        if candidate_module.in_channels == target_in_channels:
            return candidate_name

        new_conv = nn.Conv3d(
            in_channels=target_in_channels,
            out_channels=candidate_module.out_channels,
            kernel_size=candidate_module.kernel_size,
            stride=candidate_module.stride,
            padding=candidate_module.padding,
            dilation=candidate_module.dilation,
            groups=candidate_module.groups,
            bias=candidate_module.bias is not None,
            padding_mode=candidate_module.padding_mode,
            device=candidate_module.weight.device,
            dtype=candidate_module.weight.dtype,
        )

        with torch.no_grad():
            old_in = candidate_module.in_channels
            if target_in_channels > old_in:
                # Expansion: preserve pretrained channels, init the rest.
                new_conv.weight.zero_()
                new_conv.weight[:, :old_in].copy_(candidate_module.weight)
                if self.extra_channel_init == "mean":
                    mean_weight = candidate_module.weight.mean(dim=1, keepdim=True)
                    repeat = target_in_channels - old_in
                    new_conv.weight[:, old_in:target_in_channels].copy_(
                        mean_weight.repeat(1, repeat, 1, 1, 1)
                    )
            else:
                # Reduction: channel semantics differ from pretraining.
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if candidate_module.bias is not None:
                new_conv.bias.copy_(candidate_module.bias)

        parent = self.visual
        parts = candidate_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)
        return candidate_name

    def _load_pretrained_backbone_ckpt(self, ckpt_path: str) -> None:
        """Overlay a previously-trained encoder state_dict onto this backbone.

        Accepts either of two checkpoint shapes:

        - **encoder-only** dict where keys are the bare MultimodalAtlas
          state_dict keys (produced by ``scripts/extract_encoder.py``).
        - **full training checkpoint** dict with a ``"state_dict"`` or
          ``"model"`` entry plus optimizer / scheduler; we strip the
          ``backbone_model.`` prefix that ``MultiStage`` adds.

        Uses ``strict=False`` so non-encoder keys are skipped gracefully,
        AND pre-filters any keys whose tensor shape disagrees with the
        current model (e.g. the freshly Kaiming-init first Conv3d when
        adapting from 11 -> 4 channels). ``strict=False`` alone is NOT
        enough -- PyTorch still raises on shape mismatches for keys that
        exist in both source and destination.
        """
        import os

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"pretrained_backbone_ckpt={ckpt_path!r} does not exist"
            )
        logger.info(f"Loading pretrained backbone overlay from {ckpt_path}")
        raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Unwrap nested formats
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
        else:
            sd = raw

        # Strip leading "backbone_model." (from MultiStage) so keys map to
        # MultimodalAtlas.state_dict() directly.
        prefix = "backbone_model."
        sd = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in sd.items()
        }

        # Drop keys whose shape doesn't match the current model. This is
        # how the resized first Conv3d (CT 11ch -> PET 4ch reduction) is
        # handled: the new Kaiming-init layer keeps its weights, and the
        # saved [64, 11, 3, 3, 3] tensor is skipped.
        current = self.state_dict()
        shape_mismatch = []
        kept = {}
        for k, v in sd.items():
            if k in current and isinstance(v, torch.Tensor) and v.shape != current[k].shape:
                shape_mismatch.append((k, tuple(v.shape), tuple(current[k].shape)))
            else:
                kept[k] = v
        if shape_mismatch:
            logger.info(
                f"Dropping {len(shape_mismatch)} key(s) due to shape mismatch "
                f"(expected when adapting input channels). First few: "
                f"{[(k, src, dst) for k, src, dst in shape_mismatch[:3]]}"
            )

        missing, unexpected = self.load_state_dict(kept, strict=False)
        logger.info(
            f"Loaded pretrained backbone overlay: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(shape_mismatch)} shape-mismatched (skipped). "
            f"First few missing: {missing[:5]}  First few unexpected: {unexpected[:5]}"
        )

    def preprocess_single(self, image):
        """
        Preprocess a single exam for your model (for use in dataset __getitem__).

        Argscn:
            image: Tensor from dataset (normalized to [0,1] range)

        Returns:
            Preprocessed tensor ready for your model
        """
        ## hard
        # normalize_mean = self.normalize_mean
        # normalize_std = self.normalize_std
        # self.transform = Compose(
        #     [
        #         Normalize(
        #             mean=normalize_mean, std=normalize_std
        #         ),  # Normalize for single channel
        #     ]
        # )

        # image = self.transform(image)
        return image

    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input images/volumes.

        Args:
            inputs: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Feature embeddings as numpy array of shape (B, feature_dim)
        """
        inputs = inputs.to(self.device)
        inputs_as_dict = {modality: inputs}

        with torch.no_grad():
            # Check if we have the HuggingFace model with extract_vision_feats method
            if hasattr(self.model, "extract_vision_feats"):
                features = self.model.extract_vision_feats(inputs)
            else:
                # Fallback to forward pass for feature extraction
                output = self.forward(inputs)
                if isinstance(output, dict) and "features" in output:
                    features = output["features"]
                elif isinstance(output, torch.Tensor):
                    features = output
                else:
                    raise ValueError(f"Cannot extract features from model output of type {type(output)}")

            # Convert to numpy for multiprocessing compatibility
            features = features.cpu().numpy().astype(np.float32)

        return features

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor, batch=None) -> dict:
        visual = self.visual

        modality, image = batch["anatomy"][0], x
        bsz = image.shape[0]
        logger.debug(f"Entering MultiModalAtlas forward with modality: {modality}, batch size: {bsz}")

        # Atlas stores image_size / patch_size in [H, W, D] order. PyTorch
        # Conv3d patch_embed expects (B, C, D, H, W) and applies kernel[2]
        # to the LAST axis. To make these consistent for non-cubic inputs,
        # permute pillar's (B, C, D, H, W) tensor convention to (B, C, H, W, D)
        # and update the dynamic image_size to match. For cubic inputs this is
        # a no-op.
        if x.ndim == 5:
            d, h, w = x.shape[-3:]
            visual.model_config["modalities"][modality]["image_size"] = [h, w, d]
            x = x.permute(0, 1, 3, 4, 2).contiguous()

        x, grid_sizes = visual.build_scales({modality: x}, batch)

        modal_config = visual.model_config["modalities"][modality]
        image_size = modal_config["image_size"]
        merge_ratio = modal_config["merge_ratio"]
        local2global = modal_config["local2global"]
        patch_size = modal_config["patch_size"]
        multiscale_layout = visual.prepare_multiscale_layout(
            img_size=image_size,
            merge_ratio=merge_ratio,
            local2global=local2global,
            patch_size=patch_size,
        )
        grid_sizes = [layout["grid_size"] for layout in multiscale_layout]

        multiscale_feats = []
        for level, atlas_model in enumerate(visual.atlas_models):
            is_last = len(x) == 1 or level == len(visual.atlas_models) - 1
            x = atlas_model(
                x,
                grid_sizes=grid_sizes[level:],
                multiscale_layout=multiscale_layout[level:],
                merge_ratio=merge_ratio,
                local2global=local2global,
                modality=modality,
            )
            if not is_last:
                multiscale_feats.append(x[0])
                x = x[1:]

        multiscale_feats.append(x[0])

        feats_for_head = []
        for i, scale_tokens in enumerate(multiscale_feats):
            # Need to know if scale_tokens is windowed (B*NW, K, C) or flattened (B, N, C)
            # Assuming it's (B*NW, K, C) if NW > 1 based on layout
            if math.prod(grid_sizes[i]) > 1 and len(scale_tokens.shape) == 3:  # Check if likely windowed format
                rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                logger.debug(f"  Rearranging scale {i} (windowed) {scale_tokens.shape} -> {rearranged_scale.shape}")
                feats_for_head.append(rearranged_scale)
            elif len(scale_tokens.shape) == 3 and scale_tokens.shape[0] == bsz:  # Already (B, N, C)
                logger.debug(f"  Using scale {i} (already BNC) shape: {scale_tokens.shape}")
                feats_for_head.append(scale_tokens)
            else:
                logger.warning(f"  Unexpected shape for scale {i}: {scale_tokens.shape}. Attempting BNC rearrange.")
                # Try a generic rearrange, might fail if batch dim isn't divisible
                try:
                    rearranged_scale = rearrange(scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz)
                    feats_for_head.append(rearranged_scale)
                except Exception as e:
                    logger.error(f"    Failed to rearrange scale {i}: {e}. Skipping this scale for readout.")
                    continue

        if visual.multiscale_feats:
            x = []
            for i, scale in enumerate(feats_for_head):
                feats = visual.maxpool(scale.transpose(1, 2)).squeeze(2)
                x.append(feats)
            x = torch.cat(x, dim=1)
        else:
            ## return maxpool on last scale features only
            x = feats_for_head[0]
            x = visual.maxpool(x.transpose(1, 2)).squeeze(2)

        import torch.nn.functional as F

        res = [image_size[i] // patch_size[i] for i in range(3)]
        outs = []

        for idx in range(len(multiscale_layout)):
            layout = multiscale_layout[idx]
            scale = feats_for_head[idx]
            m0, m1, m2 = layout["window_dims"]
            g0, g1, g2 = layout["grid_size"]
            scale = rearrange(
                scale,
                "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)",
                g2=g2,
                g1=g1,
                g0=g0,
                m2=m2,
                m1=m1,
                m0=m0,
            )
            # interpolate to the same size
            scale = nn.functional.interpolate(scale, size=res, mode="nearest")
            outs.append(scale)

        outs = torch.cat(outs, dim=1)

        outputs = {
            "activ": outs,
            # could also consider taking scale 2 and scale 3 and interpolate up then concat along channels
            # can modify atlas for the cleanest API
            "pooled": F.normalize(x, dim=-1),
        }
        return outputs
