from collections import OrderedDict
import importlib
import warnings
from torch import nn

from pillar import models
from pillar.models.abstract_model import AbstractModel
import pillar.models.pooling as pooling
import pillar.models.heads as heads


def resolve_model_class(name):
    if name in models.__dict__:
        return models.__dict__[name]
    if name in heads.__dict__:
        return heads.__dict__[name]
    if name in pooling.__dict__:
        return pooling.__dict__[name]
    direct_imports = (
        "pillar.models.heads.basic",
        "pillar.models.heads.cumulative_hazard_layer",
        "pillar.models.heads.detr",
        "pillar.models.pooling",
        "pillar.models.backbones",
    )
    for module_name in direct_imports:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise KeyError(name)


class BaseMultiStage(AbstractModel):
    """
    Base class for MultiStage models.
    """

    def __init__(
        self,
        args,
        backbone_model_type,
        backbone_kwargs,
        head_models,  # Dict of {head_name: {"type": str, "kwargs": dict, "apply_pooling": bool, "use_pooled_features": bool}}
        pool_name="GlobalMaxPool",
        pool_kwargs={},
        dropout=0.0,
    ):
        super().__init__(args)

        # Initialize backbone
        self.backbone_model = resolve_model_class(backbone_model_type)(args=args, **backbone_kwargs)

        # Initialize pooling
        if "hidden_dim" not in pool_kwargs:
            self.pool = resolve_model_class(pool_name)(args=args, hidden_dim=self.backbone_model.hidden_dim, **pool_kwargs)
        else:
            self.pool = resolve_model_class(pool_name)(args=args, **pool_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize multiple heads
        self.head_models = nn.ModuleDict()
        self.apply_pooling = {}
        self.use_pooled_features = {}
        self.enable_at_eval = {}
        for head_name, head_config in head_models.items():
            if "input_dim" not in head_config["kwargs"]:
                head_config["kwargs"]["input_dim"] = self.backbone_model.hidden_dim
            self.head_models[head_name] = resolve_model_class(head_config["type"])(args=args, **head_config["kwargs"])
            self.apply_pooling[head_name] = head_config.get("apply_pooling", True)
            self.use_pooled_features[head_name] = head_config.get("use_pooled_features", False)
            self.enable_at_eval[head_name] = head_config.get("enable_at_eval", True)

    def _process_heads(self, features, pooled_features, split="train"):
        """
        Process heads based on their feature requirements.
        """
        # Group heads by input type for parallel computation
        use_pooled_features_heads = {}
        apply_pooling_heads = {}
        raw_feature_heads = {}
        for head_name, head_model in self.head_models.items():
            if split in ["dev", "test"] and not self.enable_at_eval[head_name]:
                continue
            if self.use_pooled_features[head_name]:
                use_pooled_features_heads[head_name] = head_model
            elif self.apply_pooling[head_name]:
                apply_pooling_heads[head_name] = head_model
            else:
                raw_feature_heads[head_name] = head_model

        # Process heads in parallel
        head_outputs = OrderedDict()

        # Process pooled feature heads
        if use_pooled_features_heads:
            results = {name: model.forward(pooled_features) for name, model in use_pooled_features_heads.items()}
            head_outputs.update(results)

        # Process apply pooling heads
        pooled_outputs = OrderedDict(self.pool(features))
        if apply_pooling_heads:
            dropout_outputs = self.dropout(pooled_outputs["hidden"])
            results = {name: model.forward(dropout_outputs) for name, model in apply_pooling_heads.items()}
            head_outputs.update(results)

        # Process raw feature heads
        if raw_feature_heads:
            results = {name: model.forward(features) for name, model in raw_feature_heads.items()}
            head_outputs.update(results)

        return head_outputs, pooled_outputs


class MultiStage(BaseMultiStage):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def forward(self, x, batch=None, split="train", **extras):
        # Pass through backbone and get features
        backbone_outputs = OrderedDict(self.backbone_model.forward(x, batch=batch))
        features = backbone_outputs["activ"]  # shape is: [batch_size, backbone_hidden_dim, D, H, W]
        pooled_features = backbone_outputs.get("pooled", None)

        # Process heads
        head_outputs, pooled_outputs = self._process_heads(features, pooled_features, split=split)

        # Merge outputs
        merged_outputs = OrderedDict()
        merged_outputs.update(backbone_outputs)
        merged_outputs.update(pooled_outputs)
        merged_outputs.update(head_outputs)

        backbone_outputs.clear()
        pooled_outputs.clear()
        head_outputs.clear()

        return merged_outputs

    def no_weight_decay(self):
        return []

    def stages(self):
        return self.backbone_model.visual.model_config["model"]["stages"]
