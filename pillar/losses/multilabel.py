from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from pillar.losses.abstract import AbstractLoss


class MultilabelBCELoss(AbstractLoss):
    def __init__(self, args, logit_key: str = "classification", target_key: str = "y", pos_weight=None, **kwargs):
        super().__init__(args, **kwargs)
        self.logit_key = logit_key
        self.target_key = target_key
        self.pos_weight = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32)

    def __call__(self, **kwargs):
        batch = kwargs["batch"]
        model_output = kwargs["model_output"]

        logits = model_output[self.logit_key]
        target = batch[self.target_key].to(dtype=logits.dtype)
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        logging_dict = OrderedDict()
        logging_dict[f"loss_{self.logit_key}"] = loss.detach()
        return loss, logging_dict

    @property
    def loss_keys(self):
        return {"target_label": self.target_key, "pred_label": self.logit_key}
