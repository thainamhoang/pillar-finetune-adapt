from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torchmetrics.functional import auroc, average_precision

from pillar.metrics.abstract import AbstractMetric


class MultilabelMetric(AbstractMetric):
    def __init__(
        self,
        args,
        dataset_info=None,
        split=None,
        logit_key: str = "classification",
        target_key: str = "y",
        threshold: float = 0.5,
        label_names: Iterable[str] | None = None,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        self.dataset_info = dataset_info
        self.split = split
        self.logit_key = logit_key
        self.target_key = target_key
        self.threshold = threshold
        self.label_names = list(label_names) if label_names is not None else None

    @property
    def metric_keys(self):
        return {"target_label": self.target_key, "pred_label": self.logit_key}

    def __call__(self, **kwargs):
        logits = kwargs[self.logit_key]
        target = kwargs[self.target_key]

        if isinstance(logits, list):
            logits = torch.stack(logits, dim=0)
        if isinstance(target, list):
            target = torch.stack(target, dim=0)

        logits = logits.float()
        target = target.float()
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        num_labels = target.shape[1]
        label_names = self.label_names or [f"label_{i}" for i in range(num_labels)]
        if len(label_names) != num_labels:
            raise ValueError(f"Expected {num_labels} label names, got {len(label_names)}")

        macro_auroc = []
        macro_ap = []
        macro_f1 = []
        macro_precision = []
        macro_recall = []
        stats = OrderedDict()

        for idx, name in enumerate(label_names):
            y = target[:, idx].int()
            p = probs[:, idx]
            yhat = preds[:, idx]

            positives = int(y.sum().item())
            negatives = int((y == 0).sum().item())

            if positives > 0 and negatives > 0:
                label_auroc = auroc(p, y, task="binary")
                label_ap = average_precision(p, y, task="binary")
                macro_auroc.append(label_auroc)
                macro_ap.append(label_ap)
                stats[f"{name}_auroc"] = label_auroc
                stats[f"{name}_ap"] = label_ap
            else:
                stats[f"{name}_auroc"] = torch.tensor(-1.0, device=target.device)
                stats[f"{name}_ap"] = torch.tensor(-1.0, device=target.device)

            tp = ((yhat == 1) & (y == 1)).sum().float()
            fp = ((yhat == 1) & (y == 0)).sum().float()
            fn = ((yhat == 0) & (y == 1)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            stats[f"{name}_precision"] = precision
            stats[f"{name}_recall"] = recall
            stats[f"{name}_f1"] = f1

            macro_precision.append(precision)
            macro_recall.append(recall)
            macro_f1.append(f1)

        stats["macro_auroc"] = (
            torch.stack(macro_auroc).mean() if macro_auroc else torch.tensor(-1.0, device=target.device)
        )
        stats["macro_ap"] = torch.stack(macro_ap).mean() if macro_ap else torch.tensor(-1.0, device=target.device)
        stats["macro_precision"] = torch.stack(macro_precision).mean()
        stats["macro_recall"] = torch.stack(macro_recall).mean()
        stats["macro_f1"] = torch.stack(macro_f1).mean()
        return stats
