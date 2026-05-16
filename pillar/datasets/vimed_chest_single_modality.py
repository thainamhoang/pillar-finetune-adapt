from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from pillar.datasets.vimed_chest_report import ViMedChestReportDataset


class ViMedChestSingleModalityDataset(Dataset):
    """Thin adapter exposing one of {ct_windows, pet_windows} as ``x``.

    Wraps ``ViMedChestReportDataset`` so the existing ``MultiStage`` training
    loop can train a single-modality encoder (CT-only or PET-only) on the
    same preprocessed ViMED cache.
    """

    def __init__(
        self,
        args,
        augmentations,
        csv_path: str | Path,
        split_group: str = "train",
        region: str = "chest",
        channels_mode: str = "ct",
        anatomy: str = "chest_ct",
        label_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        del args, augmentations, kwargs
        if channels_mode not in ("ct", "pet"):
            raise ValueError(f"channels_mode must be 'ct' or 'pet', got {channels_mode!r}")

        split_alias = {"dev": "val"}
        resolved_split = split_alias.get(split_group, split_group)
        self.inner = ViMedChestReportDataset(
            manifest_path=csv_path,
            split=resolved_split,
            region=region,
            include_raw=False,
        )
        self.channels_mode = channels_mode
        self.anatomy = anatomy
        self.label_columns = list(label_columns) if label_columns is not None else None
        self.info: dict = {}

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self.inner[idx]
        key = "ct_windows" if self.channels_mode == "ct" else "pet_windows"
        x = sample[key].float()
        _, d, h, w = x.shape
        mask = torch.zeros((1, d, h, w), dtype=torch.bool)

        labels = sample.get("labels", None)
        label_names = sample.get("label_names", self.label_columns or [])
        if labels is None and self.label_columns is not None:
            labels = torch.zeros(len(self.label_columns), dtype=torch.float32)

        return {
            "x": x,
            "y": labels.float() if isinstance(labels, torch.Tensor) else labels,
            "mask": mask,
            "image_annotations": torch.zeros_like(mask, dtype=torch.float32),
            "has_annotation": False,
            "accession": sample["study_id"],
            "sample_name": sample["study_id"],
            "study_id": sample["study_id"],
            "region": sample["region"],
            "anatomy": self.anatomy,
            "label_names": label_names,
            "report_text": sample.get("report_text", ""),
        }
