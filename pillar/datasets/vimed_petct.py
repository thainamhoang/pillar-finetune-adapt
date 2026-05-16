from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch


class ViMedPETCTDataset(torch.utils.data.Dataset):
    """
    Dataset for preprocessed ViMED PET/CT tensors.

    Expected manifest columns:
    - study_id
    - split: train / val / test
    - region
    - tensor_path
    - label columns (optional but recommended)

    Each tensor file is expected to contain:
    - x_raw: tensor shaped (2, D, H, W)
    - labels: tensor shaped (num_labels,)
    - label_names: list[str]
    - metadata: dict
    """

    def __init__(
        self,
        args,
        augmentations,
        csv_path: str,
        split_group: str = "train",
        region: Optional[str] = None,
        anatomy: Optional[str] = "chest_ct",
        windows: Optional[str] = "all",
        label_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        del augmentations, kwargs  # handled elsewhere in the training stack

        self.csv_path = csv_path
        self.region = region
        self.anatomy = anatomy
        self.windows = windows

        split_alias = {"dev": "val"}
        resolved_split = split_alias.get(split_group, split_group)

        df = pd.read_csv(self.csv_path)
        if "split" not in df.columns:
            raise ValueError("Manifest must contain a 'split' column")
        df = df[df["split"] == resolved_split].reset_index(drop=True)
        if region is not None:
            df = df[df["region"] == region].reset_index(drop=True)

        if label_columns is None:
            label_columns = [
                c
                for c in df.columns
                if c
                not in {
                    "study_id",
                    "split",
                    "region",
                    "tensor_path",
                    "ct_path",
                    "pet_path",
                    "report_en_path",
                    "z_start",
                    "z_end",
                    "raw_depth",
                    "height",
                    "weight",
                    "year",
                    "report_text",
                }
            ]

        self.df = df
        self.label_columns = list(label_columns)
        self.info = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[int(idx)]
        item = torch.load(row["tensor_path"], map_location="cpu")

        x = item["x_raw"].float()
        labels = item.get("labels", None)
        label_names = item.get("label_names", self.label_columns)

        if labels is None:
            labels = torch.tensor([float(row[c]) for c in self.label_columns], dtype=torch.float32)
            label_names = self.label_columns
        else:
            labels = labels.float()

        d, h, w = x.shape[1:]
        mask = torch.zeros((1, d, h, w), dtype=torch.bool)

        return {
            "x": x,
            "y": labels,
            "mask": mask,
            "image_annotations": torch.zeros_like(mask, dtype=torch.float32),
            "has_annotation": False,
            "accession": row["study_id"],
            "sample_name": row["study_id"],
            "study_id": row["study_id"],
            "region": row["region"],
            "anatomy": self.anatomy,
            "label_names": label_names,
            "report_text": row.get("report_text", ""),
        }
