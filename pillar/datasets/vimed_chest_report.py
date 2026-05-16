from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from pillar.utils.petct_windowing import make_dual_stream_window_inputs


class ViMedChestReportDataset(Dataset):
    """Chest-only ViMED dataset for dual-stream PET/CT report generation.

    This dataset intentionally reuses the existing preprocessing cache
    (`x_raw`) and derives CT/PET windows on the fly.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        region: str = "chest",
        include_raw: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = list(csv.DictReader(self.manifest_path.open()))
        if split is not None:
            self.rows = [r for r in self.rows if r.get("split") == split]
        self.rows = [r for r in self.rows if r.get("region") == region]
        self.include_raw = include_raw

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        item = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)
        x_raw = item["x_raw"].float()
        windows = make_dual_stream_window_inputs(x_raw)
        metadata = item.get("metadata", {})
        report_text = metadata.get("report_text", row.get("report_text", ""))

        out = {
            "ct_windows": windows["ct_windows"],
            "pet_windows": windows["pet_windows"],
            "report_text": report_text,
            "study_id": row["study_id"],
            "accession": row["study_id"],
            "region": row["region"],
            "metadata": metadata,
        }
        if "labels" in item:
            out["labels"] = item["labels"].float()
            out["label_names"] = list(item.get("label_names", []))
        if self.include_raw:
            out["x_raw"] = x_raw
        return out
