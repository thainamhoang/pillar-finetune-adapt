from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from pillar.utils.memdebug import print_mem
from pillar.utils.petct_windowing import (
    make_ct_windows_fast,
    make_pet_windows_fast,
)


class ViMedChestSingleModalityDataset(Dataset):
    """Single-modality (CT or PET) view of the ViMED chest preprocessed cache.

    Reads the manifest CSV, loads each ``.pt`` file lazily, and computes
    windowing for *only* the requested modality. This avoids the ~40%
    wasted CPU per sample incurred by proxying through
    ``ViMedChestReportDataset``, which always computes both CT and PET
    windows even when only one is consumed downstream.
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

        manifest_path = Path(csv_path)
        with manifest_path.open() as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r.get("split") == resolved_split]
        rows = [r for r in rows if r.get("region") == region]
        self.rows = rows

        self.channels_mode = channels_mode
        self.anatomy = anatomy
        self.label_columns = list(label_columns) if label_columns is not None else None
        self.info: dict = {}
        # Per-worker counter for the memdebug probe in __getitem__.
        # Each forked worker inherits 0 and counts independently.
        self._mem_probe_count = 0
        # Period (in samples) between memory probes; 0 disables.
        self._mem_probe_every = int(os.environ.get("PILLAR_MEM_PROBE_EVERY", "100"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        # Memory probe: prints on the first call this worker handles, and
        # every PILLAR_MEM_PROBE_EVERY samples thereafter. Lets us see
        # per-worker RAM growth without spamming the log.
        if self._mem_probe_every > 0 and (
            self._mem_probe_count == 0
            or self._mem_probe_count % self._mem_probe_every == 0
        ):
            print_mem(
                f"dataset[{self.channels_mode}] worker pid={os.getpid()} "
                f"count={self._mem_probe_count} idx={idx}"
            )
        self._mem_probe_count += 1

        row = self.rows[idx]
        item = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)

        # x_raw is (2, D, H, W) with channel 0 = CT, channel 1 = PET. We
        # slice the modality-specific channel before windowing so only one
        # 33 MB float volume crosses to ct/pet helpers; the unused half
        # never gets touched.
        x_raw = item["x_raw"]
        if self.channels_mode == "ct":
            x = make_ct_windows_fast(x_raw[0])  # (11, D, H, W)
        else:
            x = make_pet_windows_fast(x_raw[1])  # (4, D, H, W)

        _, d, h, w = x.shape
        mask = torch.zeros((1, d, h, w), dtype=torch.bool)

        labels = item.get("labels", None)
        if isinstance(labels, torch.Tensor):
            labels = labels.float()
        elif labels is None and self.label_columns is not None:
            labels = torch.zeros(len(self.label_columns), dtype=torch.float32)

        metadata = item.get("metadata", {})
        report_text = metadata.get("report_text", row.get("report_text", ""))

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
            "label_names": item.get("label_names", self.label_columns or []),
            "report_text": report_text,
        }
