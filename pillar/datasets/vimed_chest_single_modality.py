from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pillar.utils.memdebug import print_mem
from pillar.utils.petct_windowing import (
    make_ct_windows_fast,
    make_pet_windows_fast,
)


def _resolve_torch_dtype(name: str | torch.dtype | None) -> torch.dtype:
    if name is None or name == "float32" or name is torch.float32:
        return torch.float32
    if name == "float16" or name is torch.float16:
        return torch.float16
    if name == "bfloat16" or name is torch.bfloat16:
        return torch.bfloat16
    raise ValueError(f"Unsupported cpu_dtype={name!r}; expected float32, float16, or bfloat16")


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
        label_columns: list[str] | None = None,
        cpu_dtype: str | torch.dtype | None = "float32",
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
        self.cpu_dtype = _resolve_torch_dtype(cpu_dtype)
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
            x = make_ct_windows_fast(x_raw[0], dtype=self.cpu_dtype)  # (11, D, H, W)
        else:
            x = make_pet_windows_fast(x_raw[1], dtype=self.cpu_dtype)  # (4, D, H, W)

        labels = item.get("labels", None)
        if isinstance(labels, torch.Tensor):
            labels = labels.float()
        elif labels is None and self.label_columns is not None:
            labels = torch.zeros(len(self.label_columns), dtype=torch.float32)

        metadata = item.get("metadata", {})
        report_text = metadata.get("report_text", row.get("report_text", ""))

        # Tiny zero placeholders for engine-required keys we never read.
        # ViMED weak-label training has no per-voxel supervision; mask and
        # image_annotations would otherwise be 4.2 MB (bool) + 16.8 MB (fp32)
        # all-zero passengers in shm + pinned host memory per sample. At
        # batch_size=32, prefetch_factor=2, num_workers=12 that's ~16 GB of
        # in-flight zeros. Engine.preprocess_batch only reads these when
        # use_gpu_augs is True, which is False in our configs; if ever
        # enabled with this dataset, the GPU aug pipeline must lazy-expand
        # these placeholders to (B, 1, D, H, W).
        placeholder_bool = torch.zeros(1, dtype=torch.bool)
        placeholder_float = torch.zeros(1, dtype=torch.float32)

        return {
            "x": x,
            "y": labels,
            "mask": placeholder_bool,
            "image_annotations": placeholder_float,
            "has_annotation": False,
            "accession": row["study_id"],
            "sample_name": row["study_id"],
            "study_id": row["study_id"],
            "region": row["region"],
            "anatomy": self.anatomy,
            "label_names": item.get("label_names", self.label_columns or []),
            "report_text": report_text,
        }
