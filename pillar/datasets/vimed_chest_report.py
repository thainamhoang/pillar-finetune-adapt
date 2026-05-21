from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from pillar.utils.petct_windowing import make_dual_stream_window_inputs


# Default English prompt template (chest scope). Trained reports are
# Gemma-4-translated from Vietnamese -> English; we generate in English
# and document this in the methods section. The {ct_pads} / {pet_pads}
# slots are substituted at __init__ time with N copies of <image_ct_pad>
# and <image_pet_pad> respectively, where N matches the per-modality
# perceiver-query count of the model.
DEFAULT_CHEST_PROMPT_TEMPLATE = (
    "You are a nuclear medicine physician specializing in chest PET/CT.\n"
    "The following are paired chest PET/CT images.\n"
    "<ct>{ct_pads}</ct>\n"
    "<pet>{pet_pads}</pet>\n"
    'Generate the "Findings" section of the radiology report in English.\n'
)


class ViMedChestReportDataset(Dataset):
    """Chest-only ViMED dataset for dual-stream PET/CT report generation.

    This dataset reuses the existing preprocessing cache (``x_raw``) and
    derives CT/PET windows on the fly via
    :func:`pillar.utils.petct_windowing.make_dual_stream_window_inputs`.

    When ``tokenizer`` (or ``llm_repo_id``) is provided, the dataset also
    emits the tokenized prompt + report tensors needed by
    :class:`pillar.models.report_generators.DualStreamReportGenerator`:

    - ``prompt_token_ids``         (L_prompt,)
    - ``prompt_attention_mask``    (L_prompt,)
    - ``report_token_ids``         (L_report,) -- terminates with
      ``<|end_of_report|>``; right-padded to ``max_report_tokens``
    - ``report_attention_mask``    (L_report,)

    The placeholder tokens ``<image_ct_pad>`` (× ``num_ct_queries``) and
    ``<image_pet_pad>`` (× ``num_pet_queries``) live inside the
    ``<ct>...</ct>`` and ``<pet>...</pet>`` brackets respectively. The
    model replaces those positions with visual embeddings at forward time.

    Parameters
    ----------
    manifest_path:
        CSV with columns including ``tensor_path``, ``study_id``, ``split``,
        ``region``, optionally ``report_text``.
    split:
        e.g. ``"train"``, ``"val"``, ``"test"``. ``None`` keeps all rows.
    region:
        e.g. ``"chest"``. Only matching rows are kept.
    include_raw:
        Whether to return the raw ``(2, D, H, W)`` PET/CT tensor (used by
        debugging tools; trainer doesn't need it).
    tokenizer:
        Optional HF ``PreTrainedTokenizer`` instance with the special
        tokens from
        :data:`pillar.models.heads.report_lm.SPECIAL_TOKENS` already added.
        If you pass ``llm_repo_id`` instead, the dataset will load a fresh
        tokenizer from HF.
    llm_repo_id:
        HF repo ID of the LLM; used to build a tokenizer if ``tokenizer``
        is not provided. The dataset adds the special tokens to ensure the
        prompt template tokenizes correctly even at first epoch.
    num_ct_queries / num_pet_queries:
        Number of placeholder tokens to insert per modality. Must match
        the model's perceiver-resampler ``num_queries`` (128 default).
    prompt_template:
        Python format string with ``{ct_pads}`` and ``{pet_pads}`` slots.
        Default is :data:`DEFAULT_CHEST_PROMPT_TEMPLATE`.
    max_report_tokens:
        Truncate (and pad) the report to this length. 512 is generous for
        chest reports; the paper used 1024 for whole-body.
    require_report:
        Drop manifest rows whose ``report_text`` is empty / whitespace.
        Defaults ``True`` whenever the tokenizer is set (training mode);
        ``False`` otherwise so the encoder-only Phase A path keeps working.
    """

    def __init__(
        self,
        args=None,
        augmentations=None,
        # CSV path: ``csv_path`` is the convention used by train.py /
        # shared_dataset_kwargs; ``manifest_path`` is the older name still
        # used by Phase A scripts. Accept either.
        csv_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
        # Split: ``split_group`` is train.py convention; ``split`` is the
        # older keyword some scripts still use.
        split_group: Optional[str] = None,
        split: Optional[str] = None,
        region: str = "chest",
        include_raw: bool = False,
        # ---- Phase B tokenization (optional) ----
        tokenizer: Optional[Any] = None,
        llm_repo_id: Optional[str] = None,
        num_ct_queries: int = 128,
        num_pet_queries: int = 128,
        prompt_template: Optional[str] = None,
        max_report_tokens: int = 512,
        require_report: Optional[bool] = None,
        **kwargs,  # silently accept other YAML knobs we don't use
    ) -> None:
        del args, augmentations, kwargs  # unused by this dataset

        # Reconcile csv_path / manifest_path aliases.
        if csv_path is None and manifest_path is None:
            raise ValueError("Must pass csv_path (preferred) or manifest_path")
        path = csv_path if csv_path is not None else manifest_path
        self.manifest_path = Path(path)

        # Reconcile split / split_group aliases, with "dev" -> "val" mapping.
        resolved_split = split_group if split_group is not None else split
        split_alias = {"dev": "val"}
        if resolved_split is not None:
            resolved_split = split_alias.get(resolved_split, resolved_split)

        self.rows = list(csv.DictReader(self.manifest_path.open()))
        if resolved_split is not None:
            self.rows = [r for r in self.rows if r.get("split") == resolved_split]
        self.rows = [r for r in self.rows if r.get("region") == region]
        self.include_raw = include_raw
        # ``info`` matches the convention used by other ViMED datasets so
        # train.py's `dataset_info` plumbing still works.
        self.info: dict = {}

        # ---- tokenizer setup (Phase B only) ----
        self.num_ct_queries = num_ct_queries
        self.num_pet_queries = num_pet_queries
        self.max_report_tokens = max_report_tokens

        if tokenizer is None and llm_repo_id is not None:
            tokenizer = self._build_tokenizer(llm_repo_id)
        self.tokenizer = tokenizer

        # Default require_report: True if tokenizing (training), else False.
        if require_report is None:
            require_report = tokenizer is not None
        self.require_report = require_report

        if tokenizer is not None:
            template = prompt_template if prompt_template is not None else DEFAULT_CHEST_PROMPT_TEMPLATE
            self.prompt_template = template
            self._prebuild_prompt_tokens()
            # We can't statically filter rows by report presence because
            # report_text may live in the .pt sidecar metadata rather than
            # the CSV. Lazy skip at __getitem__ time -- the project's
            # ignore_None_collate (pillar/utils/loading.py) already drops
            # None samples from each batch.
        else:
            self.prompt_template = None
            self._prompt_token_ids = None
            self._prompt_attention_mask = None

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tokenizer(llm_repo_id: str):
        from transformers import AutoTokenizer
        from pillar.models.heads.report_lm import SPECIAL_TOKENS

        tok = AutoTokenizer.from_pretrained(
            llm_repo_id, trust_remote_code=True, use_fast=True
        )
        tok.add_special_tokens(
            {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok

    def _prebuild_prompt_tokens(self) -> None:
        """Tokenize the prompt template ONCE (it's identical per sample).

        Verifies that the resulting token sequence contains exactly
        ``num_ct_queries`` ``<image_ct_pad>`` ids and ``num_pet_queries``
        ``<image_pet_pad>`` ids, so the model's splice step has the
        positions it expects.
        """
        from pillar.models.heads.report_lm import SPECIAL_TOKENS

        ct_pad_tok = SPECIAL_TOKENS["ct_pad"]
        pet_pad_tok = SPECIAL_TOKENS["pet_pad"]
        rendered = self.prompt_template.format(
            ct_pads=ct_pad_tok * self.num_ct_queries,
            pet_pads=pet_pad_tok * self.num_pet_queries,
        )

        enc = self.tokenizer(
            rendered,
            add_special_tokens=False,  # we control BOS/EOS placement
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]

        ct_pad_id = self.tokenizer.convert_tokens_to_ids(ct_pad_tok)
        pet_pad_id = self.tokenizer.convert_tokens_to_ids(pet_pad_tok)
        n_ct = int((ids == ct_pad_id).sum().item())
        n_pet = int((ids == pet_pad_id).sum().item())
        if n_ct != self.num_ct_queries:
            raise ValueError(
                f"Prompt template produced {n_ct} <image_ct_pad> tokens; "
                f"expected {self.num_ct_queries}. Check that the special "
                "token was registered on the tokenizer (add_special_tokens)."
            )
        if n_pet != self.num_pet_queries:
            raise ValueError(
                f"Prompt template produced {n_pet} <image_pet_pad> tokens; "
                f"expected {self.num_pet_queries}."
            )

        # Cache; every sample reuses the same prompt.
        self._prompt_token_ids = ids
        self._prompt_attention_mask = mask

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Optional[dict]:
        row = self.rows[index]
        item = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)
        x_raw = item["x_raw"].float()
        windows = make_dual_stream_window_inputs(x_raw)
        metadata = item.get("metadata", {})
        report_text = metadata.get("report_text", row.get("report_text", ""))

        if self.require_report and not (report_text and report_text.strip()):
            # Lazy filter: returning None lets ignore_None_collate skip the
            # sample. Avoids static CSV peek (report may live in .pt sidecar).
            return None

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

        if self.tokenizer is not None:
            self._add_tokenized_fields(out, report_text)
        return out

    def _add_tokenized_fields(self, out: dict, report_text: str) -> None:
        """Populate prompt + report token tensors on ``out``."""
        from pillar.models.heads.report_lm import SPECIAL_TOKENS

        # Same prompt for every sample.
        out["prompt_token_ids"] = self._prompt_token_ids.clone()
        out["prompt_attention_mask"] = self._prompt_attention_mask.clone()

        # Report: append the EOR stop token before tokenization.
        eor = SPECIAL_TOKENS["eor"]
        text = (report_text or "").rstrip()
        if eor not in text:
            text = text + eor

        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_report_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        out["report_token_ids"] = enc["input_ids"][0]
        out["report_attention_mask"] = enc["attention_mask"][0]
