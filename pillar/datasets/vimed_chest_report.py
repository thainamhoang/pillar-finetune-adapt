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


# ---- SAMF (Style-Adaptive Multimodal Fusion) prompt + healthy templates ----
#
# PETRG-3D §4.2 + §5.2.1 + Fig. 4 show that injecting a "healthy reference
# report" template as ``<template>...</template>`` in the prompt is the
# single largest NLG-improving trick in their ablation (+5.4 BLEU-4,
# +5.95 ROUGE-L over DSFE-only). The mechanism: the LM no longer has to
# learn report STYLE on top of vision conditioning -- the style is handed
# to it as scaffolding. Model capacity is freed to focus on what's
# actually IN the image (deviations from the template).
#
# Their setup uses (Center_ID, Gender) keys -- multi-hospital + binary
# gender. ViMED is single-center so the center dimension collapses; we
# keep only the gender axis, which still matters for chest reports
# because breast tissue mentions differ.
#
# Authoring approach for these defaults: induced from the patterns
# observed across ViMED chest reports in the training set. Common
# baseline phrasing on healthy / unremarkable chest exams:
#   * Physiological FDG uptake in heart.
#   * No pleural / pericardial effusion.
#   * Lungs clear, no abnormal focal FDG uptake.
#   * Mediastinum / hila unremarkable.
#   * Esophagus normal.
#   * No axillary lymphadenopathy.
#   * Chest wall + pleura unremarkable.
#   * (Female only) breast tissue fibroglandular, no abnormal uptake.
#
# These are starting drafts; clinician sign-off is the natural next step
# once the SAMF wiring is verified to work mechanically.
CHEST_HEALTHY_TEMPLATES = {
    "male": (
        "Physiological FDG uptake is observed in the heart. "
        "No pleural effusion or pericardial effusion detected. "
        "The lung fields are clear with no abnormal focal FDG uptake. "
        "The mediastinum and hila show no enlarged lymph nodes or "
        "abnormal FDG metabolism. The esophagus appears normal without "
        "wall thickening or abnormal FDG uptake. No axillary "
        "lymphadenopathy is observed in either axillary fossa. "
        "The chest wall and pleura appear unremarkable."
    ),
    "female": (
        "Physiological FDG uptake is observed in the heart. "
        "No pleural effusion or pericardial effusion detected. "
        "The lung fields are clear with no abnormal focal FDG uptake. "
        "The mediastinum and hila show no enlarged lymph nodes or "
        "abnormal FDG metabolism. The esophagus appears normal without "
        "wall thickening or abnormal FDG uptake. No axillary "
        "lymphadenopathy is observed in either axillary fossa. "
        "The breast tissue appears fibroglandular with no abnormal masses "
        "or FDG uptake. The chest wall and pleura appear unremarkable."
    ),
}


# SAMF-enabled prompt template. Has two extra slots compared to the
# baseline: ``{healthy_template}`` (the gender-matched reference) and
# ``{gender_human}`` (a human-readable string for the instruction).
DEFAULT_CHEST_PROMPT_TEMPLATE_SAMF = (
    "You are a nuclear medicine physician specializing in chest PET/CT.\n"
    "The following are paired chest PET/CT images.\n"
    "<ct>{ct_pads}</ct>\n"
    "<pet>{pet_pads}</pet>\n"
    "Reference healthy chest PET/CT findings template ({gender_human} patient):\n"
    "<template>{healthy_template}</template>\n"
    'Generate the "Findings" section of the radiology report in English. '
    "Describe deviations from the reference template above; do not repeat "
    "it verbatim. Focus on observed pathology with location and size.\n"
)


def _resolve_gender_key(value: Any) -> str:
    """Map an arbitrary ``gender`` value to ``"male"`` or ``"female"``.

    Falls back to ``"male"`` for missing / unknown values. The male
    template is also a fine fallback for unisex content -- it's the
    superset minus the breast-tissue line.
    """
    if value is None:
        return "male"
    s = str(value).strip().lower()
    if s.startswith("f") or s in {"nữ", "nu", "woman", "women"}:
        return "female"
    return "male"


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
        # ---- SAMF (Style-Adaptive Multimodal Fusion) ----
        use_samf: bool = True,
        healthy_templates: Optional[dict[str, str]] = None,
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

        # SAMF state. healthy_templates is keyed by ``"male"``/``"female"``;
        # callers can override the defaults via YAML if they want
        # clinician-curated templates instead of the bootstrap ones above.
        self.use_samf = bool(use_samf)
        self.healthy_templates = (
            dict(healthy_templates) if healthy_templates is not None
            else dict(CHEST_HEALTHY_TEMPLATES)
        )

        if tokenizer is None and llm_repo_id is not None:
            tokenizer = self._build_tokenizer(llm_repo_id)
        self.tokenizer = tokenizer

        # Default require_report: True if tokenizing (training), else False.
        if require_report is None:
            require_report = tokenizer is not None
        self.require_report = require_report

        if tokenizer is not None:
            if prompt_template is not None:
                template = prompt_template  # user override; takes precedence
            elif self.use_samf:
                template = DEFAULT_CHEST_PROMPT_TEMPLATE_SAMF
            else:
                template = DEFAULT_CHEST_PROMPT_TEMPLATE
            self.prompt_template = template
            self._prebuild_prompt_variants()
            # We can't statically filter rows by report presence because
            # report_text may live in the .pt sidecar metadata rather than
            # the CSV. Lazy skip at __getitem__ time -- the project's
            # ignore_None_collate (pillar/utils/loading.py) already drops
            # None samples from each batch.
        else:
            self.prompt_template = None
            self._prompt_variants = None

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

    def _prebuild_prompt_variants(self) -> None:
        """Tokenize each prompt variant ONCE and cache it.

        Variants:
          * SAMF enabled: two variants, keyed ``"male"`` and ``"female"``,
            differing only in the ``<template>...</template>`` body. Per
            PETRG-3D §4.2, the healthy reference scaffolds the LM so it
            can focus on findings rather than style.
          * SAMF disabled: one variant under key ``""`` -- the original
            non-SAMF behavior.

        In every variant the placeholder counts (128 ``<image_ct_pad>`` +
        128 ``<image_pet_pad>``) are asserted to match the model's
        ``num_ct/pet_queries``. This is the integrity contract that
        ``DualStreamReportGenerator._splice_visual_embeds`` relies on.
        """
        from pillar.models.heads.report_lm import SPECIAL_TOKENS

        ct_pad_tok = SPECIAL_TOKENS["ct_pad"]
        pet_pad_tok = SPECIAL_TOKENS["pet_pad"]

        if self.use_samf:
            keys = ("male", "female")
        else:
            keys = ("",)

        self._prompt_variants: dict[str, dict[str, torch.Tensor]] = {}
        for key in keys:
            fmt_kwargs = dict(
                ct_pads=ct_pad_tok * self.num_ct_queries,
                pet_pads=pet_pad_tok * self.num_pet_queries,
            )
            if self.use_samf:
                fmt_kwargs["healthy_template"] = self.healthy_templates[key]
                fmt_kwargs["gender_human"] = key
            rendered = self.prompt_template.format(**fmt_kwargs)

            enc = self.tokenizer(
                rendered,
                add_special_tokens=False,
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
                    f"Prompt variant {key!r} produced {n_ct} <image_ct_pad> tokens; "
                    f"expected {self.num_ct_queries}."
                )
            if n_pet != self.num_pet_queries:
                raise ValueError(
                    f"Prompt variant {key!r} produced {n_pet} <image_pet_pad> tokens; "
                    f"expected {self.num_pet_queries}."
                )
            self._prompt_variants[key] = {"ids": ids, "mask": mask}

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

        # Resolve gender from metadata, falling back to the CSV row, then
        # to "male" if absent. Used by SAMF prompt routing AND exposed on
        # the output dict for downstream debugging / per-gender ablation.
        raw_gender = metadata.get("gender") or row.get("gender")
        gender_key = _resolve_gender_key(raw_gender)

        out = {
            "ct_windows": windows["ct_windows"],
            "pet_windows": windows["pet_windows"],
            "report_text": report_text,
            "study_id": row["study_id"],
            "accession": row["study_id"],
            "region": row["region"],
            "gender": gender_key,
            "metadata": metadata,
        }
        if "labels" in item:
            out["labels"] = item["labels"].float()
            out["label_names"] = list(item.get("label_names", []))
        if self.include_raw:
            out["x_raw"] = x_raw

        if self.tokenizer is not None:
            self._add_tokenized_fields(out, report_text, gender_key)
        return out

    def _add_tokenized_fields(
        self, out: dict, report_text: str, gender_key: str
    ) -> None:
        """Populate prompt + report token tensors on ``out``.

        With SAMF on, picks the male/female prompt variant based on
        ``gender_key``. Without SAMF, picks the single ``""`` variant.
        """
        from pillar.models.heads.report_lm import SPECIAL_TOKENS

        variant_key = gender_key if self.use_samf else ""
        variant = self._prompt_variants.get(variant_key)
        if variant is None:
            # Defensive fallback: if for some reason the variant key isn't
            # registered (e.g. SAMF was disabled at tokenize-time but the
            # caller is still passing gender), drop back to the first
            # available variant.
            variant = next(iter(self._prompt_variants.values()))

        out["prompt_token_ids"] = variant["ids"].clone()
        out["prompt_attention_mask"] = variant["mask"].clone()

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
