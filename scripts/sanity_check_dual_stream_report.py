#!/usr/bin/env python3
"""Pre-training sanity check for the Phase B dual-stream report pipeline.

Mirrors the section structure of ``scripts/sanity_check_dual_stream.py``
but covers the Phase B additions:

1. Phase A encoder artifact exists + loadable.
2. Dependencies installed (peft, sacrebleu, rouge-score, NLG tokenizer).
3. Tokenizer loads + special tokens registered + prompt template
   tokenizes to exactly 128 + 128 placeholders.
4. ``ViMedChestReportDataset`` (with tokenizer wired) emits the full
   set of tensors and the prompt token IDs include the placeholders.
5. ``DualStreamReportGenerator`` builds end-to-end on the requested
   device.
6. One training forward returns finite ``loss``.
7. One ``generate`` call returns non-empty text and terminates cleanly
   (no runaway -- bounded by ``max_new_tokens``).

Sections 5-7 require a GPU + HF access for the LLM. Pass
``--no-model`` to skip them and only check sections 1-4.

Usage::

    uv run python scripts/sanity_check_dual_stream_report.py \\
        --encoder-ckpt logs/dual-stream-pillar/fusion/dual_stream_encoder.pt \\
        --manifest /scratch/thahoa/PET/ViMed_prep_v2/manifest_splits.csv \\
        --llm-repo-id google/medgemma-1.5-4b-it
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


GREEN, YELLOW, RED, CYAN, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[1m", "\033[0m"
)
_results: list[tuple[str, str, str]] = []


def _emit(level: str, color: str, section: str, message: str) -> None:
    print(f"{color}[{level}]{RESET} {BOLD}{section}{RESET} — {message}")
    _results.append((level, section, message))


def passed(s, m):  _emit("PASS",  GREEN,  s, m)
def warned(s, m):  _emit("WARN",  YELLOW, s, m)
def failed(s, m):  _emit("FAIL",  RED,    s, m)
def info(s, m):    print(f"{CYAN}[INFO]{RESET} {BOLD}{s}{RESET} — {m}")


def section_header(title: str) -> None:
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")


# ----- 1. Encoder artifact -----

def check_encoder_artifact(path: Path) -> bool:
    section_header("1. Phase A encoder artifact")
    if not path.exists():
        failed("artifact", f"missing: {path}")
        return False
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        failed("artifact", f"torch.load raised: {e}")
        return False

    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cfg = raw.get("config", {})
    else:
        sd = raw
        cfg = {}

    n_ct = sum(1 for k in sd if k.startswith(("dual_stream.ct_encoder.", "ct_encoder.")))
    n_pet = sum(1 for k in sd if k.startswith(("dual_stream.pet_encoder.", "pet_encoder.")))
    n_fus = sum(1 for k in sd if "fusion" in k)
    info("artifact", f"{len(sd)} keys total; ct_encoder={n_ct}, pet_encoder={n_pet}, fusion={n_fus}")
    if cfg:
        info("artifact", f"config: {cfg}")
    if n_ct == 0 or n_pet == 0:
        failed("artifact", "no ct_encoder or pet_encoder substate found")
        return False
    passed("artifact", "both encoder substates present")
    return True


# ----- 2. Dependencies -----

def check_deps() -> bool:
    section_header("2. Phase B dependencies")
    ok = True
    for mod in ("peft", "sacrebleu", "rouge_score"):
        try:
            importlib.import_module(mod)
            passed("deps", f"{mod} import OK")
        except Exception as e:
            failed("deps", f"{mod} import failed: {e}")
            ok = False
    try:
        importlib.import_module("transformers")
        passed("deps", "transformers import OK")
    except Exception as e:
        failed("deps", f"transformers import failed: {e}")
        ok = False
    return ok


# ----- 3. Tokenizer + prompt -----

def check_tokenizer(llm_repo_id: str, num_ct: int, num_pet: int) -> bool:
    section_header(f"3. Tokenizer + prompt template ({llm_repo_id})")
    try:
        from transformers import AutoTokenizer
        from pillar.models.heads.report_lm import SPECIAL_TOKENS
        from pillar.datasets.vimed_chest_report import (
            DEFAULT_CHEST_PROMPT_TEMPLATE,
            DEFAULT_CHEST_PROMPT_TEMPLATE_SAMF,
            CHEST_HEALTHY_TEMPLATES,
        )
    except Exception as e:
        failed("tokenizer", f"import failed: {e}")
        return False

    try:
        tok = AutoTokenizer.from_pretrained(llm_repo_id, trust_remote_code=True, use_fast=True)
    except Exception as e:
        failed("tokenizer", f"AutoTokenizer.from_pretrained failed: {e}")
        return False
    n_added = tok.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    )
    info("tokenizer", f"added {n_added} special tokens (vocab now {len(tok)})")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ct_pad = SPECIAL_TOKENS["ct_pad"]
    pet_pad = SPECIAL_TOKENS["pet_pad"]

    def _check(label: str, rendered: str) -> bool:
        enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"][0]
        n_ct_tok = int((ids == tok.convert_tokens_to_ids(ct_pad)).sum().item())
        n_pet_tok = int((ids == tok.convert_tokens_to_ids(pet_pad)).sum().item())
        info(f"tokenizer[{label}]",
             f"len={len(ids)} #ct_pad={n_ct_tok} #pet_pad={n_pet_tok}")
        if n_ct_tok != num_ct or n_pet_tok != num_pet:
            failed(f"tokenizer[{label}]",
                   f"placeholder count mismatch: ct={n_ct_tok}/{num_ct}, "
                   f"pet={n_pet_tok}/{num_pet}")
            return False
        passed(f"tokenizer[{label}]", "placeholder counts match")
        return True

    # Non-SAMF baseline.
    rendered_base = DEFAULT_CHEST_PROMPT_TEMPLATE.format(
        ct_pads=ct_pad * num_ct, pet_pads=pet_pad * num_pet
    )
    base_ok = _check("baseline", rendered_base)

    # SAMF, both genders. Template tokens should appear in the rendered
    # form; check that <template> / </template> got tokenized as single
    # special tokens.
    samf_ok = True
    template_open_id = tok.convert_tokens_to_ids(SPECIAL_TOKENS["template_open"])
    template_close_id = tok.convert_tokens_to_ids(SPECIAL_TOKENS["template_close"])
    for gender_key in ("male", "female"):
        rendered = DEFAULT_CHEST_PROMPT_TEMPLATE_SAMF.format(
            ct_pads=ct_pad * num_ct,
            pet_pads=pet_pad * num_pet,
            healthy_template=CHEST_HEALTHY_TEMPLATES[gender_key],
            gender_human=gender_key,
        )
        if not _check(f"samf:{gender_key}", rendered):
            samf_ok = False
            continue
        enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"][0]
        n_open = int((ids == template_open_id).sum().item())
        n_close = int((ids == template_close_id).sum().item())
        if n_open != 1 or n_close != 1:
            failed(f"tokenizer[samf:{gender_key}]",
                   f"<template>/<template/> counts wrong: open={n_open}, close={n_close}")
            samf_ok = False
        else:
            passed(f"tokenizer[samf:{gender_key}]",
                   "<template>...</template> wraps the healthy reference")

    return base_ok and samf_ok


# ----- 4. Dataset with tokenizer wired -----

def check_dataset(manifest: Path, llm_repo_id: str, num_ct: int, num_pet: int) -> dict | None:
    section_header("4. ViMedChestReportDataset with tokenizer")
    try:
        from pillar.datasets.vimed_chest_report import ViMedChestReportDataset
    except Exception as e:
        failed("dataset", f"import failed: {e}")
        return None

    try:
        ds = ViMedChestReportDataset(
            csv_path=manifest,
            split_group="val",
            region="chest",
            llm_repo_id=llm_repo_id,
            num_ct_queries=num_ct,
            num_pet_queries=num_pet,
            max_report_tokens=512,
            require_report=False,  # don't depend on report-text in this sanity check
            use_samf=True,
        )
    except Exception as e:
        failed("dataset", f"instantiation failed: {e}")
        traceback.print_exc()
        return None

    # SAMF wiring: dataset must hold both male and female prompt variants.
    variants = getattr(ds, "_prompt_variants", None)
    if not isinstance(variants, dict) or set(variants.keys()) != {"male", "female"}:
        failed("dataset[samf]",
               f"_prompt_variants keys = {set(variants.keys()) if variants else None}, "
               "expected {'male', 'female'}")
    else:
        # The two variants should differ -- different gender_human + healthy
        # template body. Check by comparing token-id tensors length OR content.
        m_ids = variants["male"]["ids"]
        f_ids = variants["female"]["ids"]
        if torch.equal(m_ids, f_ids):
            failed("dataset[samf]",
                   "male and female prompt variants are byte-identical "
                   "(template substitution may not be wired)")
        else:
            passed("dataset[samf]",
                   f"male prompt len={len(m_ids)}, female prompt len={len(f_ids)} "
                   "(distinct as expected)")

    if len(ds) == 0:
        failed("dataset", "empty for split=val region=chest")
        return None
    info("dataset", f"len={len(ds)}")

    item = None
    for i in range(min(len(ds), 8)):  # find one with a non-empty report if possible
        cand = ds[i]
        if cand is not None and cand.get("report_text"):
            item = cand
            break
    if item is None:
        item = ds[0]
        if item is None:
            failed("dataset", "first 8 rows all returned None (report filter?)")
            return None
        warned("dataset", "no sample with non-empty report in the first 8; "
                          "Phase B training will need require_report=True data")

    required = {"ct_windows", "pet_windows", "prompt_token_ids",
                "prompt_attention_mask", "report_token_ids",
                "report_attention_mask", "gender"}
    missing = required - set(item.keys())
    if missing:
        failed("dataset", f"item missing keys: {missing}")
        return None
    passed("dataset", "item has all Phase B keys")

    # Gender field should be one of the registered keys.
    gk = item.get("gender")
    if gk in ("male", "female"):
        info("dataset", f"gender resolved to {gk!r}")
        passed("dataset", "gender routing works on real sample")
    else:
        failed("dataset", f"unexpected gender value: {gk!r}")

    for k in ("ct_windows", "pet_windows", "prompt_token_ids", "report_token_ids"):
        info("dataset", f"{k}: shape={tuple(item[k].shape)} dtype={item[k].dtype}")
    if item["ct_windows"].shape[0] != 11:
        failed("dataset", f"ct_windows ch={item['ct_windows'].shape[0]}, expected 11")
    if item["pet_windows"].shape[0] != 4:
        failed("dataset", f"pet_windows ch={item['pet_windows'].shape[0]}, expected 4")
    return item


# ----- 5-7. Model build + forward + generate -----

def check_model(encoder_ckpt: Path, llm_repo_id: str, item: dict, device: str,
                num_ct: int, num_pet: int) -> None:
    section_header("5. DualStreamReportGenerator build")
    try:
        from pillar.models import DualStreamReportGenerator
    except Exception as e:
        failed("model", f"import failed: {e}")
        return

    try:
        # Args namespace mimic -- the model itself only needs `args` for
        # AbstractModel.__init__; nothing in this sanity path reads it.
        from types import SimpleNamespace
        fake_args = SimpleNamespace()
        model = DualStreamReportGenerator(
            args=fake_args,
            encoder_ckpt=str(encoder_ckpt),
            ct_channels=11,
            pet_channels=4,
            num_ct_queries=num_ct,
            num_pet_queries=num_pet,
            resampler_depth=6,
            llm_repo_id=llm_repo_id,
            llm_torch_dtype="bfloat16",
            apply_lora=True,
            freeze_lm=True,
        )
    except Exception as e:
        failed("model", f"build failed: {e}")
        traceback.print_exc()
        return

    trainable, total = model._param_count()
    info("model", f"params: trainable={trainable/1e6:.1f}M, total={total/1e6:.1f}M")
    passed("model", "DualStreamReportGenerator built (encoders + resamplers + LM + LoRA)")

    model = model.to(device)
    model.eval()

    # --- 6. Forward ---
    section_header("6. Single training forward")
    batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v])
             for k, v in item.items()}
    try:
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu",
                                enabled=(device == "cuda"), dtype=torch.bfloat16):
            with torch.no_grad():
                out = model(batch)
    except Exception as e:
        failed("forward", f"raised: {e}")
        traceback.print_exc()
        return

    loss = out.get("loss")
    if loss is None:
        failed("forward", "model output missing 'loss'")
        return
    info("forward", f"loss = {loss.item():.4f}")
    if not torch.isfinite(loss):
        failed("forward", "loss is NaN/Inf")
        return
    passed("forward", "loss is finite")

    # --- 7. Generate ---
    section_header("7. Single generate call")
    try:
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu",
                                enabled=(device == "cuda"), dtype=torch.bfloat16):
            gens = model.generate(
                batch,
                max_new_tokens=64,  # short for sanity
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.05,
            )
    except Exception as e:
        failed("generate", f"raised: {e}")
        traceback.print_exc()
        return

    if not gens or not gens[0]:
        warned("generate", "empty generation -- may be normal at step 0 (untrained adapters)")
    else:
        info("generate", f"generated {len(gens[0])} chars: {gens[0][:200]!r}")
        passed("generate", "non-empty text produced and capped at max_new_tokens")


# ----- main -----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder-ckpt", required=True, type=Path,
                    help="Phase A export (dual_stream_encoder.pt)")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--llm-repo-id", default="google/medgemma-1.5-4b-it")
    ap.add_argument("--num-ct-queries", type=int, default=128)
    ap.add_argument("--num-pet-queries", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-model", action="store_true",
                    help="Skip sections 5-7 (model build / forward / generate). "
                         "Use to check tokenizer + dataset path without an LLM download.")
    args = ap.parse_args()

    artifact_ok = check_encoder_artifact(args.encoder_ckpt)
    deps_ok = check_deps()
    tok_ok = check_tokenizer(args.llm_repo_id, args.num_ct_queries, args.num_pet_queries)
    item = check_dataset(args.manifest, args.llm_repo_id,
                         args.num_ct_queries, args.num_pet_queries)

    if not args.no_model:
        if not (artifact_ok and deps_ok and tok_ok and item is not None):
            warned("model", "skipping sections 5-7 because earlier sections failed")
        else:
            device = args.device
            if device == "cuda" and not torch.cuda.is_available():
                warned("device", "CUDA unavailable; using CPU (slow)")
                device = "cpu"
            check_model(args.encoder_ckpt, args.llm_repo_id, item, device,
                        args.num_ct_queries, args.num_pet_queries)
    else:
        info("model", "skipped (--no-model)")

    section_header("Summary")
    counts = Counter(level for level, _, _ in _results)
    print(f"  {GREEN}PASS{RESET}: {counts['PASS']}")
    print(f"  {YELLOW}WARN{RESET}: {counts['WARN']}")
    print(f"  {RED}FAIL{RESET}: {counts['FAIL']}")
    if counts["FAIL"] > 0:
        print(f"\n{RED}{BOLD}❌ Phase B pipeline has issues. Fix FAILs before training.{RESET}")
        return 1
    if counts["WARN"] > 0:
        print(f"\n{YELLOW}{BOLD}⚠  Runnable but check WARNs.{RESET}")
        return 0
    print(f"\n{GREEN}{BOLD}✅ Phase B pipeline ready.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
