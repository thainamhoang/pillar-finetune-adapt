"""Frozen LLM + LoRA wrapper for report generation.

Loads a HuggingFace ``AutoModelForCausalLM`` from a config-supplied
``llm_repo_id``, adds vision/structure special tokens to its tokenizer,
resizes the embedding table, and (optionally) wraps with PEFT LoRA so
only the LoRA adapters + the projection-side params are trainable.

Special tokens added (all initialized to the mean of existing token
embeddings, the standard practice when extending a tokenizer):

- ``<ct>``, ``</ct>``  --- bracket the CT visual-token block in the prompt.
- ``<pet>``, ``</pet>`` --- same for PET.
- ``<image_ct_pad>``, ``<image_pet_pad>`` --- placeholder tokens that
  occupy the positions where the per-modality 128 resampler embeddings
  will be spliced in. The dataset inserts ``num_queries`` of each
  placeholder inside the corresponding bracket.
- ``<|end_of_report|>`` --- explicit stop token appended to every
  training report. Generation stops on this. The paper (PETRG-3D §5.2.2)
  documents this as the single biggest fix for post-report hallucination.

The ``forward`` is *not* the raw HF forward; it expects ``inputs_embeds``
(constructed by the parent ``DualStreamReportGenerator``) and ``labels``
(already -100-masked on the prompt span by the caller).

``generate`` likewise expects ``inputs_embeds`` and returns the newly
generated token ids only (stripping the visual-prompt prefix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


SPECIAL_TOKENS = {
    "ct_open":   "<ct>",
    "ct_close":  "</ct>",
    "pet_open":  "<pet>",
    "pet_close": "</pet>",
    "ct_pad":    "<image_ct_pad>",
    "pet_pad":   "<image_pet_pad>",
    "eor":       "<|end_of_report|>",
}


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    # PETRG-3D recipe targets the attention projections; we mirror that as
    # the safe default. Override in YAML if the chosen LLM uses different
    # module names (e.g. Gemma uses q_proj/k_proj/v_proj/o_proj which
    # matches; LLaMA also matches; Qwen3 MoE has different routing layers
    # so confirm before training).
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"


class ReportLM(nn.Module):
    """Wraps a HF causal LM. Handles tokenizer extension, embedding resize,
    LoRA application, and visual-token splicing.

    Construction is deferred until ``__init__`` so the *parent* model
    (``DualStreamReportGenerator``) can read off ``hidden_size`` /
    ``placeholder_id``s before constructing the per-modality projection
    that has to land in this LM's embedding dim.
    """

    def __init__(
        self,
        llm_repo_id: str,
        llm_revision: Optional[str] = None,
        num_ct_queries: int = 128,
        num_pet_queries: int = 128,
        torch_dtype: str = "bfloat16",
        lora: Optional[LoRAConfig] = None,
        apply_lora: bool = True,
        freeze_lm: bool = True,
        device_map: Optional[str | dict] = None,
    ) -> None:
        super().__init__()
        # Local imports keep `--help` paths cheap.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.llm_repo_id = llm_repo_id
        self.num_ct_queries = num_ct_queries
        self.num_pet_queries = num_pet_queries

        # ---- tokenizer ----
        tokenizer = AutoTokenizer.from_pretrained(
            llm_repo_id,
            revision=llm_revision,
            trust_remote_code=True,
            use_fast=True,
        )
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
        )
        # Ensure a pad token exists (some LMs don't ship one). Reuse EOS
        # if needed; the dataset handles padding masks separately.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.special_token_ids = {
            name: tokenizer.convert_tokens_to_ids(tok)
            for name, tok in SPECIAL_TOKENS.items()
        }

        # ---- model ----
        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_repo_id, revision=llm_revision, **model_kwargs
        )
        # Extend the embedding table for the new special tokens. New rows
        # are init'd to the mean of existing embeddings (standard practice
        # so they don't break attention magnitudes at step 0).
        if added > 0:
            self._resize_token_embeddings_with_mean_init(added)

        # Resolve text hidden_size robustly. Multimodal causal LMs (Gemma3,
        # PaliGemma, LLaVA, etc.) nest the text decoder config under
        # ``config.text_config`` and the top-level ``config.hidden_size``
        # may be missing or refer to the vision tower. Walk the config and
        # fall back to the input-embedding width, which IS the hidden dim
        # by construction.
        self.hidden_size = self._resolve_text_hidden_size()
        # Log the resolved model class + hidden size so it's visible in
        # smoke output (the failure mode this guards against is silently
        # picking the vision hidden dim instead of the text one).
        try:
            import logging as _logging
            _logging.getLogger(__name__).info(
                f"ReportLM: loaded {type(self.model).__name__} "
                f"(hidden_size={self.hidden_size}, vocab={len(self.tokenizer)})"
            )
        except Exception:
            pass

        # ---- freeze + LoRA ----
        if freeze_lm:
            for p in self.model.parameters():
                p.requires_grad = False
        if apply_lora:
            self._apply_lora(lora if lora is not None else LoRAConfig())

    # --- config resolution helpers ---

    def _resolve_text_hidden_size(self) -> int:
        """Robust hidden-size resolver for both pure-text and multimodal LMs.

        Tries, in order:
          1. ``config.hidden_size`` (Llama, Qwen, Mistral, Gemma2, ...)
          2. ``config.text_config.hidden_size`` (Gemma3 + multimodal,
             PaliGemma, LLaVA-family configs)
          3. ``model.get_input_embeddings().weight.shape[1]`` (last-resort
             but always correct -- by definition the input embedding dim
             IS the text hidden size).
        """
        cfg = self.model.config
        if hasattr(cfg, "hidden_size") and isinstance(getattr(cfg, "hidden_size"), int):
            return cfg.hidden_size
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        emb = self.model.get_input_embeddings()
        if emb is not None and emb.weight is not None:
            return int(emb.weight.shape[1])
        raise AttributeError(
            f"Could not resolve text hidden_size on "
            f"{type(self.model).__name__}; inspect config manually."
        )

    # --- embedding-table extension ---

    @torch.no_grad()
    def _resize_token_embeddings_with_mean_init(self, num_added: int) -> None:
        """Resize the model embedding table; init new rows to the existing mean.

        Some LMs (e.g. those with tied input/output embeddings) need this
        applied consistently to ``get_input_embeddings`` and
        ``get_output_embeddings``; ``resize_token_embeddings`` handles the
        tying.
        """
        old_emb = self.model.get_input_embeddings()
        old_num = old_emb.num_embeddings
        new_num = old_num + num_added

        # Compute mean BEFORE resizing (so it's over the original rows).
        mean_vec = old_emb.weight.mean(dim=0, keepdim=True)

        self.model.resize_token_embeddings(new_num)
        new_emb = self.model.get_input_embeddings()
        new_emb.weight[old_num:new_num] = mean_vec.to(new_emb.weight.dtype)

        # If the LM has an untied lm_head, do the same on the output side.
        out_emb = self.model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.shape[0] >= new_num:
            # Only init if shape changed (i.e. not tied).
            if not torch.equal(out_emb.weight[:old_num], new_emb.weight[:old_num].to(out_emb.weight.dtype)):
                pass  # tied, nothing to do
            else:
                out_emb.weight[old_num:new_num] = mean_vec.to(out_emb.weight.dtype)

    # --- LoRA ---

    def _apply_lora(self, cfg: LoRAConfig) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "peft is required for ReportLM(apply_lora=True). "
                "Install with `uv pip install peft>=0.11`."
            ) from e

        peft_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=list(cfg.target_modules),
            bias=cfg.bias,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_cfg)
        # peft sets requires_grad=True on LoRA params automatically;
        # frozen base params stay frozen.

    # --- public surface ---

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``(B, L) -> (B, L, hidden_size)``."""
        return self.get_input_embeddings()(token_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Direct passthrough to the HF model. ``labels`` should already be
        ``-100``-masked on the prompt/vision span; loss is the usual
        next-token CE on the report span (and ``<|end_of_report|>``).
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.7,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        stop_on_eor: bool = True,
    ) -> torch.Tensor:
        """Generate report tokens conditioned on prebuilt ``inputs_embeds``.

        Returns only the *newly generated* token ids (not the visual-prompt
        prefix). Caller decodes via ``self.tokenizer.decode(...)``.

        ``no_repeat_ngram_size``: when > 0, HF generate() hard-blocks any
        n-gram of that size from appearing twice in the generated
        sequence. Standard fix for the "Physiological X. Physiological X.
        Physiological X." failure mode (PETRG-3D §D.5, Fig. 8). Set to 4
        in the engine config; pass 0 to disable.
        """
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        if stop_on_eor:
            eor = self.special_token_ids["eor"]
            if eor not in eos_ids:
                eos_ids.append(eor)

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids if eos_ids else None,
        )
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        out = self.model.generate(**gen_kwargs)
        # When generating from inputs_embeds, HF returns ONLY the newly
        # generated token ids (the input has no token_ids to echo). Some
        # backends prepend a BOS — strip if present.
        if out.shape[1] > 0 and self.tokenizer.bos_token_id is not None:
            mask = out[:, 0] == self.tokenizer.bos_token_id
            if mask.all():
                out = out[:, 1:]
        return out

    # --- introspection ---

    def trainable_parameter_count(self) -> tuple[int, int]:
        """Return ``(trainable, total)`` parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
