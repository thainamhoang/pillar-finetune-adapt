"""Dual-stream PET/CT report generator (PETRG-3D aligned).

End-to-end model that composes:

1. Two frozen Pillar0-initialized Atlas encoders (CT, PET) loaded from the
   Phase A export (``logs/dual-stream-pillar/fusion/dual_stream_encoder.pt``).
   The Phase A ``TokenConcatFusion`` keys in that checkpoint are ignored —
   Phase B does NOT do encoder-side fusion; that happens at the LLM prompt
   level instead.
2. Two independent Perceiver Resamplers (128 queries × 1152 each, 6 layers).
3. Two independent Vision→LLM projection MLPs.
4. One ``ReportLM`` (frozen LLM + LoRA + extended tokenizer).

Forward inputs (from ``ViMedChestReportDataset`` with tokenizer wired):
  ``ct_windows``        (B, 11, D, H, W)
  ``pet_windows``       (B,  4, D, H, W)
  ``prompt_token_ids``  (B, L_prompt)  — contains 128×<image_ct_pad> + 128×<image_pet_pad>
  ``prompt_attention_mask`` (B, L_prompt)
  ``report_token_ids``  (B, L_report)  — report + <|end_of_report|>
  ``report_attention_mask`` (B, L_report)

Forward returns ``{"loss": ..., "logits": ...}``.

For generation, call ``model.generate(batch, ...)`` — it builds the same
prompt prefix and lets the LLM continue.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from pillar.models.abstract_model import AbstractModel
from pillar.models.backbones import ViMedChestDualStreamEncoders
from pillar.models.heads.perceiver_resampler import PerceiverResampler
from pillar.models.heads.vision_projection import VisionProjection
from pillar.models.heads.report_lm import LoRAConfig, ReportLM, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


def _strip_module_prefix(state_dict: dict, prefix: str) -> dict:
    """Keep only keys starting with ``prefix`` and strip it."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


class DualStreamReportGenerator(AbstractModel):
    """Dual-stream PET/CT → English report generator.

    Parameters
    ----------
    encoder_ckpt:
        Path to the Phase A ``dual_stream_encoder.pt`` artifact. Loaded with
        ``weights_only=False``; the ``state_dict`` substate is filtered to
        the dual-stream encoder weights only (fusion + classification keys
        are dropped). If ``None``, encoders are initialized from HF
        ``model_repo_id`` and you should overlay manually.
    ct_channels, pet_channels:
        Match what the Phase A encoders were trained with. Defaults
        ``11`` (CT) and ``4`` (PET) match the dual-stream finetune.
    freeze_encoders:
        Paper-validated default ``True``. Fine-tuning encoders on small
        downstream data hurts (PETRG-3D Fig. 4).
    num_ct_queries, num_pet_queries:
        Per-modality perceiver-resampler output token count. 128 each
        (paper default).
    resampler_depth:
        Transformer layers in each perceiver resampler. 6 (paper default).
    llm_repo_id:
        HF repo of the causal LM. Default ``google/medgemma-1.5-4b-it``;
        ``Qwen/Qwen3.5-9B`` or ``Qwen/Qwen3-8B`` are validated alternatives.
    """

    def __init__(
        self,
        args,
        encoder_ckpt: Optional[str] = None,
        ct_channels: int = 11,
        pet_channels: int = 4,
        anatomy: str = "chest_ct",
        model_repo_id: str = "YalaLab/Pillar0-ChestCT",
        model_revision: str = "main",
        encoder_device: Optional[str] = None,
        freeze_encoders: bool = True,
        # Resampler / projection
        num_ct_queries: int = 128,
        num_pet_queries: int = 128,
        resampler_depth: int = 6,
        resampler_num_heads: int = 8,
        resampler_ffn_mult: int = 4,
        resampler_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        token_dim: int = 1152,
        # LLM
        llm_repo_id: str = "google/medgemma-1.5-4b-it",
        llm_revision: Optional[str] = None,
        llm_torch_dtype: str = "bfloat16",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list[str]] = None,
        apply_lora: bool = True,
        freeze_lm: bool = True,
    ) -> None:
        super().__init__(args)

        # ---- Encoders (frozen) ----
        self.dual_stream_encoders = ViMedChestDualStreamEncoders(
            ct_channels=ct_channels,
            pet_channels=pet_channels,
            anatomy=anatomy,
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            device=encoder_device,
            # NOTE: we do NOT pass per-encoder ckpts here -- the unified
            # Phase A artifact is loaded below as a single state_dict.
        )
        if encoder_ckpt is not None:
            self._load_phase_a_encoders(encoder_ckpt)
        if freeze_encoders:
            for p in self.dual_stream_encoders.parameters():
                p.requires_grad = False
            self.dual_stream_encoders.eval()

        # ---- Per-modality resamplers ----
        self.ct_resampler = PerceiverResampler(
            dim=token_dim,
            num_queries=num_ct_queries,
            depth=resampler_depth,
            num_heads=resampler_num_heads,
            ffn_mult=resampler_ffn_mult,
            attn_dropout=resampler_dropout,
            ffn_dropout=resampler_dropout,
        )
        self.pet_resampler = PerceiverResampler(
            dim=token_dim,
            num_queries=num_pet_queries,
            depth=resampler_depth,
            num_heads=resampler_num_heads,
            ffn_mult=resampler_ffn_mult,
            attn_dropout=resampler_dropout,
            ffn_dropout=resampler_dropout,
        )

        # ---- LLM (built BEFORE projections so we know hidden_size) ----
        lora_cfg = LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules
            if lora_target_modules is not None
            else ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.lm = ReportLM(
            llm_repo_id=llm_repo_id,
            llm_revision=llm_revision,
            num_ct_queries=num_ct_queries,
            num_pet_queries=num_pet_queries,
            torch_dtype=llm_torch_dtype,
            lora=lora_cfg,
            apply_lora=apply_lora,
            freeze_lm=freeze_lm,
        )

        # ---- Per-modality adapters ----
        self.ct_adapter = VisionProjection(
            in_dim=token_dim,
            out_dim=self.lm.hidden_size,
            dropout=projection_dropout,
        )
        self.pet_adapter = VisionProjection(
            in_dim=token_dim,
            out_dim=self.lm.hidden_size,
            dropout=projection_dropout,
        )

        # ---- Cached IDs for splicing ----
        self.ct_pad_id = self.lm.special_token_ids["ct_pad"]
        self.pet_pad_id = self.lm.special_token_ids["pet_pad"]
        self.eor_id = self.lm.special_token_ids["eor"]
        self.num_ct_queries = num_ct_queries
        self.num_pet_queries = num_pet_queries
        self.token_dim = token_dim

        # Convenience: expose hidden_dim like the backbone classes do, in
        # case anything in pillar.utils expects it.
        self.hidden_dim = self.lm.hidden_size

        # ---- Param accounting log ----
        trainable, total = self._param_count()
        logger.info(
            f"DualStreamReportGenerator: {trainable/1e6:.1f}M trainable / "
            f"{total/1e6:.1f}M total params"
        )

    # ------------------------------------------------------------------
    # Encoder loading from Phase A artifact
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _load_phase_a_encoders(self, ckpt_path: str) -> None:
        """Pull the dual_stream.ct_encoder.* and dual_stream.pet_encoder.*
        substates out of the Phase A artifact and load them into our local
        ``ViMedChestDualStreamEncoders``. Fusion params (if present) are
        dropped — Phase B uses prompt-level fusion only.
        """
        logger.info(f"Loading Phase A encoder substate from {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw:
            sd = raw["model"]
        else:
            sd = raw

        # Strip the "dual_stream." prefix that ViMedChestDualStreamFusedEncoder
        # introduces, keeping only ct_encoder.*/pet_encoder.* keys. We DROP
        # fusion.* keys entirely.
        prefix = "dual_stream."
        encoder_sd = {}
        n_fusion = 0
        for k, v in sd.items():
            if k.startswith(prefix):
                k2 = k[len(prefix):]
                if k2.startswith("ct_encoder.") or k2.startswith("pet_encoder."):
                    encoder_sd[k2] = v
            elif k.startswith("ct_encoder.") or k.startswith("pet_encoder."):
                # Already in the right shape (e.g. raw dual_stream_encoders save).
                encoder_sd[k] = v
            elif k.startswith("fusion.") or k.startswith(prefix + "fusion."):
                n_fusion += 1

        missing, unexpected = self.dual_stream_encoders.load_state_dict(
            encoder_sd, strict=False
        )
        logger.info(
            f"Loaded encoder substate: {len(encoder_sd)} keys; "
            f"{len(missing)} missing, {len(unexpected)} unexpected; "
            f"{n_fusion} fusion keys ignored (Phase B uses prompt-level fusion)."
        )

    # ------------------------------------------------------------------
    # Prompt embed assembly
    # ------------------------------------------------------------------
    def _splice_visual_embeds(
        self,
        prompt_token_ids: torch.Tensor,        # (B, L_prompt)
        prompt_attention_mask: torch.Tensor,   # (B, L_prompt)
        ct_embeds: torch.Tensor,               # (B, num_ct_queries, H)
        pet_embeds: torch.Tensor,              # (B, num_pet_queries, H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace ``<image_ct_pad>`` / ``<image_pet_pad>`` token embeddings
        in the prompt with the corresponding visual embeds. Returns
        ``(input_embeds, attention_mask)``.

        Assumes the dataset placed exactly ``num_ct_queries`` ``<image_ct_pad>``
        tokens and ``num_pet_queries`` ``<image_pet_pad>`` tokens in the
        prompt (in that order, inside the bracket tokens).
        """
        B, L = prompt_token_ids.shape
        H = ct_embeds.shape[-1]
        device = prompt_token_ids.device

        prompt_embeds = self.lm.embed_tokens(prompt_token_ids)  # (B, L, H)

        # Match dtype/device of the LM embedding table so the splice doesn't
        # accidentally upcast/move tensors.
        ct_embeds = ct_embeds.to(prompt_embeds.dtype)
        pet_embeds = pet_embeds.to(prompt_embeds.dtype)

        # For each batch row, find positions of the pad tokens and write
        # the visual embeddings into those positions.
        for b in range(B):
            ct_mask = prompt_token_ids[b] == self.ct_pad_id
            pet_mask = prompt_token_ids[b] == self.pet_pad_id
            n_ct = int(ct_mask.sum().item())
            n_pet = int(pet_mask.sum().item())
            if n_ct != self.num_ct_queries:
                raise ValueError(
                    f"Batch row {b}: found {n_ct} <image_ct_pad> tokens, "
                    f"expected {self.num_ct_queries}. Check prompt template."
                )
            if n_pet != self.num_pet_queries:
                raise ValueError(
                    f"Batch row {b}: found {n_pet} <image_pet_pad> tokens, "
                    f"expected {self.num_pet_queries}."
                )
            prompt_embeds[b, ct_mask] = ct_embeds[b]
            prompt_embeds[b, pet_mask] = pet_embeds[b]

        return prompt_embeds, prompt_attention_mask

    def _build_train_inputs(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        report_token_ids: torch.Tensor,
        report_attention_mask: torch.Tensor,
        ct_embeds: torch.Tensor,
        pet_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concat prompt embeds + report embeds; build labels with -100 on
        the prompt span (loss is computed only on the report).
        """
        prompt_embeds, _ = self._splice_visual_embeds(
            prompt_token_ids, prompt_attention_mask, ct_embeds, pet_embeds
        )
        report_embeds = self.lm.embed_tokens(report_token_ids)

        input_embeds = torch.cat([prompt_embeds, report_embeds], dim=1)
        attn_mask = torch.cat(
            [prompt_attention_mask, report_attention_mask], dim=1
        )

        # Labels: -100 over prompt; report ids over report span. Apply the
        # report attention mask so padded positions don't contribute to loss.
        B, Lp = prompt_token_ids.shape
        Lr = report_token_ids.shape[1]
        labels = torch.full(
            (B, Lp + Lr), -100, dtype=torch.long, device=input_embeds.device
        )
        labels[:, Lp:] = report_token_ids
        # Mask pad positions in the report.
        pad_id = self.lm.tokenizer.pad_token_id
        if pad_id is not None:
            labels[:, Lp:][report_token_ids == pad_id] = -100
        # Also respect the attention mask in case dataset uses arbitrary
        # padding scheme.
        labels[:, Lp:][report_attention_mask == 0] = -100

        # Strengthen the EOR (end-of-report) supervision. As-is, only ONE
        # EOR target exists per sample (the single EOR token at the end of
        # report_token_ids). That's <1% of the loss signal, drowning out
        # the gradient that should teach the model to terminate. PETRG-3D
        # §D.5 names "explicit end-of-report token" as the single most
        # impactful fix for runaway generation. To make it stick:
        #
        # For each batch row, find the first EOR position and overwrite all
        # subsequent labels in the report span (currently -100 due to
        # padding) with the EOR token id. The input_ids stay padded
        # (attention_mask=0 there, so they don't enter attention), but the
        # model is now supervised to keep emitting EOR after content -- a
        # strong attractor for "once you finish, stop."
        eor_id = self.eor_id
        report_labels = labels[:, Lp:]  # view into labels (shape: B, Lr)
        for b in range(B):
            eor_positions = (report_token_ids[b] == eor_id).nonzero(as_tuple=True)[0]
            if eor_positions.numel() == 0:
                # Sample's report didn't fit -- EOR was truncated. Nothing
                # to anchor on, leave labels alone.
                continue
            first_eor = int(eor_positions[0].item())
            # Supervise EOR at every position from first_eor onward.
            report_labels[b, first_eor:] = eor_id

        return input_embeds, attn_mask, labels

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------
    def _encode_visual(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoders (frozen, no_grad) → resamplers → adapters.

        Returns ``(ct_emb, pet_emb)`` each of shape
        ``(B, num_queries, llm_hidden)``.
        """
        ct_windows = batch["ct_windows"]
        pet_windows = batch["pet_windows"]
        with torch.no_grad():
            enc_out = self.dual_stream_encoders(ct_windows, pet_windows)
            ct_activ = enc_out["ct_activ"]    # (B, 1152, 32, 32, 16)
            pet_activ = enc_out["pet_activ"]

        ct_lat = self.ct_resampler(ct_activ)   # (B, 128, 1152)
        pet_lat = self.pet_resampler(pet_activ)
        ct_emb = self.ct_adapter(ct_lat)       # (B, 128, llm_hidden)
        pet_emb = self.pet_adapter(pet_lat)
        return ct_emb, pet_emb

    def forward(self, batch: dict, **extras) -> dict:
        """Training forward. Expects batch keys listed at the module docstring."""
        ct_emb, pet_emb = self._encode_visual(batch)

        input_embeds, attention_mask, labels = self._build_train_inputs(
            prompt_token_ids=batch["prompt_token_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            report_token_ids=batch["report_token_ids"],
            report_attention_mask=batch["report_attention_mask"],
            ct_embeds=ct_emb,
            pet_embeds=pet_emb,
        )

        out = self.lm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": out.loss,
            "logits": out.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        batch: dict,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.7,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate a report for each sample in ``batch``.

        Returns a list of decoded strings (one per batch row), with any
        leading/trailing ``<|end_of_report|>`` or pad tokens stripped.
        """
        ct_emb, pet_emb = self._encode_visual(batch)
        prompt_embeds, attn = self._splice_visual_embeds(
            batch["prompt_token_ids"],
            batch["prompt_attention_mask"],
            ct_emb,
            pet_emb,
        )
        gen_ids = self.lm.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
        )
        texts = []
        for ids in gen_ids:
            txt = self.lm.tokenizer.decode(ids, skip_special_tokens=False)
            # Strip the EOR marker so downstream metrics see clean text.
            eor_tok = SPECIAL_TOKENS["eor"]
            if eor_tok in txt:
                txt = txt.split(eor_tok)[0]
            # Strip pad tokens.
            if self.lm.tokenizer.pad_token is not None:
                txt = txt.replace(self.lm.tokenizer.pad_token, "")
            texts.append(txt.strip())
        return texts

    # ------------------------------------------------------------------
    def _param_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def no_weight_decay(self):
        """Match the convention of MultiStage / Atlas; perceiver query
        params and LayerNorms shouldn't get weight decay.
        """
        names = []
        for n, _ in self.named_parameters():
            if "queries" in n or "norm" in n or "bias" in n:
                names.append(n)
        return names
