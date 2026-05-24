"""Training engine for dual-stream PET/CT report generation.

Modeled on :class:`pillar.engines.classifier.Classifier` but adapted to the
LM-loss + free-form generation regime:

- ``step()`` calls ``model(batch)`` (whole dict) rather than ``model(batch["x"])``.
  The model returns ``{"loss": <scalar>, "logits": ...}`` -- LM cross-entropy
  is computed inside the HF causal LM, not via a configured loss function.
- ``evaluate()`` runs a teacher-forced LM loss pass for ``val/lm_loss`` /
  ``val/ppl`` AND a free-run ``generate()`` pass on ``eval_generate_n_batches``
  to produce reports for BLEU/ROUGE/METEOR computation. Generated samples
  are logged to wandb as a table (reference vs generated).
- ``preprocess_batch()`` only moves tensors to device; no windowing (already
  applied in the dataset).

NLG metrics use ``sacrebleu`` (BLEU-1..4) and ``rouge-score`` (ROUGE-L).
METEOR via ``nltk.translate.meteor`` if available (optional).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import torch
import wandb
from tqdm import tqdm

from pillar.utils.logging import logger
from pillar.utils.engine import gather_predictions_dict, prefix_dict
from pillar.utils.memdebug import print_mem
from pillar.utils.misc import AverageMeter, Summary, ProgressMeter, get_is_master

from .base import Engine
from .classifier import _memory_metrics  # reuse the GPU+host snapshot helper


def _compute_nlg_metrics(refs: list[str], gens: list[str]) -> dict[str, float]:
    """Corpus-level NLG metrics. Robust to empty inputs."""
    out: dict[str, float] = {}
    if not refs or not gens:
        return out
    n = min(len(refs), len(gens))
    refs = refs[:n]
    gens = gens[:n]

    # sacrebleu — corpus-level
    try:
        import sacrebleu
        # sacrebleu expects list-of-list-of-references at the corpus call.
        bleu = sacrebleu.corpus_bleu(gens, [refs])
        out["bleu_4"] = bleu.score / 100.0
        if bleu.precisions and len(bleu.precisions) >= 4:
            for i, p in enumerate(bleu.precisions[:4], start=1):
                out[f"bleu_{i}"] = p / 100.0
    except ImportError:
        logger.warning("sacrebleu not installed; skipping BLEU.")

    # rouge-score — average over the corpus
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l = 0.0
        for ref, gen in zip(refs, gens):
            sc = scorer.score(ref, gen)
            rouge_l += sc["rougeL"].fmeasure
        out["rouge_l"] = rouge_l / max(1, len(refs))
    except ImportError:
        logger.warning("rouge-score not installed; skipping ROUGE-L.")

    # METEOR (optional)
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception:
                pass
        meteor = 0.0
        for ref, gen in zip(refs, gens):
            try:
                meteor += meteor_score([ref.split()], gen.split())
            except Exception:
                pass
        out["meteor"] = meteor / max(1, len(refs))
    except ImportError:
        pass

    return out


class ReportGenerator(Engine):
    """LM-loss + generation training engine for PET/CT → report."""

    def __init__(
        self,
        *args,
        log_interval: int = 25,
        eval_generate_n_batches: int = 2,
        eval_generate_max_new_tokens: int = 1024,
        eval_generate_top_p: float = 0.9,
        eval_generate_temperature: float = 0.7,
        eval_generate_repetition_penalty: float = 1.05,
        eval_generate_no_repeat_ngram_size: int = 0,
        log_loss_components: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.log_interval = log_interval
        self.eval_generate_n_batches = eval_generate_n_batches
        self.eval_generate_max_new_tokens = eval_generate_max_new_tokens
        self.eval_generate_top_p = eval_generate_top_p
        self.eval_generate_temperature = eval_generate_temperature
        self.eval_generate_repetition_penalty = eval_generate_repetition_penalty
        self.eval_generate_no_repeat_ngram_size = eval_generate_no_repeat_ngram_size
        self.log_loss_components = log_loss_components

        # Per-epoch accumulator for generated text -- consumed at on_epoch_end.
        self._eval_refs: list[str] = []
        self._eval_gens: list[str] = []
        self._eval_study_ids: list[str] = []

    # ------------------------------------------------------------------
    # Batch prep
    # ------------------------------------------------------------------
    def preprocess_batch(self, batch: dict, device: str = "cuda", train: bool = True) -> dict:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        # Ensure imaging tensors are float32 (encoders cast internally
        # under autocast if precision != 32).
        for k in ("ct_windows", "pet_windows"):
            if k in batch and batch[k].dtype != torch.float32:
                batch[k] = batch[k].to(torch.float32)
        return batch

    # ------------------------------------------------------------------
    # Step / loss
    # ------------------------------------------------------------------
    def step(
        self,
        model,
        batch: dict,
        batch_idx: int,
        epoch: int | None = None,
        split: str = "train",
        device: str = "cuda",
    ):
        batch = self.preprocess_batch(batch, device=device, train=(split == "train"))

        # Underlying HF model owns the LM-loss computation -- we just
        # forward the batch and read out the loss.
        with torch.amp.autocast(
            "cuda",
            dtype=self.amp_precision,
            enabled=self.amp_precision is not None,
        ):
            out = model(batch, split=split)

        loss = out["loss"]
        logging_dict = OrderedDict()
        logging_dict["lm_loss"] = loss.detach()
        try:
            with torch.no_grad():
                logging_dict["ppl"] = torch.exp(loss.detach().float().clamp(max=20.0))
        except Exception:
            pass

        # Predictions kept for downstream gathering (light: ids + study).
        predictions_dict: dict[str, Any] = {}
        if "study_id" in batch and not isinstance(batch["study_id"], torch.Tensor):
            predictions_dict["study_id"] = list(batch["study_id"])

        return loss, logging_dict, predictions_dict

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train_one_epoch(
        self,
        model: torch.nn.Module,
        dataloader,
        optimizer,
        device,
        epoch: int,
        loss_scaler,
        lr_scheduler,
        args,
        log_interval: int | None = None,
        clip_grad=None,
        log_loss_components: bool = False,
    ) -> None:
        model.train()
        # Encoders are frozen but their .eval() state matters for any norm
        # layers; keep them in eval explicitly.
        if hasattr(model, "module"):
            inner = model.module
        else:
            inner = model
        if hasattr(inner, "dual_stream_encoders"):
            inner.dual_stream_encoders.eval()

        if log_interval is None:
            log_interval = self.log_interval

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("LM loss", ":.4e")
        lr = AverageMeter("lr", ":.4e", summary_type=Summary.NONE)
        max_mem = AverageMeter("Max mem (MiB)", ":.0f", summary_type=Summary.NONE)

        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, lr, losses, max_mem],
            prefix=f"Epoch: [{epoch}]",
        )
        end = time.time()
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch} Training", disable=not get_is_master())
        ):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                max_mem.update(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if batch_idx == self.limit_num_batches:
                break
            if batch is None:
                continue  # ignore_None_collate may have dropped the whole batch

            if (lr_scheduler is not None) and (batch_idx % self.accum_iter == 0):
                lr_scheduler.adjust_learning_rate(batch_idx / len(dataloader) + epoch)

            with torch.amp.autocast(
                "cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None
            ):
                loss, logging_dict, _ = self.step(
                    model, batch, batch_idx, epoch=epoch, split="train", device=device
                )

            loss = loss / self.accum_iter
            loss_scaler(
                loss,
                optimizer,
                parameters=[p for p in model.parameters() if p.requires_grad],
                clip_grad=clip_grad,
                create_graph=False,
                need_update=(batch_idx + 1) % self.accum_iter == 0,
            )

            if (batch_idx + 1) % self.accum_iter == 0:
                optimizer.zero_grad()
                self.global_step += 1

            loss_value = loss.item() * self.accum_iter
            # Batch size from any tensor key we have.
            bs = batch.get("ct_windows", batch.get("report_token_ids")).size(0)
            losses.update(loss_value, bs)
            lr_value = optimizer.param_groups[0]["lr"]
            lr.update(lr_value)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_interval == 0:
                if get_is_master() and not self.args.main.disable_wandb:
                    wandb.log(
                        {
                            "train/lm_loss": loss_value,
                            "train/ppl": float(torch.exp(torch.tensor(loss_value).clamp(max=20.0))),
                            "lr": lr_value,
                            **_memory_metrics(),
                        },
                        step=self.global_step,
                    )
                progress.display(batch_idx + 1, tqdm_write=True)
                if log_loss_components and get_is_master() and not self.args.main.disable_wandb:
                    for k, v in logging_dict.items():
                        wandb.log({f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v},
                                  step=self.global_step)

        # No epoch-level gathering of predictions for training (saves memory).
        return None

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    def evaluate(
        self,
        model,
        dataloader,
        device,
        epoch: int | None = None,
        split: str = "val",
        gather_predictions: bool = False,
        log_loss_components: bool = False,
        ckpt_dir: str | None = None,
    ) -> dict:
        model.eval()
        inner = model.module if hasattr(model, "module") else model

        # Reset per-evaluation accumulators.
        self._eval_refs = []
        self._eval_gens = []
        self._eval_study_ids = []

        total_loss = 0.0
        total_count = 0
        desc = "Evaluation" if split == "val" else "Testing"

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch} {desc}" if epoch else desc,
                 disable=not get_is_master())
        ):
            if batch_idx == self.limit_num_batches:
                break
            if batch is None:
                continue

            torch.cuda.empty_cache()
            batch = self.preprocess_batch(batch, device=device, train=False)

            # --- LM loss pass ---
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                with torch.no_grad():
                    out = inner(batch, split=split)
            loss = out["loss"].detach().float().item()
            bs = batch.get("ct_windows", batch.get("report_token_ids")).size(0)
            total_loss += loss * bs
            total_count += bs

            # --- Generation pass on a few batches ---
            if batch_idx < self.eval_generate_n_batches:
                try:
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        gen_texts = inner.generate(
                            batch,
                            max_new_tokens=self.eval_generate_max_new_tokens,
                            top_p=self.eval_generate_top_p,
                            temperature=self.eval_generate_temperature,
                            repetition_penalty=self.eval_generate_repetition_penalty,
                            no_repeat_ngram_size=self.eval_generate_no_repeat_ngram_size,
                        )
                except Exception as e:
                    logger.warning(f"generate() raised on batch {batch_idx}: {e}")
                    gen_texts = ["[generation_failed]"] * bs

                refs = list(batch["report_text"]) if "report_text" in batch else [""] * bs
                study_ids = list(batch["study_id"]) if "study_id" in batch else [str(i) for i in range(bs)]
                self._eval_refs.extend(refs)
                self._eval_gens.extend(gen_texts)
                self._eval_study_ids.extend(study_ids)

            torch.cuda.empty_cache()

        # Aggregate metrics.
        avg_loss = total_loss / max(1, total_count)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss).clamp(max=20.0)))
        nlg = _compute_nlg_metrics(self._eval_refs, self._eval_gens)

        epoch_metrics = {
            f"{split}/lm_loss": avg_loss,
            f"{split}/ppl": avg_ppl,
        }
        for k, v in nlg.items():
            epoch_metrics[f"{split}/{k}"] = v

        if get_is_master():
            logger.info(f"=== Epoch {epoch} {split} ===")
            for k, v in epoch_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            if not self.args.main.disable_wandb:
                wandb_payload = {**epoch_metrics, **_memory_metrics()}
                wandb.log(wandb_payload, step=self.global_step)

                # Sample table (predicted vs reference) -- first 8 rows.
                if self._eval_gens:
                    table = wandb.Table(columns=["study_id", "reference", "generated"])
                    for sid, ref, gen in list(
                        zip(self._eval_study_ids, self._eval_refs, self._eval_gens)
                    )[:8]:
                        table.add_data(sid, ref, gen)
                    wandb.log({f"{split}/samples": table}, step=self.global_step)

            print_mem(f"{split} ep={epoch}")

        return epoch_metrics
