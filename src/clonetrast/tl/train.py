"""Training loop for CloneTrast (SupCon-style contrastive) model."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Sampler, TensorDataset

from ..model import (
    CloneDataset,
    CloneSizeModel,
    CloneTrastModel,
    contrastive_loss_clone,
)
from ..pp.normalize import get_expression_matrix
from .metrics import clone_purity

VALIDATION_METRIC_COLUMN = "clone_purity"


def _build_hidden_dims_from_arch(
    *,
    depth: int,
    width: int,
    width_decay: float,
    min_hidden_dim: int,
    embedding_dim: int,
) -> list[int]:
    """Build hidden layer widths from architecture controls.

    The first hidden layer starts at ``width`` and each subsequent layer is scaled
    by ``width_decay`` (geometric progression), then clamped by ``min_hidden_dim``.
    """
    if depth <= 0:
        raise ValueError(f"depth must be >= 1, got {depth}.")
    if width <= 0:
        raise ValueError(f"width must be >= 1, got {width}.")
    if width_decay <= 0:
        raise ValueError(f"width_decay must be > 0, got {width_decay}.")
    if min_hidden_dim <= 0:
        raise ValueError(f"min_hidden_dim must be >= 1, got {min_hidden_dim}.")
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}.")

    min_allowed_hidden_dim = max(min_hidden_dim, embedding_dim)
    dims: list[int] = []
    for i in range(depth):
        dim = int(round(width * math.pow(width_decay, i)))
        dims.append(max(min_allowed_hidden_dim, dim))
    return dims


def _validate_dropout(dropout: float) -> float:
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
    return dropout


def _validate_weight_decay(weight_decay: float) -> float:
    weight_decay = float(weight_decay)
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
    return weight_decay


def _save_checkpoint(obj: dict, path: Path) -> None:
    """Save checkpoint via buffer then temp file and replace; avoids Windows error 32 (file in use).

    PyTorch's zipfile writer can open by path internally and trigger sharing violations on Windows.
    We use legacy serialization into a BytesIO so PyTorch never touches the filesystem, then write
    the bytes to a unique temp file and replace the target.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    # Legacy format writes only to the buffer; zip format may open by path on some setups
    torch.save(obj, buffer, _use_new_zipfile_serialization=False)
    # Unique temp name so multiple checkpoints (e.g. model + size) don't collide
    tmp_path = path.parent / (path.stem + "_" + uuid.uuid4().hex[:8] + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(buffer.getvalue())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _clone_level_roc_auc(clone_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC on per-clone mean predictions with median-split true clone-size labels."""
    clone_ids = np.asarray(clone_ids).ravel()
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    if not (len(clone_ids) == len(y_true) == len(y_pred)):
        raise ValueError("clone_ids, y_true, and y_pred must have equal length.")

    order = np.argsort(clone_ids.astype(str), kind="mergesort")
    sorted_clone = clone_ids[order].astype(str)
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    unique_clones, starts = np.unique(sorted_clone, return_index=True)

    mean_true = np.empty(len(unique_clones), dtype=np.float32)
    mean_pred = np.empty(len(unique_clones), dtype=np.float32)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(sorted_clone)
        mean_true[i] = float(np.mean(sorted_true[start:end]))
        mean_pred[i] = float(np.mean(sorted_pred[start:end]))

    if len(mean_true) < 2:
        return float("nan")
    threshold = float(np.median(mean_true))
    labels = mean_true > threshold
    n_pos = int(np.sum(labels))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = mean_pred.astype(np.float64, copy=False)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    n = sorted_scores.size
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0  # 1-based average rank for ties
        ranks[i:j] = avg_rank
        i = j
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_pos_ranks = float(np.sum(full_ranks[labels]))
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Return average ranks (1-based) with tie handling."""
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty_like(sorted_x, dtype=np.float64)
    i = 0
    n = sorted_x.size
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty_like(ranks_sorted, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _clone_level_spearman(
    clone_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Spearman correlation on per-clone mean true/predicted clone-size values."""
    clone_ids = np.asarray(clone_ids).ravel()
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    if not (len(clone_ids) == len(y_true) == len(y_pred)):
        raise ValueError("clone_ids, y_true, and y_pred must have equal length.")

    order = np.argsort(clone_ids.astype(str), kind="mergesort")
    sorted_clone = clone_ids[order].astype(str)
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    unique_clones, starts = np.unique(sorted_clone, return_index=True)

    mean_true = np.empty(len(unique_clones), dtype=np.float64)
    mean_pred = np.empty(len(unique_clones), dtype=np.float64)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(sorted_clone)
        mean_true[i] = float(np.mean(sorted_true[start:end]))
        mean_pred[i] = float(np.mean(sorted_pred[start:end]))

    if mean_true.size < 2:
        return float("nan")

    r_true = _average_ranks(mean_true)
    r_pred = _average_ranks(mean_pred)
    std_true = float(np.std(r_true))
    std_pred = float(np.std(r_pred))
    if std_true == 0.0 or std_pred == 0.0:
        return float("nan")
    corr = np.corrcoef(r_true, r_pred)[0, 1]
    return float(corr)


def _cell_level_wasserstein_log(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """1D Wasserstein-1 between cell-level log-scaled clone-size distributions.

    Expects ``y_true`` / ``y_pred`` already in log clone-size space (same as
    ``obs['clone_id_size']`` / :class:`~clonetrast.model.CloneSizeModel` outputs).
    Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    if y_true.size == 0:
        return float("nan")
    return float(wasserstein_distance(y_true, y_pred))


def _clone_level_wasserstein_log(
    clone_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """1D Wasserstein-1 between clone-level mean log-scaled clone-size distributions.

    Aggregates cells to per-clone mean true / predicted values, then computes
    Wasserstein-1 between those two empirical distributions. Expects ``y_true`` /
    ``y_pred`` already in log clone-size space (same as ``obs['clone_id_size']`` /
    :class:`~clonetrast.model.CloneSizeModel` outputs). Lower is better.
    """
    clone_ids = np.asarray(clone_ids).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if not (len(clone_ids) == len(y_true) == len(y_pred)):
        raise ValueError("clone_ids, y_true, and y_pred must have equal length.")
    if y_true.size == 0:
        return float("nan")

    order = np.argsort(clone_ids.astype(str), kind="mergesort")
    sorted_clone = clone_ids[order].astype(str)
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    unique_clones, starts = np.unique(sorted_clone, return_index=True)

    mean_true = np.empty(len(unique_clones), dtype=np.float64)
    mean_pred = np.empty(len(unique_clones), dtype=np.float64)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(sorted_clone)
        mean_true[i] = float(np.mean(sorted_true[start:end]))
        mean_pred[i] = float(np.mean(sorted_pred[start:end]))

    if mean_true.size == 0:
        return float("nan")
    return float(wasserstein_distance(mean_true, mean_pred))


def _combined_wasserstein_log(
    clone_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(cell_w, clone_w, cell_w + clone_w)`` on log clone sizes."""
    cell_w = _cell_level_wasserstein_log(y_true, y_pred)
    clone_w = _clone_level_wasserstein_log(clone_ids, y_true, y_pred)
    return cell_w, clone_w, float(cell_w + clone_w)


_SIZE_SELECTION_METRICS = frozenset({"wasserstein", "wasserstein_spearman"})


def _size_selection_objective(
    wasserstein_sum: float,
    spearman: float | None,
    selection_metric: str,
) -> float:
    """Combine size metrics into the scalar used for Optuna / checkpointing / early stopping.

    ``"wasserstein"``: cell W1 + clone W1. ``"wasserstein_spearman"``: cell W1 +
    clone W1 + (1 - clone-level Spearman); an undefined Spearman counts as 0.
    """
    if selection_metric == "wasserstein":
        return float(wasserstein_sum)
    rho = 0.0 if spearman is None or np.isnan(spearman) else float(spearman)
    return float(wasserstein_sum) + (1.0 - rho)


_SIZE_TRAINING_LOSSES = frozenset({"mse", "mse_ranking", "clone_pearson"})


def _batch_clone_means(
    clone_ids: np.ndarray,
    pred: torch.Tensor,
    true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-clone mean prediction and target within a batch (differentiable)."""
    clone_ids = np.asarray(clone_ids).astype(str).ravel()
    unique_clones, inverse = np.unique(clone_ids, return_inverse=True)
    if unique_clones.size < 2:
        return None

    inv = torch.as_tensor(inverse, device=pred.device, dtype=torch.long)
    n_clones = int(unique_clones.size)
    counts = torch.bincount(inv, minlength=n_clones).to(dtype=pred.dtype)
    sum_pred = pred.new_zeros(n_clones).scatter_add_(0, inv, pred)
    sum_true = pred.new_zeros(n_clones).scatter_add_(0, inv, true)
    counts_safe = counts.clamp_min(1.0)
    return sum_pred / counts_safe, sum_true / counts_safe


def _differentiable_pearson_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return 1 - Pearson correlation (minimize to maximize correlation)."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = x_c.pow(2).mean().sqrt() * y_c.pow(2).mean().sqrt()
    corr = (x_c * y_c).mean() / denom.clamp_min(eps)
    return 1.0 - corr


def _pairwise_ranking_loss(mean_pred: torch.Tensor, mean_true: torch.Tensor) -> torch.Tensor:
    """Logistic pairwise ranking loss on clone-level means (ties skipped)."""
    diff_true = mean_true.unsqueeze(1) - mean_true.unsqueeze(0)
    diff_pred = mean_pred.unsqueeze(1) - mean_pred.unsqueeze(0)
    mask = diff_true > 0
    if not bool(mask.any().item()):
        return mean_pred.new_tensor(0.0)
    return torch.nn.functional.softplus(-diff_pred[mask]).mean()


class _SizeLossBreakdown(NamedTuple):
    loss: torch.Tensor
    mse: torch.Tensor
    rank_loss: torch.Tensor | None = None
    pearson_loss: torch.Tensor | None = None


def _compute_size_training_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    clone_ids: np.ndarray,
    *,
    loss_mode: str,
    ranking_loss_weight: float = 1.0,
) -> _SizeLossBreakdown:
    """Training loss for the clone-size model."""
    mse = torch.nn.functional.mse_loss(pred, true)
    if loss_mode == "mse":
        return _SizeLossBreakdown(loss=mse, mse=mse)

    aggregates = _batch_clone_means(clone_ids, pred, true)
    if aggregates is None:
        return _SizeLossBreakdown(loss=mse, mse=mse)

    mean_pred, mean_true = aggregates
    if loss_mode == "clone_pearson":
        pearson_loss = _differentiable_pearson_loss(mean_pred, mean_true)
        return _SizeLossBreakdown(
            loss=pearson_loss,
            mse=mse,
            pearson_loss=pearson_loss,
        )
    if loss_mode == "mse_ranking":
        rank_loss = _pairwise_ranking_loss(mean_pred, mean_true)
        loss = mse + float(ranking_loss_weight) * rank_loss
        return _SizeLossBreakdown(
            loss=loss,
            mse=mse,
            rank_loss=rank_loss,
        )

    raise ValueError(
        f"Unsupported size_training_loss={loss_mode!r}. "
        f"Use one of: {sorted(_SIZE_TRAINING_LOSSES)}."
    )


class PatientWiseBatchSampler(Sampler[list[int]]):
    """BatchSampler that yields batches of indices where each batch has cells from up to k patients.

    Each epoch uses every cell exactly once. Batches have at most batch_size cells; the last
    batch(es) for a group may be smaller when the selected patients have fewer cells together.
    This lets the model separate clone_ids within patients and between patients.
    """

    def __init__(
        self,
        patient_ids: np.ndarray,
        batch_size: int,
        patients_per_batch: int = 2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.patient_ids = np.asarray(patient_ids).ravel()
        self.n_cells = len(self.patient_ids)
        self.batch_size = batch_size
        self.patients_per_batch = max(1, patients_per_batch)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __iter__(self):
        """Yield batches of cell indices mixing up to ``patients_per_batch`` patients."""
        # Map patient_id -> set of cell indices (for fast removal)
        patient_to_indices: dict[Any, set[int]] = {}
        for idx in range(self.n_cells):
            pid = self.patient_ids[idx]
            if pid not in patient_to_indices:
                patient_to_indices[pid] = set()
            patient_to_indices[pid].add(idx)

        remaining = {p: set(indices) for p, indices in patient_to_indices.items()}
        patients_with_cells = [p for p in remaining if remaining[p]]

        # BatchNorm requires >1 sample per channel; never yield a batch of size 1
        min_batch_size = 2
        while patients_with_cells:
            k = min(self.patients_per_batch, len(patients_with_cells))
            selected = self.rng.choice(patients_with_cells, size=k, replace=False)
            if np.isscalar(selected):
                selected = [selected]
            else:
                selected = list(selected)
            pool_set = set()
            for p in selected:
                pool_set |= remaining[p]
            pool = list(pool_set)
            # If pool has < min_batch_size cells, add more patients until we have enough or none left
            while len(pool) < min_batch_size and len(selected) < len(patients_with_cells):
                extra = [p for p in patients_with_cells if p not in selected]
                n_need = min(1, len(extra))
                add = self.rng.choice(extra, size=n_need, replace=False)
                if np.isscalar(add):
                    add = [add]
                selected.extend(add)
                for p in add:
                    pool_set |= remaining[p]
                pool = list(pool_set)
            if len(pool) < min_batch_size:
                # Only 1 cell left in dataset; remove it so loop ends, skip this batch
                for idx in pool:
                    p = self.patient_ids[idx]
                    remaining[p].discard(idx)
                patients_with_cells = [p for p in remaining if remaining[p]]
                continue
            self.rng.shuffle(pool)
            take = min(self.batch_size, len(pool))
            batch_indices = pool[:take]

            for idx in batch_indices:
                p = self.patient_ids[idx]
                remaining[p].discard(idx)

            patients_with_cells = [p for p in remaining if remaining[p]]
            yield batch_indices

    def __len__(self) -> int:
        """Return an approximate number of batches per epoch."""
        # Approximate: at least ceil(n_cells / batch_size) batches
        n_batches = 0
        patient_to_indices: dict[Any, set[int]] = {}
        for idx in range(self.n_cells):
            pid = self.patient_ids[idx]
            if pid not in patient_to_indices:
                patient_to_indices[pid] = set()
            patient_to_indices[pid].add(idx)
        remaining = {p: len(s) for p, s in patient_to_indices.items()}
        total = sum(remaining.values())
        while total > 0:
            n_batches += 1
            total -= min(self.batch_size, total)
        return n_batches


class CloneTrastTrainer:
    """Train the contrastive CloneTrast encoder; clone-size regression uses :func:`train_size_model`."""

    def __init__(
        self,
        adata_train,
        adata_val=None,
        clone_id_key: str = "clone_id",
        hidden_dims: list[int] | None = None,
        encoder_depth: int | None = None,
        encoder_width: int | None = None,
        encoder_width_decay: float = 0.5,
        min_hidden_dim: int = 32,
        embedding_dim: int = 128,
        projection_dim: int = 64,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
        temperature: float = 0.25,
        contrastive_loss_variant: str = "sup_out",
        batch_size: int = 256,
        epochs: int = 100,
        lr: float = 1e-3,
        scheduler: str = "none",
        scheduler_t_max: int | None = None,
        scheduler_eta_min: float = 1e-6,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        device: str | None = None,
        out_dir: str | Path = "clonetrast_out",
        seed: int = 0,
        patient_wise_batching: bool = False,
        patients_per_batch: int = 2,
        patient_key: str = "patient",
        report_metric: Callable[[int, float], None] | None = None,
        early_stopping: bool = False,
        patience: int = 25,
        min_delta: float = 1e-4,
        warmup_epochs: int = 0,
        metric_min_clone_size: int = 2,
        metric_k: int = 10,
        metric_weight_by_cells: bool = True,
        compute_train_metric: bool = False,
    ) -> None:
        """Set up trainer.

        Parameters
        ----------
        adata_train
            AnnData with prepared expression (e.g. after prepare_adata) and obs[clone_id_key].
        adata_val
            Optional validation AnnData. When provided, validation contrastive loss is
            logged each epoch, while checkpointing, early stopping, and ``report_metric``
            use validation :func:`~clonetrast.tl.metrics.clone_purity` (higher is better).
        clone_id_key
            Key in adata.obs for clone identifiers.
        hidden_dims
            Encoder hidden layer sizes. Default [512, 256].
            If provided, takes precedence over encoder_depth/encoder_width controls.
        encoder_depth
            Number of hidden layers when hidden_dims is not provided.
            Useful for architecture search (e.g. Optuna).
        encoder_width
            First hidden layer width when hidden_dims is not provided.
            Useful for architecture search (e.g. Optuna).
        encoder_width_decay
            Geometric decay factor for hidden widths when generated from
            encoder_depth/encoder_width. For example: width=512, decay=0.5,
            depth=3 -> [512, 256, 128].
        min_hidden_dim
            Minimum hidden width when generated from architecture controls.
        embedding_dim
            Output embedding dimension of the expression encoder.
        projection_dim
            Projection head output dimension (for contrastive loss).
        dropout
            Dropout rate applied after each encoder MLP block (default 0.1).
        weight_decay
            L2 penalty coefficient for AdamW (default 0.0).
        temperature
            Temperature τ for the SupCon loss. Lower values (e.g. 0.25-0.35) sharpen
            separation between clones; default 0.25.
        contrastive_loss_variant
            Supervised contrastive loss variant from Khosla et al. 2020:
            ``"sup_out"`` for Eq. (2) (default) or ``"sup_in"`` for Eq. (3).
        batch_size
            Training batch size.
        epochs
            Number of training epochs.
        lr
            Learning rate for main contrastive model.
        scheduler
            Learning-rate scheduler for the main model optimizer.
            Options: "none", "cosine", "plateau".
        scheduler_t_max
            For cosine scheduler: cycle length in epochs. If None, uses total epochs.
        scheduler_eta_min
            For cosine scheduler: minimum learning rate.
        scheduler_patience
            For plateau scheduler: epochs with no improvement before LR reduction.
        scheduler_factor
            For plateau scheduler: LR is multiplied by this factor when triggered.
        device
            'cuda', 'cpu', or None (auto-detect).
        out_dir
            Directory to save checkpoint and config.
        seed
            Random seed.
        patient_wise_batching
            If True, each batch contains cells from up to patients_per_batch (random) patients so the
            model learns to separate clone_ids within and between patients. Requires patient_key in adata.obs.
        patients_per_batch
            When patient_wise_batching is True, number of patients to mix per batch (default 2).
            Batches are formed by sampling this many patients and then sampling cells up to batch_size.
        patient_key
            Key in adata.obs for patient identifiers. Used when patient_wise_batching is True.
        report_metric
            Optional callback (step: int, value: float) -> None. If provided, called each epoch
            with the optimization metric. With validation data this is weighted
            :func:`~clonetrast.tl.metrics.clone_purity` on the joint validation embedding
            (higher is better). Without validation data it falls back to training loss
            (lower is better). Use with Optuna: ``lambda step, value: trial.report(value, step)``.
        early_stopping
            If True and adata_val is provided, stop training when validation clone purity
            does not improve.
        patience
            Number of consecutive non-improving validation-metric epochs tolerated before early stop.
        min_delta
            Minimum required validation clone-purity improvement to reset early-stopping patience.
        warmup_epochs
            Number of initial epochs before early-stopping checks start.
        metric_min_clone_size
            Minimum clone size included in :func:`~clonetrast.tl.metrics.clone_purity`
            (default 2).
        metric_k
            *k* for :func:`~clonetrast.tl.metrics.clone_purity` (default 10).
        metric_weight_by_cells
            If True (default), overall clone purity weights each patient by cell count;
            if False, each patient counts equally (when ``patient_key`` is present).
        compute_train_metric
            If True, each epoch also encodes the full training set and computes
            clone purity on train embeddings (expensive for large data).

        """
        self.report_metric = report_metric
        self.adata_train = adata_train
        self.adata_val = adata_val
        self.clone_id_key = clone_id_key
        self.patient_wise_batching = patient_wise_batching
        self.patients_per_batch = max(1, patients_per_batch)
        self.patient_key = patient_key
        self.encoder_depth = encoder_depth
        self.encoder_width = encoder_width
        self.encoder_width_decay = float(encoder_width_decay)
        self.min_hidden_dim = int(max(1, min_hidden_dim))
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = projection_dim
        self.dropout = _validate_dropout(dropout)
        self.weight_decay = _validate_weight_decay(weight_decay)
        min_allowed_hidden_dim = max(self.min_hidden_dim, self.embedding_dim)
        self.temperature = temperature
        if contrastive_loss_variant not in ("sup_out", "sup_in"):
            raise ValueError(
                'contrastive_loss_variant must be "sup_out" or "sup_in", '
                f"got {contrastive_loss_variant!r}"
            )
        self.contrastive_loss_variant = contrastive_loss_variant
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.scheduler = str(scheduler).lower()
        self.scheduler_t_max = scheduler_t_max
        self.scheduler_eta_min = float(scheduler_eta_min)
        self.scheduler_patience = int(max(1, scheduler_patience))
        self.scheduler_factor = float(scheduler_factor)
        self.out_dir = Path(out_dir)
        self.seed = seed
        self.early_stopping = bool(early_stopping)
        self.patience = int(max(1, patience))
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.metric_min_clone_size = int(max(2, metric_min_clone_size))
        self.metric_k = int(max(1, metric_k))
        self.metric_weight_by_cells = bool(metric_weight_by_cells)
        self.compute_train_metric = bool(compute_train_metric)
        if hidden_dims is not None:
            self.hidden_dims = [max(int(d), min_allowed_hidden_dim) for d in hidden_dims]
        elif self.encoder_depth is not None and self.encoder_width is not None:
            self.hidden_dims = _build_hidden_dims_from_arch(
                depth=int(self.encoder_depth),
                width=int(self.encoder_width),
                width_decay=self.encoder_width_decay,
                min_hidden_dim=self.min_hidden_dim,
                embedding_dim=int(self.embedding_dim),
            )
        else:
            self.hidden_dims = [max(512, min_allowed_hidden_dim), max(256, min_allowed_hidden_dim)]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._gene_order: np.ndarray | None = None
        self._model: CloneTrastModel | None = None

        if self.scheduler not in {"none", "cosine", "plateau"}:
            raise ValueError(
                f"Unsupported scheduler={scheduler!r}. "
                "Use one of: 'none', 'cosine', 'plateau'."
            )

    def _get_data(self, adata) -> tuple[np.ndarray, np.ndarray]:
        X = get_expression_matrix(adata)
        if self.clone_id_key not in adata.obs:
            raise ValueError(
                "Training requires clone_id_key in adata.obs. "
                f"Got clone_id_key={self.clone_id_key!r}."
            )
        clone_ids = np.asarray(adata.obs[self.clone_id_key].astype(str))
        return X, clone_ids

    def train(self) -> CloneTrastModel:
        """Run training and save checkpoint and config. Returns the trained model."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.out_dir.mkdir(parents=True, exist_ok=True)

        X, clone_ids = self._get_data(self.adata_train)
        n_genes = X.shape[1]
        self._gene_order = np.arange(n_genes)
        train_patient_ids = None
        if self.patient_key in self.adata_train.obs:
            train_patient_ids = np.asarray(self.adata_train.obs[self.patient_key].astype(str)).ravel()

        dataset = CloneDataset(X, clone_ids)
        if self.patient_wise_batching:
            if self.patient_key not in self.adata_train.obs:
                raise ValueError(
                    f"patient_wise_batching=True requires adata.obs['{self.patient_key}']. "
                    f"Available keys: {list(self.adata_train.obs.keys())[:20]}..."
                )
            patient_ids = np.asarray(self.adata_train.obs[self.patient_key].astype(str)).ravel()
            if len(patient_ids) != len(clone_ids):
                raise ValueError(
                    f"patient_key '{self.patient_key}' length {len(patient_ids)} != number of cells {len(clone_ids)}."
                )
            batch_sampler = PatientWiseBatchSampler(
                patient_ids,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                rng=np.random.default_rng(self.seed),
            )
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )

        val_loader = None
        val_patient_ids = None
        if self.adata_val is not None:
            X_val, clone_ids_val = self._get_data(self.adata_val)
            if X_val.shape[1] != n_genes:
                raise ValueError(
                    f"Validation gene dimension ({X_val.shape[1]}) does not match training ({n_genes})."
                )
            val_dataset = CloneDataset(X_val, clone_ids_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )
            if self.patient_key not in self.adata_val.obs:
                raise ValueError(
                    f"Validation metric requires adata_val.obs['{self.patient_key}'] for patient weighting. "
                    f"Available keys: {list(self.adata_val.obs.keys())[:20]}..."
                )
            val_patient_ids = np.asarray(self.adata_val.obs[self.patient_key].astype(str)).ravel()

        if self.patient_wise_batching:
            print(f"Using patient-wise batching (k={self.patients_per_batch} patients per batch).")

        # Create main contrastive model
        model = CloneTrastModel(
            n_genes=n_genes,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            projection_dim=self.projection_dim,
            dropout=self.dropout,
        ).to(self.device)
        self._model = model
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = None
        if self.scheduler == "cosine":
            t_max = int(self.scheduler_t_max) if self.scheduler_t_max is not None else int(max(1, self.epochs))
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(1, t_max),
                eta_min=self.scheduler_eta_min,
            )
        elif self.scheduler == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="max" if self.adata_val is not None else "min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )

        # Track best losses to save only the best models
        best_loss = float("inf")
        best_epoch = -1
        # Best checkpoint: maximize validation clone purity when val set exists; else minimize train loss
        best_metric = float("-inf") if self.adata_val is not None else float("inf")
        ckpt_path = self.out_dir / "model.pt"
        loss_history_path = self.out_dir / "loss_history.csv"
        no_improve_epochs = 0
        best_es_metric = float("-inf") if self.adata_val is not None else float("inf")
        metric_column_name = VALIDATION_METRIC_COLUMN
        with open(loss_history_path, "w", newline="") as f:
            writer = csv.writer(f)
            if val_loader is not None:
                header = ["epoch", "contrastive_loss"]
                if self.compute_train_metric:
                    header.append(f"train_{metric_column_name}")
                header.extend(["val_contrastive_loss", f"val_{metric_column_name}"])
                writer.writerow(header)
            else:
                header = ["epoch", "contrastive_loss"]
                if self.compute_train_metric:
                    header.append(f"train_{metric_column_name}")
                writer.writerow(header)

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch_x = batch[0].to(self.device)
                batch_c = batch[1].to(self.device)

                # Forward: contrastive loss on L2-normalized projection head outputs.
                _h, z = model(batch_x)
                loss = contrastive_loss_clone(
                    z,
                    batch_c,
                    temperature=self.temperature,
                    variant=self.contrastive_loss_variant,
                )

                # Backward: main model
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

                n_batches += 1

            mean_loss = total_loss / max(n_batches, 1)
            mean_val_loss = None
            val_metric = None
            val_metric_value = None
            train_metric_value = None
            if train_patient_ids is not None and self.compute_train_metric:
                with torch.no_grad():
                    train_embeddings = []
                    for start in range(0, X.shape[0], self.batch_size):
                        end = min(start + self.batch_size, X.shape[0])
                        batch_x = torch.from_numpy(X[start:end]).to(self.device)
                        h = model.encode(batch_x)
                        train_embeddings.append(h.cpu().numpy())
                    train_embeddings_all = np.vstack(train_embeddings)
                train_metric_value = clone_purity(
                    train_embeddings_all,
                    clone_ids,
                    k=self.metric_k,
                    n_jobs=1,
                    min_clone_size=self.metric_min_clone_size,
                    patient_ids=train_patient_ids,
                    weight_by_cells=self.metric_weight_by_cells,
                )
            if val_loader is not None:
                model.eval()
                total_val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_x = batch[0].to(self.device)
                        batch_c = batch[1].to(self.device)

                        _h, z = model(batch_x)
                        loss = contrastive_loss_clone(
                            z,
                            batch_c,
                            temperature=self.temperature,
                            variant=self.contrastive_loss_variant,
                        )
                        total_val_loss += loss.item()
                        n_val_batches += 1

                mean_val_loss = total_val_loss / max(n_val_batches, 1)

                # Joint validation embedding (all validation cells); KNN restricted per patient.
                with torch.no_grad():
                    val_embeddings = []
                    for start in range(0, X_val.shape[0], self.batch_size):
                        end = min(start + self.batch_size, X_val.shape[0])
                        batch_x = torch.from_numpy(X_val[start:end]).to(self.device)
                        h = model.encode(batch_x)
                        val_embeddings.append(h.cpu().numpy())
                    val_embeddings_all = np.vstack(val_embeddings)
                val_metric_value = clone_purity(
                    val_embeddings_all,
                    clone_ids_val,
                    k=self.metric_k,
                    n_jobs=1,
                    min_clone_size=self.metric_min_clone_size,
                    patient_ids=val_patient_ids,
                    weight_by_cells=self.metric_weight_by_cells,
                )
                if val_metric_value is not None and not np.isnan(val_metric_value):
                    val_metric = float(val_metric_value)
                else:
                    val_metric = float("-inf")

            # Optimization metric: validation clone purity (max) if val set exists; else train loss (min).
            selection_higher_is_better = val_metric is not None
            if val_metric is not None:
                metric_for_report = float(val_metric)
            else:
                metric_for_report = mean_loss
            if lr_scheduler is not None:
                if self.scheduler == "plateau":
                    lr_scheduler.step(metric_for_report)
                else:
                    lr_scheduler.step()
            if self.report_metric is not None:
                self.report_metric(epoch, metric_for_report)

            with open(loss_history_path, "a", newline="") as f:
                writer = csv.writer(f)
                if val_loader is not None:
                    row = [epoch + 1, mean_loss]
                    if self.compute_train_metric:
                        row.append(train_metric_value if train_metric_value is not None else "")
                    row.extend(
                        [
                            mean_val_loss if mean_val_loss is not None else "",
                            val_metric_value if val_metric_value is not None else "",
                        ]
                    )
                    writer.writerow(row)
                else:
                    row = [epoch + 1, mean_loss]
                    if self.compute_train_metric:
                        row.append(train_metric_value if train_metric_value is not None else "")
                    writer.writerow(row)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                pw_tag = f" [patient-wise, k={self.patients_per_batch}]" if self.patient_wise_batching else ""
                if mean_val_loss is not None:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs}{pw_tag}  "
                        f"train_loss = {mean_loss:.4f}  val_loss = {mean_val_loss:.4f}  "
                        f"val_metric = {metric_for_report:.6f}"
                    )
                else:
                    print(f"Epoch {epoch + 1}/{self.epochs}{pw_tag}  loss = {mean_loss:.4f}")

            # Save models if these are the best losses so far
            checkpoint_loss = mean_val_loss if mean_val_loss is not None else mean_loss
            improved = (
                metric_for_report > (best_metric + self.min_delta)
                if selection_higher_is_better
                else metric_for_report < (best_metric - self.min_delta)
            )
            if improved:
                best_metric = metric_for_report
                best_loss = checkpoint_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                _save_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "n_genes": n_genes,
                        "hidden_dims": self.hidden_dims,
                        "embedding_dim": self.embedding_dim,
                        "projection_dim": self.projection_dim,
                        "best_loss": best_loss,
                        "best_metric": best_metric,
                        "epoch": epoch + 1,
                    },
                    ckpt_path,
                )
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(
                        f"  -> Saved best contrastive model (metric = {best_metric:.6f}, "
                        f"val_loss = {best_loss:.4f})"
                    )

            if val_metric is not None:
                if val_metric > (best_es_metric + self.min_delta):
                    best_es_metric = val_metric
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            if (
                self.early_stopping
                and val_metric is not None
                and (epoch + 1) >= self.warmup_epochs
                and no_improve_epochs >= self.patience
            ):
                print(
                    f"Early stopping at epoch {epoch + 1}: no validation metric improvement "
                    f"for {self.patience} epoch(s) (min_delta={self.min_delta:g})."
                )
                break

        print(f"Training complete. Best loss: {best_loss:.4f}")
        if best_epoch > 0:
            print(f"Best checkpoint saved from epoch {best_epoch} (metric = {best_metric:.6f}).")

        # Load the best model to ensure returned model matches saved checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        self._model = model

        config = {
            "n_genes": n_genes,
            "hidden_dims": self.hidden_dims,
            "encoder_depth": self.encoder_depth,
            "encoder_width": self.encoder_width,
            "encoder_width_decay": self.encoder_width_decay,
            "min_hidden_dim": self.min_hidden_dim,
            "embedding_dim": self.embedding_dim,
            "projection_dim": self.projection_dim,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "contrastive_loss_variant": self.contrastive_loss_variant,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "scheduler": self.scheduler,
            "scheduler_t_max": self.scheduler_t_max,
            "scheduler_eta_min": self.scheduler_eta_min,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "clone_id_key": self.clone_id_key,
            "has_size_model": False,
            "patient_wise_batching": self.patient_wise_batching,
            "patients_per_batch": self.patients_per_batch,
            "patient_key": self.patient_key,
            "has_validation_data": self.adata_val is not None,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "warmup_epochs": self.warmup_epochs,
            "metric_name": f"val_{VALIDATION_METRIC_COLUMN}",
            "selection_metric": VALIDATION_METRIC_COLUMN,
            "metric_higher_is_better": True,
            "metric_min_clone_size": self.metric_min_clone_size,
            "metric_k": self.metric_k,
            "metric_weight_by_cells": self.metric_weight_by_cells,
            "compute_train_metric": self.compute_train_metric,
            "checkpoint_selection_metric": f"val_{VALIDATION_METRIC_COLUMN}",
            "early_stopping_metric": f"val_{VALIDATION_METRIC_COLUMN}",
        }
        with open(self.out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        # Save var_names so test data can be aligned to same genes
        var_names = list(self.adata_train.var_names)
        with open(self.out_dir / "gene_order.json", "w") as f:
            json.dump(var_names, f)
        print(f"Saved best model and config to {self.out_dir}")
        return model

    @property
    def model(self) -> CloneTrastModel | None:
        """Return the trained contrastive model, if any."""
        return self._model

    @property
    def gene_order(self) -> np.ndarray | None:
        """Return the training gene order used for alignment."""
        return self._gene_order


def train(
    adata,
    adata_val=None,
    clone_id_key: str = "clone_id",
    hidden_dims: list[int] | None = None,
    encoder_depth: int | None = None,
    encoder_width: int | None = None,
    encoder_width_decay: float = 0.5,
    min_hidden_dim: int = 32,
    scheduler: str = "none",
    scheduler_t_max: int | None = None,
    scheduler_eta_min: float = 1e-6,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    out_dir: str | Path = "clonetrast_out",
    patient_wise_batching: bool = False,
    patients_per_batch: int = 2,
    report_metric: Callable[[int, float], None] | None = None,
    early_stopping: bool = False,
    patience: int = 25,
    min_delta: float = 1e-4,
    warmup_epochs: int = 0,
    metric_min_clone_size: int = 2,
    metric_weight_by_cells: bool = True,
    compute_train_metric: bool = False,
    **kwargs: Any,
) -> CloneTrastModel:
    """Train CloneTrast on adata and return the model.

    Expects adata to be already prepared (e.g. via pp.prepare_adata).
    Size-model training is intentionally disabled in this function.
    Use ct.tl.train_size_model(...) for clone-size model training/optimization.
    For architecture search, either pass explicit hidden_dims or set
    encoder_depth + encoder_width (with encoder_width_decay/min_hidden_dim).
    Use patient_wise_batching=True and patients_per_batch=k for patient-wise batching with k patients per batch.
    If adata_val is provided, validation contrastive loss is logged each epoch, while
    reporting, checkpointing, and early stopping use validation clone purity
    (higher is better).
    If report_metric is provided, it is called each epoch with the optimization
    metric (validation clone purity when adata_val exists - higher is better;
    otherwise training loss - lower is better), e.g.
    ``lambda step, value: trial.report(value, step)`` for Optuna.
    Set ``compute_train_metric=True`` to also compute clone purity on the full
    training embedding each epoch (logged to ``loss_history.csv``; costly on large data).
    ``metric_weight_by_cells`` (default True) weights patients by cell count when aggregating clone purity.
    """
    trainer = CloneTrastTrainer(
        adata,
        adata_val=adata_val,
        clone_id_key=clone_id_key,
        hidden_dims=hidden_dims,
        encoder_depth=encoder_depth,
        encoder_width=encoder_width,
        encoder_width_decay=encoder_width_decay,
        min_hidden_dim=min_hidden_dim,
        scheduler=scheduler,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        out_dir=out_dir,
        patient_wise_batching=patient_wise_batching,
        patients_per_batch=patients_per_batch,
        report_metric=report_metric,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        warmup_epochs=warmup_epochs,
        metric_min_clone_size=metric_min_clone_size,
        metric_weight_by_cells=metric_weight_by_cells,
        compute_train_metric=compute_train_metric,
        **kwargs,
    )
    return trainer.train()


def train_size_model(
    adata_train,
    adata_val=None,
    *,
    clone_id_key: str = "clone_id",
    clone_id_size_key: str = "clone_id_size",
    hidden_dims: list[int] | None = None,
    encoder_depth: int | None = None,
    encoder_width: int | None = None,
    encoder_width_decay: float = 0.5,
    min_hidden_dim: int = 32,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    scheduler: str = "none",
    scheduler_t_max: int | None = None,
    scheduler_eta_min: float = 1e-6,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    device: str | None = None,
    out_dir: str | Path = "clonetrast_out_size",
    seed: int = 0,
    patient_wise_batching: bool = False,
    patients_per_batch: int = 2,
    patient_key: str = "patient",
    report_metric: Callable[[int, float], None] | None = None,
    early_stopping: bool = False,
    patience: int = 25,
    min_delta: float = 1e-4,
    warmup_epochs: int = 0,
    size_training_loss: str = "mse",
    ranking_loss_weight: float = 1.0,
    selection_metric: str = "wasserstein",
    max_size_factor: float | None = None,
) -> CloneSizeModel:
    """Train only clone-size model and optimize combined Wasserstein distance.

    See ``selection_metric`` below to also include ``1 - clone-level Spearman``.

    The primary metric (Optuna ``report_metric``, checkpoint selection, early
    stopping, plateau LR) is the **sum** of cell-level and clone-level 1D
    Wasserstein-1 distances between true and predicted **log-scaled** clone
    sizes (lower is better). Clone-level W1 uses per-clone means. Targets should
    already be log-scaled in ``obs[clone_id_size_key]``, matching
    :class:`~clonetrast.model.CloneSizeModel`.

    Hyperparameters mirror :func:`train` / :class:`CloneTrastTrainer` where
    applicable: architecture search via ``hidden_dims`` or
    ``encoder_depth`` + ``encoder_width`` (+ decay / min width), ``dropout``,
    ``weight_decay``, LR schedulers (``cosine``, ``plateau``), patient-wise
    batching, and early stopping on validation Wasserstein (when ``adata_val`` is
    provided). Cell/clone Wasserstein components, clone-level Spearman, and
    validation MSE / loss are logged to ``size_loss_history.csv`` for diagnostics
    but only the combined (sum) Wasserstein is used for Optuna, checkpointing, or
    early stopping.

    ``size_training_loss`` controls the optimization objective:

    - ``"mse"``: per-cell mean squared error (default).
    - ``"mse_ranking"``: MSE plus ``ranking_loss_weight`` times a clone-level
      pairwise logistic ranking loss within each batch.
    - ``"clone_pearson"``: ``1 - Pearson`` correlation on per-clone mean
      predictions and targets within each batch (falls back to MSE when a batch
      has fewer than two clones).

    ``selection_metric`` controls the metric used for Optuna, checkpointing,
    early stopping, and plateau LR (lower is better):

    - ``"wasserstein"``: cell W1 + clone W1 (default).
    - ``"wasserstein_spearman"``: cell W1 + clone W1 + (1 - clone-level Spearman).

    ``max_size_factor`` (e.g. ``1.5``) caps predictions at ``max_size_factor``
    times the largest raw clone size in ``adata_train``, i.e. at
    ``max(log size) + log(max_size_factor)`` in log space. ``None`` disables the cap.
    """
    if clone_id_key not in adata_train.obs:
        raise KeyError(f"adata_train.obs['{clone_id_key}'] not found.")
    if clone_id_size_key not in adata_train.obs:
        raise KeyError(f"adata_train.obs['{clone_id_size_key}'] not found.")

    sched = str(scheduler).lower()
    if sched not in {"none", "cosine", "plateau"}:
        raise ValueError(
            f"Unsupported scheduler={scheduler!r}. Use one of: 'none', 'cosine', 'plateau'."
        )

    loss_mode = str(size_training_loss).lower()
    if loss_mode not in _SIZE_TRAINING_LOSSES:
        raise ValueError(
            f"Unsupported size_training_loss={size_training_loss!r}. "
            f"Use one of: {sorted(_SIZE_TRAINING_LOSSES)}."
        )
    ranking_loss_weight = float(ranking_loss_weight)
    if ranking_loss_weight < 0.0:
        raise ValueError(f"ranking_loss_weight must be >= 0, got {ranking_loss_weight}.")
    selection_metric = str(selection_metric).lower()
    if selection_metric not in _SIZE_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection_metric={selection_metric!r}. "
            f"Use one of: {sorted(_SIZE_SELECTION_METRICS)}."
        )
    if max_size_factor is not None and float(max_size_factor) <= 0.0:
        raise ValueError(f"max_size_factor must be > 0, got {max_size_factor}.")
    metric_name = (
        "wasserstein_log_sum"
        if selection_metric == "wasserstein"
        else "wasserstein_log_sum_plus_1m_spearman"
    )
    metric_label = (
        "combined Wasserstein (cell + clone, log sizes)"
        if selection_metric == "wasserstein"
        else "combined Wasserstein (cell + clone, log sizes) + (1 - clone Spearman)"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    x_train = get_expression_matrix(adata_train).astype(np.float32, copy=False)
    y_train = np.asarray(adata_train.obs[clone_id_size_key], dtype=np.float32).ravel()
    c_train = np.asarray(adata_train.obs[clone_id_key].astype(str)).ravel()
    n_genes = x_train.shape[1]
    max_log_size = (
        float(np.max(y_train)) + float(np.log(float(max_size_factor)))
        if max_size_factor is not None
        else None
    )

    min_h = int(max(1, min_hidden_dim))
    if hidden_dims is not None:
        size_hidden_dims: list[int] | None = [max(int(d), min_h) for d in hidden_dims]
    elif encoder_depth is not None and encoder_width is not None:
        size_hidden_dims = _build_hidden_dims_from_arch(
            depth=int(encoder_depth),
            width=int(encoder_width),
            width_decay=float(encoder_width_decay),
            min_hidden_dim=min_h,
            embedding_dim=min_h,
        )
    else:
        size_hidden_dims = None

    ds_train = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.arange(x_train.shape[0], dtype=torch.long),
    )
    if patient_wise_batching:
        if patient_key not in adata_train.obs:
            raise ValueError(
                f"patient_wise_batching=True requires adata.obs['{patient_key}']. "
                f"Available keys: {list(adata_train.obs.keys())[:20]}..."
            )
        patient_ids = np.asarray(adata_train.obs[patient_key].astype(str)).ravel()
        if len(patient_ids) != len(c_train):
            raise ValueError(
                f"patient_key '{patient_key}' length {len(patient_ids)} != number of cells {len(c_train)}."
            )
        batch_sampler = PatientWiseBatchSampler(
            patient_ids,
            batch_size=batch_size,
            patients_per_batch=max(1, patients_per_batch),
            rng=np.random.default_rng(seed),
        )
        train_loader = DataLoader(
            ds_train,
            batch_sampler=batch_sampler,
            num_workers=0,
            pin_memory=(dev.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(dev.type == "cuda"),
        )

    if patient_wise_batching:
        print(
            f"Size model: using patient-wise batching "
            f"(k={max(1, patients_per_batch)} patients per batch)."
        )
    if loss_mode != "mse":
        extra = (
            f", ranking_loss_weight={ranking_loss_weight:g}"
            if loss_mode == "mse_ranking"
            else ""
        )
        print(f"Size model: training loss = {loss_mode}{extra}")
    if max_log_size is not None:
        print(
            f"Size model: predictions capped at {float(max_size_factor):g}x max train clone size "
            f"(log cap = {max_log_size:.4f})."
        )
    if selection_metric != "wasserstein":
        print(f"Size model: selection metric = {metric_label}")

    x_val = y_val = c_val = None
    val_loader = None
    if adata_val is not None:
        if clone_id_key not in adata_val.obs:
            raise KeyError(f"adata_val.obs['{clone_id_key}'] not found.")
        if clone_id_size_key not in adata_val.obs:
            raise KeyError(f"adata_val.obs['{clone_id_size_key}'] not found.")
        x_val = get_expression_matrix(adata_val).astype(np.float32, copy=False)
        if x_val.shape[1] != n_genes:
            raise ValueError(
                f"Validation gene dimension ({x_val.shape[1]}) does not match training ({n_genes})."
            )
        y_val = np.asarray(adata_val.obs[clone_id_size_key], dtype=np.float32).ravel()
        c_val = np.asarray(adata_val.obs[clone_id_key].astype(str)).ravel()
        ds_val = TensorDataset(
            torch.from_numpy(x_val),
            torch.from_numpy(y_val),
            torch.arange(x_val.shape[0], dtype=torch.long),
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(dev.type == "cuda"),
        )

    dropout = _validate_dropout(dropout)
    weight_decay = _validate_weight_decay(weight_decay)

    model = CloneSizeModel(
        n_genes=n_genes,
        hidden_dims=size_hidden_dims,
        dropout=dropout,
        max_log_size=max_log_size,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = None
    if sched == "cosine":
        t_max = int(scheduler_t_max) if scheduler_t_max is not None else int(max(1, epochs))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, t_max),
            eta_min=float(scheduler_eta_min),
        )
    elif sched == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(scheduler_factor),
            patience=int(max(1, scheduler_patience)),
        )
    best_metric = float("inf")
    best_epoch = -1
    size_ckpt_path = out_dir / "size_model.pt"
    loss_history_path = out_dir / "size_loss_history.csv"
    es_stagnation = 0
    es_patience = int(max(1, patience))
    warmup_e = int(max(0, warmup_epochs))
    best_es_metric = float("inf")
    loss_history_header = [
        "epoch",
        "train_mse",
        "train_loss",
        "train_rank_loss",
        "train_pearson_loss",
        "train_wasserstein_log_cell",
        "train_wasserstein_log_clone_mean",
        "train_wasserstein_log_sum",
        "train_spearman_clone_mean",
        "train_objective",
    ]
    if val_loader is not None:
        loss_history_header.extend(
            [
                "val_mse",
                "val_loss",
                "val_rank_loss",
                "val_pearson_loss",
                "val_wasserstein_log_cell",
                "val_wasserstein_log_clone_mean",
                "val_wasserstein_log_sum",
                "val_spearman_clone_mean",
                "val_objective",
            ]
        )
    with open(loss_history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(loss_history_header)

    for epoch in range(epochs):
        model.train()
        total_train_mse = 0.0
        total_train_loss = 0.0
        total_train_rank_loss = 0.0
        n_train_rank_batches = 0
        total_train_pearson_loss = 0.0
        n_train_pearson_batches = 0
        n_train_batches = 0
        train_preds = []
        train_clone_ids_batches: list[np.ndarray] = []
        train_true_batches: list[np.ndarray] = []
        for bx, by, bidx in train_loader:
            bx = bx.to(dev)
            by = by.to(dev)
            pred = model(bx)
            bidx_np = bidx.detach().cpu().numpy()
            batch_clone_ids = c_train[bidx_np]
            breakdown = _compute_size_training_loss(
                pred,
                by,
                batch_clone_ids,
                loss_mode=loss_mode,
                ranking_loss_weight=ranking_loss_weight,
            )
            loss = breakdown.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_mse += float(breakdown.mse.item())
            total_train_loss += float(loss.item())
            if breakdown.rank_loss is not None:
                total_train_rank_loss += float(breakdown.rank_loss.item())
                n_train_rank_batches += 1
            if breakdown.pearson_loss is not None:
                total_train_pearson_loss += float(breakdown.pearson_loss.item())
                n_train_pearson_batches += 1
            n_train_batches += 1
            train_preds.append(pred.detach().cpu().numpy())
            train_clone_ids_batches.append(batch_clone_ids)
            train_true_batches.append(y_train[bidx_np])

        train_mse = total_train_mse / max(n_train_batches, 1)
        train_loss = total_train_loss / max(n_train_batches, 1)
        train_rank_loss = (
            total_train_rank_loss / n_train_rank_batches if n_train_rank_batches else ""
        )
        train_pearson_loss = (
            total_train_pearson_loss / n_train_pearson_batches
            if n_train_pearson_batches
            else ""
        )
        train_wasserstein_cell = None
        train_wasserstein_clone = None
        train_wasserstein = None
        train_spearman = None
        train_objective = None
        if train_preds and train_true_batches and train_clone_ids_batches:
            train_pred_all = np.concatenate(train_preds, axis=0)
            train_true_all = np.concatenate(train_true_batches, axis=0)
            train_clone_ids_all = np.concatenate(train_clone_ids_batches, axis=0)
            (
                train_wasserstein_cell,
                train_wasserstein_clone,
                train_wasserstein,
            ) = _combined_wasserstein_log(
                train_clone_ids_all, train_true_all, train_pred_all
            )
            train_spearman = _clone_level_spearman(
                train_clone_ids_all, train_true_all, train_pred_all
            )
            train_objective = _size_selection_objective(
                train_wasserstein, train_spearman, selection_metric
            )
        val_mse = None
        val_loss = None
        val_rank_loss = ""
        val_pearson_loss = ""
        val_wasserstein_cell = None
        val_wasserstein_clone = None
        val_wasserstein = None
        val_spearman = None
        val_objective = None
        metric_for_selection = (
            train_objective if train_objective is not None else float("inf")
        )
        metric_for_early_stopping = metric_for_selection
        if val_loader is not None and x_val is not None and y_val is not None and c_val is not None:
            model.eval()
            total_val_mse = 0.0
            total_val_loss = 0.0
            total_val_rank_loss = 0.0
            n_val_rank_batches = 0
            total_val_pearson_loss = 0.0
            n_val_pearson_batches = 0
            n_val_batches = 0
            val_preds = []
            with torch.no_grad():
                for bx, by, bidx in val_loader:
                    bx = bx.to(dev)
                    by = by.to(dev)
                    pred = model(bx)
                    bidx_np = bidx.detach().cpu().numpy()
                    batch_clone_ids = c_val[bidx_np]
                    breakdown = _compute_size_training_loss(
                        pred,
                        by,
                        batch_clone_ids,
                        loss_mode=loss_mode,
                        ranking_loss_weight=ranking_loss_weight,
                    )
                    total_val_mse += float(breakdown.mse.item())
                    total_val_loss += float(breakdown.loss.item())
                    if breakdown.rank_loss is not None:
                        total_val_rank_loss += float(breakdown.rank_loss.item())
                        n_val_rank_batches += 1
                    if breakdown.pearson_loss is not None:
                        total_val_pearson_loss += float(breakdown.pearson_loss.item())
                        n_val_pearson_batches += 1
                    n_val_batches += 1
                    val_preds.append(pred.cpu().numpy())
            val_mse = total_val_mse / max(n_val_batches, 1)
            val_loss = total_val_loss / max(n_val_batches, 1)
            val_rank_loss = (
                total_val_rank_loss / n_val_rank_batches if n_val_rank_batches else ""
            )
            val_pearson_loss = (
                total_val_pearson_loss / n_val_pearson_batches
                if n_val_pearson_batches
                else ""
            )
            metric_for_early_stopping = float("inf")
            if val_preds:
                val_pred_all = np.concatenate(val_preds, axis=0)
                (
                    val_wasserstein_cell,
                    val_wasserstein_clone,
                    val_wasserstein,
                ) = _combined_wasserstein_log(c_val, y_val, val_pred_all)
                val_spearman = _clone_level_spearman(c_val, y_val, val_pred_all)
                val_objective = _size_selection_objective(
                    val_wasserstein, val_spearman, selection_metric
                )
                metric_for_selection = val_objective
                metric_for_early_stopping = val_objective

        if np.isnan(metric_for_selection):
            metric_for_selection = float("inf")
        if np.isnan(metric_for_early_stopping):
            metric_for_early_stopping = float("inf")

        if lr_scheduler is not None:
            if sched == "plateau":
                lr_scheduler.step(metric_for_early_stopping)
            else:
                lr_scheduler.step()

        if metric_for_selection < best_metric:
            best_metric = metric_for_selection
            best_epoch = epoch + 1
            _save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "n_genes": n_genes,
                    "hidden_dims": size_hidden_dims,
                    "best_metric_rocauc": best_metric,
                    "epoch": epoch + 1,
                },
                size_ckpt_path,
            )

        if early_stopping and val_loader is not None:
            if metric_for_early_stopping < best_es_metric - float(min_delta):
                best_es_metric = metric_for_early_stopping
                es_stagnation = 0
            else:
                es_stagnation += 1
            if (
                (epoch + 1) >= warmup_e
                and es_stagnation >= es_patience
            ):
                print(
                    f"Early stopping at epoch {epoch + 1}: no validation {metric_label} improvement "
                    f"for {es_patience} epoch(s) (min_delta={min_delta:g})."
                )
                break

        with open(loss_history_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                epoch + 1,
                train_mse,
                train_loss,
                train_rank_loss,
                train_pearson_loss,
                train_wasserstein_cell,
                train_wasserstein_clone,
                train_wasserstein,
                train_spearman,
                train_objective,
            ]
            if val_loader is not None:
                row.extend(
                    [
                        val_mse,
                        val_loss,
                        val_rank_loss,
                        val_pearson_loss,
                        val_wasserstein_cell,
                        val_wasserstein_clone,
                        val_wasserstein,
                        val_spearman,
                        val_objective,
                    ]
                )
            writer.writerow(row)

        if report_metric is not None:
            report_metric(epoch, metric_for_selection)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            rank_msg = (
                f"  train_rank_loss = {train_rank_loss:.4f}"
                if train_rank_loss != ""
                else ""
            )
            pearson_msg = (
                f"  train_pearson_loss = {train_pearson_loss:.4f}"
                if train_pearson_loss != ""
                else ""
            )
            if val_loader is not None:
                val_rank_msg = (
                    f"  val_rank_loss = {val_rank_loss:.4f}"
                    if val_rank_loss != ""
                    else ""
                )
                val_pearson_msg = (
                    f"  val_pearson_loss = {val_pearson_loss:.4f}"
                    if val_pearson_loss != ""
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}  train_mse = {train_mse:.4f}  "
                    f"train_loss = {train_loss:.4f}{rank_msg}{pearson_msg}  "
                    f"train_wasserstein_sum = {train_wasserstein:.4f}  "
                    f"(cell={train_wasserstein_cell:.4f}, clone={train_wasserstein_clone:.4f})  "
                    f"train_spearman = {train_spearman:.4f}  "
                    f"val_mse = {val_mse:.4f}  "
                    f"val_loss = {val_loss:.4f}{val_rank_msg}{val_pearson_msg}  "
                    f"val_wasserstein_sum = {val_wasserstein:.4f}  "
                    f"(cell={val_wasserstein_cell:.4f}, clone={val_wasserstein_clone:.4f})  "
                    f"val_spearman = {val_spearman:.4f}  "
                    f"val_objective = {val_objective:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs}  train_mse = {train_mse:.4f}  "
                    f"train_loss = {train_loss:.4f}{rank_msg}{pearson_msg}  "
                    f"train_wasserstein_sum = {train_wasserstein:.4f}  "
                    f"(cell={train_wasserstein_cell:.4f}, clone={train_wasserstein_clone:.4f})  "
                    f"train_spearman = {train_spearman:.4f}"
                )

    if best_epoch < 0:
        final_epoch = (epoch + 1) if epochs > 0 else 0
        print(
            f"Warning: no epoch improved the checkpoint metric ({metric_label}); "
            f"saving final model weights from epoch {final_epoch}."
        )
        best_epoch = final_epoch
        _save_checkpoint(
            {
                "model_state_dict": model.state_dict(),
                "n_genes": n_genes,
                "hidden_dims": size_hidden_dims,
                "best_metric_rocauc": best_metric,
                "epoch": final_epoch,
            },
            size_ckpt_path,
        )

    size_ckpt = torch.load(size_ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(size_ckpt["model_state_dict"], strict=True)
    model.eval()

    config = {
        "n_genes": n_genes,
        "hidden_dims": size_hidden_dims,
        "size_model_hidden_dims": size_hidden_dims,
        "encoder_depth": encoder_depth,
        "encoder_width": encoder_width,
        "encoder_width_decay": encoder_width_decay,
        "min_hidden_dim": min_hidden_dim,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "clone_id_key": clone_id_key,
        "clone_id_size_key": clone_id_size_key,
        "has_size_model": True,
        "trained_size_model_only": True,
        "size_metric": metric_name,
        "size_metric_higher_is_better": False,
        "selection_metric": selection_metric,
        "size_training_loss": loss_mode,
        "ranking_loss_weight": ranking_loss_weight,
        "early_stopping_metric": f"val_{metric_name}",
        "checkpoint_selection_metric": f"val_{metric_name}",
        "max_size_factor": max_size_factor,
        "max_log_size": max_log_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "scheduler": sched,
        "scheduler_t_max": scheduler_t_max,
        "scheduler_eta_min": scheduler_eta_min,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "seed": seed,
        "patient_wise_batching": patient_wise_batching,
        "patients_per_batch": max(1, patients_per_batch),
        "patient_key": patient_key,
        "has_validation_data": adata_val is not None,
        "early_stopping": early_stopping,
        "patience": es_patience,
        "min_delta": min_delta,
        "warmup_epochs": warmup_e,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "gene_order.json", "w") as f:
        json.dump(list(adata_train.var_names), f)
    print(
        f"Size-model training complete. Best {metric_label}: "
        f"{best_metric:.4f} (epoch {best_epoch})."
    )
    print(f"Saved size model and config to {out_dir}")
    return model


def load_model(
    checkpoint_dir: str | Path,
    device: str | None = None,
    load_size_model: bool = True,
) -> CloneTrastModel | tuple[CloneTrastModel, CloneSizeModel]:
    """Load trained model(s) from checkpoint directory.

    Parameters
    ----------
    checkpoint_dir
        Directory containing model.pt and config.json.
    device
        Device to load model on. If None, auto-detects.
    load_size_model
        If True and size model exists, load and include in return.

    Returns
    -------
    model alone; or (model, size_model) when available and requested.

    """
    checkpoint_dir = Path(checkpoint_dir)
    ckpt = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    model = CloneTrastModel(
        n_genes=config["n_genes"],
        hidden_dims=config["hidden_dims"],
        embedding_dim=config["embedding_dim"],
        projection_dim=config["projection_dim"],
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    size_model = None
    if load_size_model and config.get("has_size_model", False):
        size_ckpt_path = checkpoint_dir / "size_model.pt"
        if size_ckpt_path.exists():
            size_ckpt = torch.load(size_ckpt_path, map_location="cpu", weights_only=True)
            # Backward compatibility:
            # - New checkpoints store fixed size-model architecture as None/default.
            # - Older checkpoints may only have config["hidden_dims"].
            size_hidden_dims = config.get("size_model_hidden_dims", config.get("hidden_dims"))
            size_model = CloneSizeModel(
                n_genes=config["n_genes"],
                hidden_dims=size_hidden_dims,
                dropout=float(config.get("dropout", 0.1)),
                max_log_size=config.get("max_log_size"),
            )
            size_model.load_state_dict(size_ckpt["model_state_dict"], strict=True)
            size_model = size_model.to(dev)
            size_model.eval()

    if size_model is not None:
        return model, size_model
    return model
