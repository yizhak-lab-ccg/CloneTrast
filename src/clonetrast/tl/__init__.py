"""Tools for training, embedding and visualization."""

from __future__ import annotations

from .embedding import compute_umap, embed, predict_clone_size
from .metrics import clone_purity, evaluate_embeddings
from .pretrained import (
    default_pretrained_cache_dir,
    ensure_clone_size_checkpoint,
    ensure_contrastive_checkpoint,
)
from .train import CloneTrastTrainer, load_model, train, train_size_model

__all__ = [
    "compute_umap",
    "default_pretrained_cache_dir",
    "embed",
    "ensure_clone_size_checkpoint",
    "ensure_contrastive_checkpoint",
    "predict_clone_size",
    "evaluate_embeddings",
    "clone_purity",
    "CloneTrastTrainer",
    "load_model",
    "train",
    "train_size_model",
]
