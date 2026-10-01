"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from clonetrast.model import CloneSizeModel, CloneTrastModel


@pytest.fixture
def tiny_ckpt_dir(tmp_path: Path) -> Path:
    """Minimal contrastive checkpoint directory (CPU-loadable)."""
    n_genes = 5
    genes = [f"g{i}" for i in range(n_genes)]
    model = CloneTrastModel(
        n_genes=n_genes,
        hidden_dims=[16],
        embedding_dim=8,
        projection_dim=4,
    )
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "model.pt")
    cfg = {
        "n_genes": n_genes,
        "hidden_dims": [16],
        "embedding_dim": 8,
        "projection_dim": 4,
        "has_size_model": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "gene_order.json").write_text(json.dumps(genes), encoding="utf-8")
    return tmp_path


@pytest.fixture
def tiny_ckpt_with_size(tmp_path: Path) -> Path:
    """Checkpoint with contrastive + size models."""
    n_genes = 5
    genes = [f"g{i}" for i in range(n_genes)]
    hidden = [16, 8]
    model = CloneTrastModel(
        n_genes=n_genes,
        hidden_dims=[16],
        embedding_dim=8,
        projection_dim=4,
    )
    size_model = CloneSizeModel(n_genes=n_genes, hidden_dims=hidden)
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "model.pt")
    torch.save({"model_state_dict": size_model.state_dict()}, tmp_path / "size_model.pt")
    cfg = {
        "n_genes": n_genes,
        "hidden_dims": [16],
        "size_model_hidden_dims": hidden,
        "embedding_dim": 8,
        "projection_dim": 4,
        "has_size_model": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "gene_order.json").write_text(json.dumps(genes), encoding="utf-8")
    return tmp_path
