"""Tests for CloneDataset."""

import numpy as np
import pytest
import torch

from clonetrast.model import CloneDataset


def test_clone_dataset_basic():
    X = np.random.randn(10, 20).astype(np.float32)
    clone_ids = np.array([f"c{i // 3}" for i in range(10)])
    ds = CloneDataset(X, clone_ids)
    assert len(ds) == 10
    x, c = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (20,)
    assert isinstance(c, int)


def test_clone_dataset_with_sizes():
    X = np.random.randn(6, 5).astype(np.float32)
    clone_ids = np.array(["a", "a", "b", "b", "c", "c"])
    sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    ds = CloneDataset(X, clone_ids, clone_sizes=sizes)
    x, c, s = ds[2]
    assert s.item() == pytest.approx(3.0)


def test_clone_dataset_gene_order():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    clone_ids = np.array(["x", "y", "z"])
    order = np.array([2, 0, 1, 3])
    ds = CloneDataset(X, clone_ids, gene_order=order)
    x0, _ = ds[0]
    expected = torch.tensor([2.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    assert torch.allclose(x0, expected)
