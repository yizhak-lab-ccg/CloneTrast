"""Tests for normalize helpers beyond prepare_adata."""

import numpy as np
import pytest
from anndata import AnnData

from clonetrast.pp.normalize import (
    align_genes_to_train,
    get_expression_matrix,
    normalize_expression_matrix,
)


def test_get_expression_matrix_prefers_layer():
    X = np.ones((3, 4), dtype=np.float32)
    adata = AnnData(X)
    adata.layers["clonetrast"] = np.full((3, 4), 2.0, dtype=np.float32)
    mat = get_expression_matrix(adata, use_layer="clonetrast")
    assert mat[0, 0] == pytest.approx(2.0)


def test_align_genes_reorders_and_fills_zeros():
    # Training order: A, B, C - adata has columns B, A (extra D dropped)
    X = np.array([[10, 20, 99]], dtype=np.int32)
    adata = AnnData(X)
    adata.var_names = ["B", "A", "D"]
    train_order = ["A", "B", "C"]
    out = align_genes_to_train(adata, train_order, source_layer=None)
    assert out.shape == (1, 3)
    assert out[0, 0] == pytest.approx(20.0)  # A from second column
    assert out[0, 1] == pytest.approx(10.0)  # B
    assert out[0, 2] == pytest.approx(0.0)  # C missing


def test_normalize_expression_matrix():
    X = np.array([[100, 300]], dtype=np.int32)
    out = normalize_expression_matrix(X, target_sum=1000.0, log1p=True)
    row_sum = float(out.sum())
    assert row_sum > 0
    assert out.shape == X.shape
