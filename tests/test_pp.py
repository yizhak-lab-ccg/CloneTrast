"""Tests for preprocessing."""

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from clonetrast.pp import prepare_adata, split_train_test


def test_prepare_adata():
    X = np.random.poisson(1.0, (50, 100)).astype(np.int32)
    adata = AnnData(X)
    prepare_adata(
        adata,
        use_raw=False,
        target_sum=10_000.0,
        log1p=True,
        copy=False,
    )
    assert adata.X.shape == (50, 100)


def test_prepare_adata_row_normalization():
    X = np.random.poisson(1.0, (20, 50)).astype(np.int32)
    adata = AnnData(X)
    prepare_adata(
        adata,
        use_raw=False,
        target_sum=10_000.0,
        log1p=False,
        copy=False,
    )
    row_sums = np.asarray(adata.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.full(20, 10_000.0), rtol=1e-5, atol=0.01)


def test_prepare_adata_csr_sparse():
    """CSR sparse count matrix is converted to dense and normalized correctly."""
    X_dense = np.random.poisson(1.0, (30, 80)).astype(np.int32)
    X_csr = csr_matrix(X_dense)
    adata = AnnData(X_csr)
    prepare_adata(
        adata,
        use_raw=False,
        target_sum=10_000.0,
        log1p=True,
        copy=False,
    )
    assert not hasattr(adata.X, "toarray"), "Output should be dense"
    assert adata.X.shape == (30, 80)


def test_split_train_test():
    n_cells = 100
    clone_ids = np.array([f"clone_{i % 10}" for i in range(n_cells)])  # 10 clones
    patient_ids = np.array([f"patient_{i % 5}" for i in range(n_cells)])  # 5 patients
    X = np.random.poisson(1.0, (n_cells, 50)).astype(np.int32)
    adata = AnnData(X, obs={"clone_id": clone_ids, "patient": patient_ids})
    prepare_adata(adata, use_raw=False, copy=False)
    train, test = split_train_test(
        adata, patient_key="patient", clone_id_key="clone_id", test_fraction=0.3, seed=42
    )
    train_clones = set(train.obs["clone_id"].unique())
    test_clones = set(test.obs["clone_id"].unique())
    assert train_clones.isdisjoint(test_clones)
    assert len(train) + len(test) == n_cells
