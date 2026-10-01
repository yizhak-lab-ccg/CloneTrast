"""Tests for clone purity metrics."""

import numpy as np
import pytest
from anndata import AnnData

from clonetrast.tl.metrics import clone_purity, evaluate_embeddings


def test_clone_purity_global_knn():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((12, 16)).astype(np.float32)
    clones = np.array([f"c{i // 3}" for i in range(12)])
    p = clone_purity(emb, clones, k=5, n_jobs=1, min_clone_size=2)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_clone_purity_with_patients():
    emb = np.random.randn(20, 8).astype(np.float32)
    clones = np.array([f"c{i // 4}" for i in range(20)])
    patients = np.array([f"p{i // 10}" for i in range(20)])
    p = clone_purity(
        emb,
        clones,
        k=3,
        n_jobs=1,
        min_clone_size=2,
        patient_ids=patients,
        weight_by_cells=True,
    )
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_clone_purity_per_patient_dict():
    emb = np.random.randn(12, 4).astype(np.float32)
    clones = np.array([f"c{i // 3}" for i in range(12)])
    patients = np.array(["p0"] * 6 + ["p1"] * 6)
    out = clone_purity(
        emb,
        clones,
        k=2,
        patient_ids=patients,
        return_per_patient=True,
        weight_by_cells=False,
    )
    overall, per_p = out
    assert isinstance(overall, float)
    assert isinstance(per_p, dict)


def test_evaluate_embeddings_joint():
    n = 15
    rng = np.random.default_rng(0)
    adata = AnnData(rng.random((n, 3)).astype(np.float32))
    adata.obs["clone_id"] = [f"c{i // 5}" for i in range(n)]
    adata.obs["patient"] = [f"p{i // 8}" for i in range(n)]
    adata.obsm["X_clonetrast"] = rng.standard_normal((n, 8)).astype(np.float32)
    out = evaluate_embeddings(
        adata,
        obsm_key="X_clonetrast",
        clone_id_key="clone_id",
        patient_key="patient",
        k=5,
        min_clone_size=2,
        return_purity_per_patient=True,
        weight_by_cells=True,
    )
    assert "purity" in out
    assert "purity_per_patient" in out
    assert 0.0 <= out["purity"] <= 1.0


def test_evaluate_embeddings_missing_obsm_raises():
    adata = AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obs["clone_id"] = ["a", "b", "c"]
    with pytest.raises(KeyError, match="obsm"):
        evaluate_embeddings(adata, obsm_key="X_missing")
