"""Tests for embedding helpers."""

import json
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from clonetrast.tl.embedding import compute_umap, embed, predict_clone_size
from clonetrast.tl.train import load_model as real_load_model


def _adata_from_ckpt(ckpt_dir: Path, n_cells: int = 4, seed: int = 1) -> AnnData:
    genes = json.loads((ckpt_dir / "gene_order.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n_cells, len(genes))).astype(np.int32)
    adata = AnnData(X)
    adata.var_names = genes
    return adata


def test_embed_writes_obsm(tiny_ckpt_dir):
    adata = _adata_from_ckpt(tiny_ckpt_dir, n_cells=4, seed=1)
    embed(adata, tiny_ckpt_dir, device="cpu", batch_size=2, use_pretrained=False)
    assert "X_clonetrast" in adata.obsm
    assert adata.obsm["X_clonetrast"].shape == (4, 8)


def test_compute_umap_small_neighbors():
    n = 20
    rng = np.random.default_rng(2)
    adata = AnnData(np.ones((n, 3), dtype=np.float32))
    adata.obsm["X_clonetrast"] = rng.standard_normal((n, 8)).astype(np.float32)
    compute_umap(
        adata,
        obsm_key="X_clonetrast",
        umap_key="X_umap_test",
        n_neighbors=5,
        save_neighbors=False,
    )
    assert adata.obsm["X_umap_test"].shape == (n, 2)


def test_predict_clone_size(tiny_ckpt_with_size):
    adata = _adata_from_ckpt(tiny_ckpt_with_size, n_cells=6, seed=3)
    predict_clone_size(
        adata, tiny_ckpt_with_size, device="cpu", batch_size=4, use_pretrained=False
    )
    assert "predicted_clone_size" in adata.obs
    assert len(adata.obs["predicted_clone_size"]) == 6


def test_embed_ensembl_mismatch_raises(tiny_ckpt_dir):
    genes = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003", "ENSG00000000004", "ENSG00000000005"]
    (tiny_ckpt_dir / "gene_order.json").write_text(json.dumps(genes), encoding="utf-8")
    X = np.random.poisson(5, size=(2, 5)).astype(np.int32)
    adata = AnnData(X)
    adata.var_names = ["TP53", "MYC", "EGFR", "KRAS", "BRAF"]
    with pytest.raises(ValueError, match="Ensembl"):
        embed(adata, tiny_ckpt_dir, device="cpu", use_pretrained=False)


def test_embed_ensembl_ids_match(tiny_ckpt_dir):
    genes = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003", "ENSG00000000004", "ENSG00000000005"]
    (tiny_ckpt_dir / "gene_order.json").write_text(json.dumps(genes), encoding="utf-8")
    X = np.random.poisson(5, size=(2, 5)).astype(np.int32)
    adata = AnnData(X)
    adata.var_names = genes
    embed(adata, tiny_ckpt_dir, device="cpu", use_pretrained=False)
    assert adata.obsm["X_clonetrast"].shape == (2, 8)


def test_embed_use_pretrained_mocked(tiny_ckpt_dir, monkeypatch: pytest.MonkeyPatch):
    adata = _adata_from_ckpt(tiny_ckpt_dir)
    seen: dict[str, object] = {}

    def fake_ensure(cache=None):
        seen["cache"] = cache
        return tiny_ckpt_dir

    monkeypatch.setattr("clonetrast.tl.embedding.ensure_contrastive_checkpoint", fake_ensure)
    embed(
        adata,
        checkpoint_dir=None,
        device="cpu",
        use_pretrained=True,
        pretrained_cache_dir="/tmp/fake-cache",
    )
    assert seen["cache"] == "/tmp/fake-cache"
    assert "X_clonetrast" in adata.obsm


def test_embed_normalize_raw_and_auto_device(tiny_ckpt_dir):
    adata = _adata_from_ckpt(tiny_ckpt_dir, n_cells=3, seed=4)
    embed(
        adata,
        tiny_ckpt_dir,
        use_layer=None,
        normalize_raw=True,
        device=None,
        batch_size=2,
        use_pretrained=False,
    )
    assert adata.obsm["X_clonetrast"].shape == (3, 8)


def test_embed_handles_load_model_tuple(tiny_ckpt_dir, monkeypatch: pytest.MonkeyPatch):
    adata = _adata_from_ckpt(tiny_ckpt_dir, n_cells=2, seed=5)

    def as_tuple(ckpt, device=None, load_size_model=True):
        model = real_load_model(ckpt, device=device, load_size_model=False)
        return model, object()

    monkeypatch.setattr("clonetrast.tl.embedding.load_model", as_tuple)
    embed(adata, tiny_ckpt_dir, device="cpu", use_pretrained=False)
    assert adata.obsm["X_clonetrast"].shape == (2, 8)


def test_predict_clone_size_use_pretrained_mocked(tiny_ckpt_with_size, monkeypatch: pytest.MonkeyPatch):
    adata = _adata_from_ckpt(tiny_ckpt_with_size, n_cells=4, seed=6)

    def fake_ensure(cache=None):
        return tiny_ckpt_with_size

    monkeypatch.setattr("clonetrast.tl.embedding.ensure_clone_size_checkpoint", fake_ensure)
    predict_clone_size(
        adata,
        checkpoint_dir=None,
        device=None,
        normalize_raw=True,
        use_layer=None,
        use_pretrained=True,
        pretrained_cache_dir="/tmp/fake-size-cache",
    )
    assert "predicted_clone_size" in adata.obs
    assert len(adata.obs["predicted_clone_size"]) == 4


def test_predict_clone_size_missing_size_model_raises(tiny_ckpt_dir):
    adata = _adata_from_ckpt(tiny_ckpt_dir, n_cells=2, seed=7)
    with pytest.raises(ValueError, match="Size model not found"):
        predict_clone_size(adata, tiny_ckpt_dir, device="cpu", use_pretrained=False)


def test_compute_umap_saves_neighbors():
    n = 20
    rng = np.random.default_rng(8)
    adata = AnnData(np.ones((n, 3), dtype=np.float32))
    adata.obsm["X_clonetrast"] = rng.standard_normal((n, 8)).astype(np.float32)
    compute_umap(
        adata,
        obsm_key="X_clonetrast",
        umap_key="umap_custom",
        n_neighbors=5,
        save_neighbors=True,
    )
    assert adata.obsm["umap_custom"].shape == (n, 2)
    assert "connectivities_umap_custom" in adata.obsp
    assert "distances_umap_custom" in adata.obsp
    assert adata.uns["neighbors_umap_custom"]["connectivities_key"] == "connectivities_umap_custom"
