"""Short CPU training smoke tests."""

from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from anndata import AnnData

import clonetrast as ct
from clonetrast.tl.train import (
    CloneTrastTrainer,
    PatientWiseBatchSampler,
    _batch_clone_means,
    _build_hidden_dims_from_arch,
    _cell_level_wasserstein_log,
    _clone_level_roc_auc,
    _clone_level_spearman,
    _clone_level_wasserstein_log,
    _combined_wasserstein_log,
    _compute_size_training_loss,
    _pairwise_ranking_loss,
    _save_checkpoint,
    _validate_dropout,
    _validate_weight_decay,
    load_model,
)

# Small encoder so CPU runs stay fast; widths must be >= embedding_dim (trainer constraint).
_FAST_ARCH = {
    "hidden_dims": [64, 32],
    "embedding_dim": 32,
    "projection_dim": 16,
    "batch_size": 16,
    "device": "cpu",
}


def _tiny_split_shared_genes(n_genes: int = 24):
    rng = np.random.default_rng(42)
    genes = [f"G{i}" for i in range(n_genes)]

    n_train = 32
    clone_ids = np.array([f"c{i // 8}" for i in range(n_train)])
    patient_ids = np.array([f"p{i // 16}" for i in range(n_train)])
    X = rng.poisson(5, size=(n_train, n_genes)).astype(np.int32)
    ad_train = AnnData(X, obs={"clone_id": clone_ids, "patient": patient_ids})
    ad_train.var_names = genes

    n_val = 16
    clone_ids_v = np.array([f"cv{i // 4}" for i in range(n_val)])
    patient_ids_v = np.array([f"pv{i // 8}" for i in range(n_val)])
    Xv = rng.poisson(5, size=(n_val, n_genes)).astype(np.int32)
    ad_val = AnnData(Xv, obs={"clone_id": clone_ids_v, "patient": patient_ids_v})
    ad_val.var_names = genes

    ct.pp.prepare_adata(ad_train, use_raw=False, copy=False)
    ct.pp.prepare_adata(ad_val, use_raw=False, copy=False)
    return ad_train, ad_val


def _tiny_split_with_clone_sizes(n_genes: int = 24):
    """Train/val AnnData with obs['clone_id_size'] for train_size_model."""
    rng = np.random.default_rng(99)
    genes = [f"G{i}" for i in range(n_genes)]

    n_train = 32
    clone_ids = np.array([f"c{i // 8}" for i in range(n_train)])
    patient_ids = np.array([f"p{i // 16}" for i in range(n_train)])
    clone_sizes = rng.uniform(1.0, 10.0, size=n_train).astype(np.float32)
    X = rng.poisson(5, size=(n_train, n_genes)).astype(np.int32)
    ad_train = AnnData(
        X,
        obs={
            "clone_id": clone_ids,
            "patient": patient_ids,
            "clone_id_size": clone_sizes,
        },
    )
    ad_train.var_names = genes

    n_val = 16
    clone_ids_v = np.array([f"cv{i // 4}" for i in range(n_val)])
    patient_ids_v = np.array([f"pv{i // 8}" for i in range(n_val)])
    clone_sizes_v = rng.uniform(1.0, 10.0, size=n_val).astype(np.float32)
    Xv = rng.poisson(5, size=(n_val, n_genes)).astype(np.int32)
    ad_val = AnnData(
        Xv,
        obs={
            "clone_id": clone_ids_v,
            "patient": patient_ids_v,
            "clone_id_size": clone_sizes_v,
        },
    )
    ad_val.var_names = genes

    ct.pp.prepare_adata(ad_train, use_raw=False, copy=False)
    ct.pp.prepare_adata(ad_val, use_raw=False, copy=False)
    return ad_train, ad_val


def test_trainer_one_epoch_checkpoint(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        clone_id_key="clone_id",
        out_dir=tmp_path,
        epochs=1,
        **_FAST_ARCH,
        patient_key="patient",
        scheduler="none",
    )
    model = trainer.train()
    assert model is not None
    assert trainer.model is model
    assert trainer.gene_order is not None
    assert len(trainer.gene_order) == ad_train.n_vars
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "gene_order.json").exists()


def test_load_model_roundtrip(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        clone_id_key="clone_id",
        out_dir=tmp_path,
        epochs=1,
        **_FAST_ARCH,
        patient_key="patient",
        scheduler="none",
    )
    trainer.train()
    m = load_model(tmp_path, device=None, load_size_model=False)
    assert m.n_genes == ad_train.n_vars


def test_trainer_cosine_scheduler(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=2,
        scheduler="cosine",
        scheduler_t_max=2,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    assert (tmp_path / "model.pt").exists()


def test_trainer_plateau_scheduler_val(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=2,
        scheduler="plateau",
        scheduler_patience=1,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    assert (tmp_path / "loss_history.csv").exists()


def test_trainer_plateau_scheduler_train_only(tmp_path: Path):
    ad_train, _ = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=None,
        out_dir=tmp_path,
        epochs=2,
        scheduler="plateau",
        scheduler_patience=1,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    assert (tmp_path / "model.pt").exists()


def test_trainer_patient_wise_batching(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        patient_wise_batching=True,
        patients_per_batch=2,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    assert (tmp_path / "config.json").exists()


def test_trainer_compute_train_metric(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        compute_train_metric=True,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    text = (tmp_path / "loss_history.csv").read_text()
    assert "train_clone_purity" in text


def test_trainer_report_metric_callback(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    reported: list[tuple[int, float]] = []

    def _cb(step: int, value: float) -> None:
        reported.append((step, value))

    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        report_metric=_cb,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    assert len(reported) == 1
    assert reported[0][0] == 0


def test_trainer_early_stopping_patched_metric(tmp_path: Path):
    """Constant validation metric forces stagnation so early stopping exits before max epochs."""
    ad_train, ad_val = _tiny_split_shared_genes()
    with patch.object(import_module("clonetrast.tl.train"), "clone_purity", return_value=0.5):
        trainer = CloneTrastTrainer(
            ad_train,
            adata_val=ad_val,
            out_dir=tmp_path,
            epochs=20,
            early_stopping=True,
            patience=1,
            warmup_epochs=0,
            **_FAST_ARCH,
            patient_key="patient",
        )
        trainer.train()
    assert (tmp_path / "model.pt").exists()


def test_tl_train_wrapper(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    model = ct.tl.train(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        patient_key="patient",
        **_FAST_ARCH,
    )
    assert model is not None
    assert (tmp_path / "model.pt").exists()


def test_train_size_model_smoke(tmp_path: Path):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    size_m = ct.tl.train_size_model(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        batch_size=16,
        device="cpu",
        hidden_dims=[64, 32],
        patient_key="patient",
    )
    assert size_m is not None
    assert (tmp_path / "size_model.pt").exists()
    assert (tmp_path / "size_loss_history.csv").exists()


def test_train_size_model_cosine_scheduler(tmp_path: Path):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    ct.tl.train_size_model(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=2,
        batch_size=16,
        device="cpu",
        hidden_dims=[64, 32],
        scheduler="cosine",
        scheduler_t_max=2,
        patient_key="patient",
    )
    assert (tmp_path / "size_loss_history.csv").exists()


def test_invalid_scheduler_raises(tmp_path: Path):
    ad_train, _ = _tiny_split_shared_genes()
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        CloneTrastTrainer(
            ad_train,
            adata_val=None,
            scheduler="not_a_scheduler",
            out_dir=tmp_path,
            epochs=1,
            **_FAST_ARCH,
        )


def test_patient_wise_batch_sampler_covers_cells():
    patient_ids = np.array(["a", "a", "b", "b", "c", "c"])
    sampler = PatientWiseBatchSampler(
        patient_ids,
        batch_size=4,
        patients_per_batch=2,
        rng=np.random.default_rng(0),
    )
    seen = set()
    for batch in sampler:
        seen.update(batch)
    assert seen == set(range(6))
    assert len(sampler) >= 1


def test_dropout_and_weight_decay_saved_in_config(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        dropout=0.25,
        weight_decay=1e-4,
        **_FAST_ARCH,
        patient_key="patient",
    )
    trainer.train()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["dropout"] == 0.25
    assert config["weight_decay"] == 1e-4


def test_train_size_model_weight_decay_in_config(tmp_path: Path):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    ct.tl.train_size_model(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        batch_size=16,
        device="cpu",
        hidden_dims=[64, 32],
        dropout=0.2,
        weight_decay=1e-3,
        patient_key="patient",
    )
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["dropout"] == 0.2
    assert config["weight_decay"] == 1e-3


@pytest.mark.parametrize("size_training_loss", ["mse", "mse_ranking", "clone_pearson"])
def test_train_size_model_training_loss_modes(tmp_path: Path, size_training_loss: str):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    out = tmp_path / size_training_loss
    ct.tl.train_size_model(
        ad_train,
        adata_val=ad_val,
        out_dir=out,
        epochs=1,
        batch_size=16,
        device="cpu",
        hidden_dims=[64, 32],
        patient_key="patient",
        size_training_loss=size_training_loss,
        ranking_loss_weight=0.5,
    )
    config = json.loads((out / "config.json").read_text())
    assert config["size_training_loss"] == size_training_loss
    assert config["ranking_loss_weight"] == 0.5

    history_path = out / "size_loss_history.csv"
    assert history_path.exists()
    lines = history_path.read_text().strip().splitlines()
    header = lines[0].split(",")
    assert header == [
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
    row = lines[1].split(",")
    assert row[0] == "1"
    train_mse = float(row[1])
    train_loss = float(row[2])
    train_rank_loss = row[3]
    train_pearson_loss = row[4]
    val_mse = float(row[10])
    val_loss = float(row[11])
    val_rank_loss = row[12]
    val_pearson_loss = row[13]
    if size_training_loss == "mse":
        assert train_loss == pytest.approx(train_mse)
        assert train_rank_loss == ""
        assert train_pearson_loss == ""
        assert val_loss == pytest.approx(val_mse)
        assert val_rank_loss == ""
        assert val_pearson_loss == ""
    elif size_training_loss == "mse_ranking":
        assert train_rank_loss != ""
        assert train_pearson_loss == ""
        assert train_loss >= train_mse
        assert val_rank_loss != ""
        assert val_pearson_loss == ""
        assert val_loss >= val_mse
    else:
        assert train_rank_loss == ""
        assert train_pearson_loss != ""
        assert val_rank_loss == ""
        assert val_pearson_loss != ""


def test_train_size_model_invalid_training_loss_raises(tmp_path: Path):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    with pytest.raises(ValueError, match="Unsupported size_training_loss"):
        ct.tl.train_size_model(
            ad_train,
            adata_val=ad_val,
            out_dir=tmp_path,
            epochs=1,
            batch_size=16,
            device="cpu",
            hidden_dims=[64, 32],
            patient_key="patient",
            size_training_loss="not_a_loss",
        )


def test_train_size_model_saves_fallback_when_wasserstein_undefined(tmp_path: Path):
    """NaN validation Wasserstein must not leave size_model.pt missing."""
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    with patch.object(
        import_module("clonetrast.tl.train"),
        "_combined_wasserstein_log",
        return_value=(float("nan"), float("nan"), float("nan")),
    ):
        model = ct.tl.train_size_model(
            ad_train,
            adata_val=ad_val,
            out_dir=tmp_path,
            epochs=2,
            batch_size=16,
            device="cpu",
            hidden_dims=[64, 32],
            patient_key="patient",
        )
    assert model is not None
    assert (tmp_path / "size_model.pt").exists()


def test_cell_level_wasserstein_log_edges():
    y_true = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y_pred = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    assert _cell_level_wasserstein_log(y_true, y_pred) == pytest.approx(0.0)

    shifted = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert _cell_level_wasserstein_log(y_true, shifted) == pytest.approx(1.0)

    assert np.isnan(_cell_level_wasserstein_log(np.array([]), np.array([])))
    with pytest.raises(ValueError, match="equal length"):
        _cell_level_wasserstein_log(y_true, y_pred[:2])


def test_clone_level_wasserstein_log_edges():
    clone_ids = np.array(["a", "a", "b", "b", "c", "c"])
    y_true = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    y_pred = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    assert _clone_level_wasserstein_log(clone_ids, y_true, y_pred) == pytest.approx(0.0)

    # Shift each clone mean by +1 → W1 = 1 between clone-mean distributions.
    shifted = y_pred + 1.0
    assert _clone_level_wasserstein_log(clone_ids, y_true, shifted) == pytest.approx(1.0)

    # Within-clone averaging: cells differ but clone means match → W1 = 0.
    noisy_pred = np.array([1.0, -1.0, 2.0, 0.0, 3.0, 1.0], dtype=np.float64)
    assert _clone_level_wasserstein_log(clone_ids, y_true, noisy_pred) == pytest.approx(0.0)

    assert np.isnan(
        _clone_level_wasserstein_log(np.array([]), np.array([]), np.array([]))
    )
    with pytest.raises(ValueError, match="equal length"):
        _clone_level_wasserstein_log(clone_ids, y_true, y_pred[:2])


def test_combined_wasserstein_log_is_sum():
    clone_ids = np.array(["a", "a", "b", "b"])
    y_true = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float64)
    y_pred = np.array([1.0, 1.0, 3.0, 3.0], dtype=np.float64)
    cell_w, clone_w, combined = _combined_wasserstein_log(clone_ids, y_true, y_pred)
    assert cell_w == pytest.approx(1.0)
    assert clone_w == pytest.approx(1.0)
    assert combined == pytest.approx(cell_w + clone_w)


def test_build_hidden_dims_from_arch():
    dims = _build_hidden_dims_from_arch(
        depth=3,
        width=64,
        width_decay=0.5,
        min_hidden_dim=16,
        embedding_dim=32,
    )
    assert dims == [64, 32, 32]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"depth": 0, "width": 8, "width_decay": 0.5, "min_hidden_dim": 4, "embedding_dim": 4}, "depth"),
        ({"depth": 1, "width": 0, "width_decay": 0.5, "min_hidden_dim": 4, "embedding_dim": 4}, "width"),
        ({"depth": 1, "width": 8, "width_decay": 0.0, "min_hidden_dim": 4, "embedding_dim": 4}, "width_decay"),
        ({"depth": 1, "width": 8, "width_decay": 0.5, "min_hidden_dim": 0, "embedding_dim": 4}, "min_hidden_dim"),
        ({"depth": 1, "width": 8, "width_decay": 0.5, "min_hidden_dim": 4, "embedding_dim": 0}, "embedding_dim"),
    ],
)
def test_build_hidden_dims_from_arch_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build_hidden_dims_from_arch(**kwargs)


def test_validate_dropout_and_weight_decay():
    assert _validate_dropout(0.0) == 0.0
    assert _validate_weight_decay(0.0) == 0.0
    with pytest.raises(ValueError, match="dropout"):
        _validate_dropout(1.0)
    with pytest.raises(ValueError, match="weight_decay"):
        _validate_weight_decay(-1e-3)


def test_trainer_architecture_and_validation_without_training(tmp_path: Path):
    ad_train, _ = _tiny_split_shared_genes()
    from_arch = CloneTrastTrainer(
        ad_train,
        hidden_dims=None,
        encoder_depth=3,
        encoder_width=64,
        encoder_width_decay=0.5,
        min_hidden_dim=16,
        out_dir=tmp_path / "arch",
        epochs=1,
        **{k: v for k, v in _FAST_ARCH.items() if k != "hidden_dims"},
    )
    assert from_arch.hidden_dims == [64, 32, 32]
    assert from_arch.model is None
    assert from_arch.gene_order is None

    defaulted = CloneTrastTrainer(
        ad_train,
        hidden_dims=None,
        encoder_depth=None,
        encoder_width=None,
        out_dir=tmp_path / "default",
        epochs=1,
        **{k: v for k, v in _FAST_ARCH.items() if k != "hidden_dims"},
    )
    assert defaulted.hidden_dims == [512, 256]

    with pytest.raises(ValueError, match="contrastive_loss_variant"):
        CloneTrastTrainer(ad_train, contrastive_loss_variant="nope", out_dir=tmp_path, **_FAST_ARCH)
    with pytest.raises(ValueError, match="dropout"):
        CloneTrastTrainer(ad_train, dropout=1.5, out_dir=tmp_path, **_FAST_ARCH)
    with pytest.raises(ValueError, match="weight_decay"):
        CloneTrastTrainer(ad_train, weight_decay=-0.1, out_dir=tmp_path, **_FAST_ARCH)


def test_trainer_missing_clone_id_raises(tmp_path: Path):
    ad_train, _ = _tiny_split_shared_genes()
    del ad_train.obs["clone_id"]
    trainer = CloneTrastTrainer(ad_train, adata_val=None, out_dir=tmp_path, epochs=1, **_FAST_ARCH)
    with pytest.raises(ValueError, match="clone_id_key"):
        trainer.train()


def test_trainer_patient_wise_requires_patient_key(tmp_path: Path):
    ad_train, _ = _tiny_split_shared_genes()
    del ad_train.obs["patient"]
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=None,
        out_dir=tmp_path,
        epochs=1,
        patient_wise_batching=True,
        **_FAST_ARCH,
    )
    with pytest.raises(ValueError, match="patient_wise_batching"):
        trainer.train()


def test_trainer_val_gene_mismatch_raises(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    ad_val = ad_val[:, :8].copy()
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        patient_key="patient",
        **_FAST_ARCH,
    )
    with pytest.raises(ValueError, match="gene dimension"):
        trainer.train()


def test_trainer_val_missing_patient_key_raises(tmp_path: Path):
    ad_train, ad_val = _tiny_split_shared_genes()
    del ad_val.obs["patient"]
    trainer = CloneTrastTrainer(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path,
        epochs=1,
        patient_key="patient",
        **_FAST_ARCH,
    )
    with pytest.raises(ValueError, match="Validation metric requires"):
        trainer.train()


def test_clone_level_spearman_and_roc_auc_edges():
    clone_ids = np.array(["a", "a", "b", "b"])
    y_true = np.array([1.0, 1.0, 3.0, 3.0], dtype=np.float32)
    y_pred = np.array([0.2, 0.4, 2.5, 2.7], dtype=np.float32)
    spearman = _clone_level_spearman(clone_ids, y_true, y_pred)
    auc = _clone_level_roc_auc(clone_ids, y_true, y_pred)
    assert spearman == pytest.approx(1.0)
    assert 0.0 <= auc <= 1.0

    tied_pred = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.isnan(_clone_level_spearman(clone_ids, y_true, tied_pred))
    auc_ties = _clone_level_roc_auc(clone_ids, y_true, tied_pred)
    assert 0.0 <= auc_ties <= 1.0

    one_clone = np.array(["only", "only"])
    assert np.isnan(_clone_level_spearman(one_clone, y_true[:2], y_pred[:2]))
    assert np.isnan(_clone_level_roc_auc(one_clone, y_true[:2], y_pred[:2]))

    same_true = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    assert np.isnan(_clone_level_roc_auc(clone_ids, same_true, y_pred))

    with pytest.raises(ValueError, match="equal length"):
        _clone_level_spearman(clone_ids, y_true, y_pred[:2])
    with pytest.raises(ValueError, match="equal length"):
        _clone_level_roc_auc(clone_ids, y_true, y_pred[:2])


def test_size_loss_helpers_single_clone_and_ties():
    pred = torch.tensor([1.0, 2.0])
    true = torch.tensor([1.5, 1.5])
    clone_ids = np.array(["a", "a"])
    assert _batch_clone_means(clone_ids, pred, true) is None
    breakdown = _compute_size_training_loss(
        pred, true, clone_ids, loss_mode="clone_pearson"
    )
    assert breakdown.pearson_loss is None
    assert breakdown.loss.item() == pytest.approx(breakdown.mse.item())

    rank = _pairwise_ranking_loss(torch.tensor([1.0, 2.0]), torch.tensor([5.0, 5.0]))
    assert rank.item() == pytest.approx(0.0)

    two_clones = np.array(["a", "b"])
    with pytest.raises(ValueError, match="Unsupported size_training_loss"):
        _compute_size_training_loss(pred, true, two_clones, loss_mode="not_a_loss")


def test_save_checkpoint_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "nested" / "model.pt"
    _save_checkpoint({"n_genes": 3}, path)
    assert path.is_file()

    def fail_replace(_src, _dst):
        raise OSError("file in use")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="file in use"):
        _save_checkpoint({"n_genes": 4}, path)
    assert list(path.parent.glob("*.tmp")) == []


def test_patient_wise_sampler_skips_leftover_singleton():
    patient_ids = np.array(["a", "a", "b"])
    sampler = PatientWiseBatchSampler(
        patient_ids,
        batch_size=2,
        patients_per_batch=1,
        rng=np.random.default_rng(0),
    )
    batches = list(sampler)
    seen = {idx for batch in batches for idx in batch}
    assert all(len(batch) >= 2 for batch in batches)
    assert len(seen) == 2
    assert seen.issubset({0, 1, 2})


def test_load_model_with_size(tiny_ckpt_with_size: Path):
    pair = load_model(tiny_ckpt_with_size, device="cpu", load_size_model=True)
    assert isinstance(pair, tuple)
    model, size_model = pair
    assert model.n_genes == 5
    assert size_model.n_genes == 5


def test_load_model_size_flag_without_file(tiny_ckpt_dir: Path):
    cfg_path = tiny_ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["has_size_model"] = True
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    model = load_model(tiny_ckpt_dir, device="cpu", load_size_model=True)
    assert not isinstance(model, tuple)


def test_train_size_model_error_paths(tmp_path: Path):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    with pytest.raises(KeyError, match="clone_id_size"):
        bad = ad_train.copy()
        del bad.obs["clone_id_size"]
        ct.tl.train_size_model(bad, out_dir=tmp_path, epochs=1, device="cpu", hidden_dims=[64, 32])
    with pytest.raises(KeyError, match="clone_id"):
        bad = ad_train.copy()
        del bad.obs["clone_id"]
        ct.tl.train_size_model(bad, out_dir=tmp_path, epochs=1, device="cpu", hidden_dims=[64, 32])
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        ct.tl.train_size_model(
            ad_train,
            adata_val=ad_val,
            out_dir=tmp_path,
            epochs=1,
            device="cpu",
            hidden_dims=[64, 32],
            scheduler="nope",
        )
    with pytest.raises(ValueError, match="ranking_loss_weight"):
        ct.tl.train_size_model(
            ad_train,
            adata_val=ad_val,
            out_dir=tmp_path,
            epochs=1,
            device="cpu",
            hidden_dims=[64, 32],
            size_training_loss="mse_ranking",
            ranking_loss_weight=-1.0,
        )


def test_train_size_model_no_val_and_arch_and_plateau(tmp_path: Path):
    ad_train, ad_val = _tiny_split_with_clone_sizes()
    reported: list[float] = []
    model = ct.tl.train_size_model(
        ad_train,
        adata_val=None,
        out_dir=tmp_path / "no_val",
        epochs=1,
        batch_size=16,
        device="cpu",
        encoder_depth=2,
        encoder_width=64,
        min_hidden_dim=32,
        patient_key="patient",
        report_metric=lambda _step, value: reported.append(value),
    )
    assert model is not None
    assert reported
    assert (tmp_path / "no_val" / "size_model.pt").exists()

    ct.tl.train_size_model(
        ad_train,
        adata_val=ad_val,
        out_dir=tmp_path / "plateau",
        epochs=2,
        batch_size=16,
        device="cpu",
        hidden_dims=[64, 32],
        scheduler="plateau",
        scheduler_patience=1,
        patient_key="patient",
        patient_wise_batching=True,
        patients_per_batch=2,
        early_stopping=True,
        patience=1,
        warmup_epochs=0,
    )
    assert (tmp_path / "plateau" / "size_loss_history.csv").exists()
