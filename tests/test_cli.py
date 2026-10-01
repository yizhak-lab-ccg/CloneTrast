"""CLI smoke and wiring tests (heavy work mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clonetrast import cli


def test_cli_help_exits_zero():
    old = sys.argv[:]
    try:
        sys.argv = ["clonetrast", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 0
    finally:
        sys.argv = old


def test_cli_no_command_prints_help(capsys):
    old = sys.argv[:]
    try:
        sys.argv = ["clonetrast"]
        cli.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "train" in out.lower() or "usage" in out.lower()


def test_main_dispatches_train(monkeypatch: pytest.MonkeyPatch):
    called: list[object] = []

    def fake_train(args):
        called.append(args)

    monkeypatch.setattr(cli, "_run_train", fake_train)
    old = sys.argv[:]
    try:
        sys.argv = ["clonetrast", "train", "data.h5ad", "--epochs", "2"]
        cli.main()
    finally:
        sys.argv = old
    assert len(called) == 1
    assert called[0].epochs == 2
    assert called[0].input == Path("data.h5ad")


def test_main_dispatches_embed(monkeypatch: pytest.MonkeyPatch):
    called: list[object] = []
    monkeypatch.setattr(cli, "_run_embed", lambda args: called.append(args))
    old = sys.argv[:]
    try:
        sys.argv = ["clonetrast", "embed", "data.h5ad", "ckpt", "--out", "out.h5ad"]
        cli.main()
    finally:
        sys.argv = old
    assert len(called) == 1
    assert called[0].checkpoint_dir == Path("ckpt")
    assert called[0].out == Path("out.h5ad")


def test_main_dispatches_run(monkeypatch: pytest.MonkeyPatch):
    called: list[object] = []
    monkeypatch.setattr(cli, "_run_full", lambda args: called.append(args))
    old = sys.argv[:]
    try:
        sys.argv = ["clonetrast", "run", "data.h5ad", "--seed", "7"]
        cli.main()
    finally:
        sys.argv = old
    assert len(called) == 1
    assert called[0].seed == 7


def _adata_with_columns(columns: list[str]) -> MagicMock:
    adata = MagicMock()
    adata.obs.columns = columns
    return adata


def test_run_train_wires_prepare_split_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adata = _adata_with_columns(["clone_id"])
    adata_train = MagicMock()
    adata_test = MagicMock()
    trainer_instance = MagicMock()
    trainer_cls = MagicMock(return_value=trainer_instance)

    monkeypatch.setattr("anndata.read_h5ad", lambda path: adata)
    monkeypatch.setattr("clonetrast.pp.prepare_adata", MagicMock())
    monkeypatch.setattr(
        "clonetrast.pp.split_train_test",
        MagicMock(return_value=(adata_train, adata_test)),
    )
    monkeypatch.setattr("clonetrast.tl.CloneTrastTrainer", trainer_cls)

    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        input=tmp_path / "data.h5ad",
        clone_id="clone_id",
        out_dir=out_dir,
        test_fraction=0.2,
        epochs=3,
        batch_size=16,
        lr=1e-3,
        temperature=0.25,
        contrastive_loss_variant="sup_out",
        seed=0,
        device="cpu",
    )
    cli._run_train(args)

    assert out_dir.is_dir()
    adata_train.write_h5ad.assert_called_once_with(out_dir / "adata_train.h5ad")
    adata_test.write_h5ad.assert_called_once_with(out_dir / "adata_test.h5ad")
    trainer_cls.assert_called_once()
    kwargs = trainer_cls.call_args.kwargs
    assert kwargs["epochs"] == 3
    assert kwargs["device"] == "cpu"
    assert kwargs["out_dir"] == out_dir
    trainer_instance.train.assert_called_once()


def test_run_train_missing_clone_id_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("anndata.read_h5ad", lambda path: _adata_with_columns(["patient"]))
    monkeypatch.setattr("clonetrast.pp.prepare_adata", MagicMock())
    args = SimpleNamespace(
        input=tmp_path / "data.h5ad",
        clone_id="clone_id",
        out_dir=tmp_path / "out",
        test_fraction=0.2,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        temperature=0.25,
        contrastive_loss_variant="sup_out",
        seed=0,
        device=None,
    )
    with pytest.raises(SystemExit, match="clone_id"):
        cli._run_train(args)


def test_run_embed_default_and_custom_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adata = MagicMock()
    monkeypatch.setattr("anndata.read_h5ad", lambda path: adata)
    prepare = MagicMock()
    embed = MagicMock()
    compute_umap = MagicMock()
    monkeypatch.setattr("clonetrast.pp.prepare_adata", prepare)
    monkeypatch.setattr("clonetrast.tl.embed", embed)
    monkeypatch.setattr("clonetrast.tl.compute_umap", compute_umap)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    args = SimpleNamespace(
        input=tmp_path / "data.h5ad",
        checkpoint_dir=ckpt,
        out=None,
        clone_id="clone_id",
        device="cpu",
    )
    cli._run_embed(args)
    prepare.assert_called_once_with(adata, copy=False)
    embed.assert_called_once_with(adata, ckpt, device="cpu", use_pretrained=False)
    compute_umap.assert_called_once()
    adata.write_h5ad.assert_called_once_with(ckpt / "adata_embedded.h5ad")

    adata.reset_mock()
    custom_out = tmp_path / "custom.h5ad"
    args.out = custom_out
    cli._run_embed(args)
    adata.write_h5ad.assert_called_once_with(custom_out)


def test_run_full_wires_train_and_embed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adata = _adata_with_columns(["clone_id"])
    adata_train = MagicMock()
    adata_test = MagicMock()
    trainer_instance = MagicMock()
    trainer_cls = MagicMock(return_value=trainer_instance)
    embed = MagicMock()
    compute_umap = MagicMock()

    monkeypatch.setattr("anndata.read_h5ad", lambda path: adata)
    monkeypatch.setattr("clonetrast.pp.prepare_adata", MagicMock())
    monkeypatch.setattr(
        "clonetrast.pp.split_train_test",
        MagicMock(return_value=(adata_train, adata_test)),
    )
    monkeypatch.setattr("clonetrast.tl.CloneTrastTrainer", trainer_cls)
    monkeypatch.setattr("clonetrast.tl.embed", embed)
    monkeypatch.setattr("clonetrast.tl.compute_umap", compute_umap)

    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        input=tmp_path / "data.h5ad",
        clone_id="clone_id",
        out_dir=out_dir,
        test_fraction=0.25,
        epochs=2,
        batch_size=8,
        temperature=0.1,
        contrastive_loss_variant="sup_in",
        seed=1,
    )
    cli._run_full(args)

    trainer_instance.train.assert_called_once()
    assert embed.call_count == 2
    assert compute_umap.call_count == 2
    adata_train.write_h5ad.assert_any_call(out_dir / "adata_train.h5ad")
    adata_train.write_h5ad.assert_any_call(out_dir / "adata_train_embedded.h5ad")
    adata_test.write_h5ad.assert_any_call(out_dir / "adata_test.h5ad")
    adata_test.write_h5ad.assert_any_call(out_dir / "adata_test_embedded.h5ad")


def test_run_full_missing_clone_id_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("anndata.read_h5ad", lambda path: _adata_with_columns(["patient"]))
    args = SimpleNamespace(
        input=tmp_path / "data.h5ad",
        clone_id="clone_id",
        out_dir=tmp_path / "out",
        test_fraction=0.2,
        epochs=1,
        batch_size=16,
        temperature=0.25,
        contrastive_loss_variant="sup_out",
        seed=0,
    )
    with pytest.raises(SystemExit, match="clone_id"):
        cli._run_full(args)
