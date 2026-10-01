"""Command-line interface for CloneTrast."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Run the CloneTrast command-line interface."""
    parser = argparse.ArgumentParser(
        description="CloneTrast: supervised contrastive (SupCon) learning for clonal-functional architecture from gene expression."
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # Train
    p_train = sub.add_parser("train", help="Prepare data, split train/test, and train the model")
    p_train.add_argument("input", type=Path, help="Path to AnnData (e.g. .h5ad)")
    p_train.add_argument("--clone-id", default="clone_id", help="obs column for clone_id (default: clone_id)")
    p_train.add_argument("--out-dir", type=Path, default=Path("clonetrast_out"), help="Output directory")
    p_train.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of clones for test set")
    p_train.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--temperature", type=float, default=0.25, help="Contrastive loss temperature (lower = sharper clone separation)")
    p_train.add_argument(
        "--contrastive-loss-variant",
        choices=["sup_out", "sup_in"],
        default="sup_out",
        help='SupCon loss variant: sup_out = Khosla et al. Eq. (2), sup_in = Eq. (3)',
    )
    p_train.add_argument("--seed", type=int, default=0, help="Random seed")
    p_train.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto)")

    # Embed
    p_embed = sub.add_parser("embed", help="Compute embeddings for (test) data using a trained model")
    p_embed.add_argument("input", type=Path, help="Path to AnnData (.h5ad)")
    p_embed.add_argument("checkpoint_dir", type=Path, help="Path to checkpoint directory (from train)")
    p_embed.add_argument("--out", type=Path, default=None, help="Save AnnData with embeddings to this path")
    p_embed.add_argument("--clone-id", default="clone_id", help="obs column for clone_id (metadata only)")
    p_embed.add_argument("--device", default=None, help="Device: cuda or cpu")

    # Full pipeline (train + embed train & test + UMAP)
    p_run = sub.add_parser("run", help="Full pipeline: prepare, split, train, embed, and UMAP")
    p_run.add_argument("input", type=Path, help="Path to AnnData (.h5ad)")
    p_run.add_argument("--clone-id", default="clone_id", help="obs column for clone_id")
    p_run.add_argument("--out-dir", type=Path, default=Path("clonetrast_out"), help="Output directory")
    p_run.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of clones for test set")
    p_run.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p_run.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p_run.add_argument("--temperature", type=float, default=0.25, help="Contrastive loss temperature (lower = sharper clone separation)")
    p_run.add_argument(
        "--contrastive-loss-variant",
        choices=["sup_out", "sup_in"],
        default="sup_out",
        help='SupCon loss variant: sup_out = Khosla et al. Eq. (2), sup_in = Eq. (3)',
    )
    p_run.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.command == "train":
        _run_train(args)
    elif args.command == "embed":
        _run_embed(args)
    elif args.command == "run":
        _run_full(args)
    else:
        parser.print_help()
        return


def _run_train(args) -> None:
    import anndata as ad

    from .pp import prepare_adata, split_train_test
    from .tl import CloneTrastTrainer

    adata = ad.read_h5ad(args.input)
    if "clone_id" not in adata.obs.columns and args.clone_id not in adata.obs.columns:
        raise SystemExit(f"Expected obs['{args.clone_id}']. Add clone_id to your AnnData.")
    prepare_adata(adata, copy=False)
    adata_train, adata_test = split_train_test(
        adata, clone_id_key=args.clone_id, test_fraction=args.test_fraction, seed=args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adata_train.write_h5ad(args.out_dir / "adata_train.h5ad")
    adata_test.write_h5ad(args.out_dir / "adata_test.h5ad")
    trainer = CloneTrastTrainer(
        adata_train,
        clone_id_key=args.clone_id,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        contrastive_loss_variant=args.contrastive_loss_variant,
        seed=args.seed,
        device=args.device,
    )
    trainer.train()
    print("Done. Train and test AnnData saved to", args.out_dir)


def _run_embed(args) -> None:
    import anndata as ad

    from .pp import prepare_adata
    from .tl import compute_umap, embed

    adata = ad.read_h5ad(args.input)
    prepare_adata(adata, copy=False)
    embed(adata, args.checkpoint_dir, device=args.device, use_pretrained=False)
    compute_umap(adata, obsm_key="X_clonetrast", umap_key="X_umap_clonetrast")
    out_path = args.out or (args.checkpoint_dir / "adata_embedded.h5ad")
    adata.write_h5ad(out_path)
    print("Embeddings and UMAP saved. AnnData written to", out_path)


def _run_full(args) -> None:
    import anndata as ad

    from .pp import prepare_adata, split_train_test
    from .tl import CloneTrastTrainer, compute_umap, embed

    adata = ad.read_h5ad(args.input)
    if args.clone_id not in adata.obs.columns:
        raise SystemExit(f"Expected obs['{args.clone_id}']. Add clone_id to your AnnData.")
    prepare_adata(adata, copy=False)
    adata_train, adata_test = split_train_test(
        adata, clone_id_key=args.clone_id, test_fraction=args.test_fraction, seed=args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adata_train.write_h5ad(args.out_dir / "adata_train.h5ad")
    adata_test.write_h5ad(args.out_dir / "adata_test.h5ad")

    trainer = CloneTrastTrainer(
        adata_train,
        clone_id_key=args.clone_id,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        contrastive_loss_variant=args.contrastive_loss_variant,
        seed=args.seed,
    )
    trainer.train()

    for name, sub in [("train", adata_train), ("test", adata_test)]:
        embed(sub, args.out_dir, device=None, use_pretrained=False)
        compute_umap(sub, obsm_key="X_clonetrast", umap_key="X_umap_clonetrast")
        sub.write_h5ad(args.out_dir / f"adata_{name}_embedded.h5ad")

    print("Done. Outputs in", args.out_dir)
