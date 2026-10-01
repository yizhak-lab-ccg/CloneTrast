"""Compute embeddings and 2D visualization from a trained CloneTrast model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from anndata import AnnData

from ..model import CloneSizeModel
from ..pp.normalize import align_genes_to_train, normalize_expression_matrix
from .pretrained import ensure_clone_size_checkpoint, ensure_contrastive_checkpoint
from .train import load_model


def _ensure_ensembl_ids_if_needed(adata: AnnData, train_var_names: list[str]) -> None:
    """Raise a helpful error when training uses Ensembl IDs but input does not."""
    train_looks_ensembl = any("ENSG" in str(g) for g in train_var_names)
    if not train_looks_ensembl:
        return
    input_looks_ensembl = any("ENSG" in str(g) for g in adata.var_names)
    if input_looks_ensembl:
        return
    raise ValueError(
        "Gene identifiers in `adata.var_names` do not look like Ensembl IDs (missing 'ENSG'), "
        "but the checkpoint was trained on Ensembl IDs. Please provide genes as `ensembl_ids` "
        "(e.g. 'ENSG00000141510') so they can be aligned to the training gene order."
    )


def _align_genes(
    adata: AnnData,
    train_var_names: list[str],
    use_layer: str | None = "clonetrast",
) -> np.ndarray:
    """Return expression matrix aligned to train gene order (missing genes → 0, extra genes dropped).

    If use_layer is None, uses adata.X.
    """
    return align_genes_to_train(adata, train_var_names, source_layer=use_layer)


def _load_size_model_only(checkpoint_dir: Path, device: str | torch.device | None = None) -> CloneSizeModel:
    """Load size model directly from size_model.pt without requiring model.pt."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    size_ckpt_path = checkpoint_dir / "size_model.pt"
    if not size_ckpt_path.exists():
        raise ValueError(f"Size model not found at {size_ckpt_path}.")

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    size_ckpt = torch.load(size_ckpt_path, map_location="cpu", weights_only=True)

    size_hidden_dims = config.get("size_model_hidden_dims", config.get("hidden_dims"))
    size_model = CloneSizeModel(
        n_genes=config["n_genes"],
        hidden_dims=size_hidden_dims,
        max_log_size=config.get("max_log_size"),
    )
    size_model.load_state_dict(size_ckpt["model_state_dict"], strict=True)
    size_model = size_model.to(dev)
    size_model.eval()
    return size_model


def embed(
    adata: AnnData,
    checkpoint_dir: str | Path | None = None,
    obsm_key: str = "X_clonetrast",
    use_layer: str = "clonetrast",
    normalize_raw: bool = False,
    device: str | None = None,
    batch_size: int = 512,
    *,
    use_pretrained: bool = True,
    pretrained_cache_dir: str | Path | None = None,
) -> AnnData:
    """Compute CloneTrast embeddings for adata and store in adata.obsm[obsm_key].

    Unseen data is aligned to training gene order: genes are re-ordered to match
    training, extra genes are dropped, and missing genes are filled with zeros.
    Alignment is done first; if the data are raw counts, set normalize_raw=True to
    apply the same normalization as training (target sum, log1p) after
    alignment. For clone-size prediction use :func:`predict_clone_size`.

    Parameters
    ----------
    adata
        AnnData with expression. For unseen data use a layer with raw counts
        (e.g. 'counts') and set use_layer plus normalize_raw=True, or provide
        already-prepared expression in the same gene order as training.
    checkpoint_dir
        Directory with model.pt, config.json, gene_order.json (e.g. trained output or
        unzipped pretrained bundle). Not required when ``use_pretrained=True`` (default).
    use_pretrained
        If True (default), load (and download if needed) public contrastive weights from
        Figshare into ``pretrained_cache_dir`` (default: ``~/.cache/clonetrast/pretrained_models``),
        under ``.../contrastive_model/``. ``checkpoint_dir`` is ignored when this is True.
    pretrained_cache_dir
        Root cache directory when ``use_pretrained=True``. Overrides the
        ``CLONETRAST_PRETRAINED_CACHE`` environment variable when set.
    obsm_key
        Key in adata.obsm where embedding will be stored.
    use_layer
        Layer or X to use for expression (e.g. 'counts' for raw, 'clonetrast' if prepared).
    normalize_raw
        If True, apply training-style normalization (target sum 10k, log1p)
        to the aligned matrix before encoding. Use when adata contains raw counts.
    device
        'cuda', 'cpu', or None (auto).
    batch_size
        Batch size for forward pass.

    Returns
    -------
    adata (modified in place) with adata.obsm[obsm_key] set to (n_cells, embedding_dim).

    """
    if use_pretrained:
        contrastive_dir = ensure_contrastive_checkpoint(pretrained_cache_dir)
    else:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when use_pretrained=False.")
        contrastive_dir = Path(checkpoint_dir)

    checkpoint_dir = contrastive_dir
    with open(checkpoint_dir / "gene_order.json") as f:
        train_var_names = json.load(f)
    _ensure_ensembl_ids_if_needed(adata, train_var_names)
    print(f"[CloneTrast] Loading expression from layer/X and aligning to {len(train_var_names)} training genes...")
    X = _align_genes(adata, train_var_names, use_layer=use_layer)
    if normalize_raw:
        print("[CloneTrast] Applying normalization (target sum, log1p)...")
        X = normalize_expression_matrix(X)

    print(f"[CloneTrast] Loading model from {checkpoint_dir}...")
    model_result = load_model(checkpoint_dir, device=device, load_size_model=False)
    if isinstance(model_result, tuple):
        model = model_result[0]  # Extract contrastive model from tuple
    else:
        model = model_result
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    n_cells = X.shape[0]
    embeddings = []
    print(f"[CloneTrast] Computing expression embeddings for {n_cells} cells (batch_size={batch_size}, device={device})...")
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch = torch.from_numpy(X[start:end]).to(device)
            h = model.encode(batch)
            embeddings.append(h.cpu().numpy())
    adata.obsm[obsm_key] = np.vstack(embeddings)
    emb_dim = adata.obsm[obsm_key].shape[1]
    print(f"[CloneTrast] Expression embeddings done: shape ({n_cells}, {emb_dim}) → adata.obsm['{obsm_key}']")

    return adata


def predict_clone_size(
    adata: AnnData,
    checkpoint_dir: str | Path | None = None,
    use_layer: str = "clonetrast",
    normalize_raw: bool = False,
    obs_key: str = "predicted_clone_size",
    device: str | None = None,
    batch_size: int = 512,
    *,
    use_pretrained: bool = True,
    pretrained_cache_dir: str | Path | None = None,
) -> AnnData:
    """Predict clone size for each cell and store in adata.obs[obs_key].

    Checkpoint must have been trained with clone_id_size. Predictions are in the
    same log space as the training target (obs['clone_id_size']). Unseen data is
    aligned to training genes (reorder, drop extra, missing → 0); set normalize_raw=True
    if the data are raw counts.

    Parameters
    ----------
    adata
        AnnData with expression (same genes as training, or raw counts with normalize_raw=True).
    checkpoint_dir
        Directory with size_model.pt, config.json, gene_order.json. Not required when
        ``use_pretrained=True``.
    use_pretrained
        If True, load (and download if needed) the public clone-size checkpoint from
        Figshare into ``pretrained_cache_dir`` (default: ``~/.cache/clonetrast/pretrained_models``).
        ``checkpoint_dir`` is ignored when this is True.
    pretrained_cache_dir
        Root cache directory when ``use_pretrained=True``. Overrides ``CLONETRAST_PRETRAINED_CACHE``
        when set.
    use_layer
        Layer or X to use for expression.
    normalize_raw
        If True, apply training-style normalization after gene alignment.
    obs_key
        obs key where predictions will be stored.
    device
        'cuda', 'cpu', or None (auto).
    batch_size
        Batch size for forward pass.

    Returns
    -------
    adata (modified in place) with adata.obs[obs_key] set to predicted log clone size.

    """
    if use_pretrained:
        checkpoint_dir = ensure_clone_size_checkpoint(pretrained_cache_dir)
    else:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when use_pretrained=False.")
        checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "gene_order.json") as f:
        train_var_names = json.load(f)
    _ensure_ensembl_ids_if_needed(adata, train_var_names)
    print("[CloneTrast] Aligning expression to training genes for clone size prediction...")
    X = _align_genes(adata, train_var_names, use_layer=use_layer)
    if normalize_raw:
        X = normalize_expression_matrix(X)
    print(f"[CloneTrast] Loading size model from {checkpoint_dir}...")
    size_model = _load_size_model_only(checkpoint_dir, device=device)
    if device is None:
        device = next(size_model.parameters()).device
    else:
        device = torch.device(device)
    n_cells = X.shape[0]
    preds = []
    print(f"[CloneTrast] Predicting clone size for {n_cells} cells...")
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch = torch.from_numpy(X[start:end]).to(device)
            p = size_model(batch)
            preds.append(p.cpu().numpy())
    adata.obs[obs_key] = np.concatenate(preds)
    print(f"[CloneTrast] Clone size predictions → adata.obs['{obs_key}']")
    return adata


def compute_umap(
    adata: AnnData,
    obsm_key: str = "X_clonetrast",
    umap_key: str = "X_umap_clonetrast",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    metric: str = "cosine",
    random_state: int = 0,
    save_neighbors: bool = True,
) -> AnnData:
    """Compute UMAP 2D from embedding and store in adata.obsm[umap_key].

    Optionally saves the UMAP neighbor graph to adata.obsp and adata.uns in
    scanpy-compatible format, with key names derived from umap_key (e.g.
    umap_key='X_umap_clonetrast' → connectivities_umap_clonetrast,
    distances_umap_clonetrast, neighbors_umap_clonetrast).
    """
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(adata.obsm[obsm_key])
    adata.obsm[umap_key] = X_umap

    if save_neighbors and hasattr(reducer, "graph_") and reducer.graph_ is not None:
        suffix = umap_key.replace("X_", "", 1) if umap_key.startswith("X_") else umap_key
        conn_key = f"connectivities_{suffix}"
        dist_key = f"distances_{suffix}"
        neighbors_key = f"neighbors_{suffix}"
        conn = reducer.graph_.tocsr()
        adata.obsp[conn_key] = conn
        dist = conn.copy()
        dist.data = 1.0 - dist.data
        adata.obsp[dist_key] = dist
        adata.uns[neighbors_key] = {
            "connectivities_key": conn_key,
            "distances_key": dist_key,
            "params": {
                "n_neighbors": n_neighbors,
                "method": "umap",
                "metric": metric,
                "min_dist": min_dist,
                "random_state": random_state,
            },
        }
        print(f"[CloneTrast] Neighbor graph saved to adata.obsp['{conn_key}'], adata.obsp['{dist_key}'], adata.uns['{neighbors_key}']")

    return adata
