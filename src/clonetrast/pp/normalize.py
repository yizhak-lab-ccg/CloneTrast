"""Normalize gene expression and prepare AnnData for CloneTrast training."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray


def prepare_adata(
    adata: AnnData,
    layer: str | None = None,
    use_raw: bool = True,
    target_sum: float | None = 10_000.0,
    log1p: bool = True,
    copy: bool = False,
) -> AnnData:
    """Prepare AnnData with raw UMI counts for CloneTrast (normalize).

    Uses the matrix in `adata.X` or `adata.raw.X` (if use_raw) or the given layer.
    Sparse matrices (e.g. ``scipy.sparse.csr_matrix``) are supported and converted to
    dense internally. Applies in order: (1) row-normalization to target sum per cell
    (e.g. 10k counts), (2) log1p.

    Parameters
    ----------
    adata
        AnnData with gene expression (cells × genes).
    layer
        Layer to use (e.g. 'counts'). If None, uses X or raw.X.
    use_raw
        If True and adata.raw is set, use adata.raw.X; otherwise use adata.X.
    target_sum
        Target total counts per cell (row sum). Each cell is scaled so its sum equals
        this value. Use None to skip row-normalization (e.g. if already normalized).
    log1p
        Apply log1p transform to counts (after row-normalization).
    copy
        Return a copy of adata; otherwise modify in place.

    Returns
    -------
    AnnData with prepared expression in ``adata.X`` (or a copy). If the input
    is an AnnData view (e.g. after subsetting), a copy is made and the
    returned object must be used (e.g. ``adata = prepare_adata(adata, ...)``).

    """
    if copy:
        adata = adata.copy()
    elif adata.is_view:
        # Assigning to adata.X on a view (e.g. after adata[mask]) triggers sparse
        # indexing in the parent and can raise ValueError. Use a full copy.
        adata = adata.copy()

    if layer is not None:
        X = adata.layers[layer]
    elif use_raw and adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X

    # Support sparse (e.g. CSR/CSC) by converting to dense once for normalization
    if hasattr(X, "toarray"):
        X = X.toarray().copy()
    else:
        X = np.asarray(X).copy()

    if X.dtype != np.float32:
        X = X.astype(np.float32)

    # 1. Row-normalize to target sum per cell (e.g. 10k counts)
    if target_sum is not None:
        row_sums = np.array(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums[:, np.newaxis] * target_sum

    if log1p:
        np.log1p(X, out=X)

    adata.X = X

    return adata


def split_train_test(
    adata: AnnData,
    patient_key: str = "patient",
    clone_id_key: str = "clone_id",
    test_fraction: float = 0.2,
    min_cells_per_clone_train: int = 2,
    seed: int = 0,
) -> tuple[AnnData, AnnData]:
    """Split AnnData into train and test by patient so patients do not leak across splits.

    Patients are assigned to train or test; all cells of a patient go to the same split.
    This ensures no patient has cells in both train and test. Among patients eligible
    for test (with clones having at least `min_cells_per_clone_train` cells), a
    fraction are randomly assigned to test.

    Parameters
    ----------
    adata
        AnnData with `obs[patient_key]`, `obs[clone_id_key]`, and prepared expression.
    patient_key
        Key in adata.obs containing patient identifiers. Splitting is by patient.
    clone_id_key
        Key in adata.obs containing clone identifiers (used for eligibility for test).
    test_fraction
        Fraction of eligible patients (not cells) to put in the test set.
    min_cells_per_clone_train
        Minimum cells per clone for a patient to be eligible for test (others train-only).
    seed
        Random seed for reproducible split.

    Returns
    -------
    adata_train, adata_test
        Subset AnnData objects. All cells from a given patient are in one split only.

    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(adata.obs[patient_key].astype(str))
    clone_ids = np.asarray(adata.obs[clone_id_key].astype(str))

    # Cells per clone in one O(n) pass (no set/list of clone IDs; many clones have size 1)
    unique_clones, clone_inverse = np.unique(clone_ids, return_inverse=True)
    cells_per_clone_arr = np.bincount(clone_inverse, minlength=len(unique_clones))
    min_cells = max(2, min_cells_per_clone_train)
    # Boolean per clone index; then index by cell -> O(n), no huge string set
    eligible_clone_flag = cells_per_clone_arr >= min_cells
    cell_has_eligible_clone = eligible_clone_flag[clone_inverse]

    # Eligible patients: those with at least one cell in an eligible clone
    eligible_patients = np.unique(patient_ids[cell_has_eligible_clone])

    n_test_patients = max(1, int(len(eligible_patients) * test_fraction))
    n_test_patients = min(n_test_patients, len(eligible_patients))
    test_patients = set(rng.choice(eligible_patients, size=n_test_patients, replace=False))

    # Vectorized masks (O(n)), avoid Python list over 1M elements
    train_mask = ~np.isin(patient_ids, np.array(list(test_patients)))
    test_mask = ~train_mask

    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    return adata_train, adata_test


def get_expression_matrix(adata: AnnData, use_layer: str = "clonetrast") -> NDArray[np.float32]:
    """Return the expression matrix as a dense float32 array for training/inference."""
    if use_layer in adata.layers:
        X = adata.layers[use_layer]
    else:
        X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def align_genes_to_train(
    adata: AnnData,
    train_var_names: list[str],
    source_layer: str | None = None,
) -> NDArray[np.float32]:
    """Align unseen data to training gene order (before normalization).

    Returns a matrix with columns in the same order as training: re-orders genes to
    match train_var_names, drops genes not in training, and fills missing genes with
    zeros. Column index j in the output corresponds to the model's feature index j
    (i.e. the j-th gene in train_var_names). Use this on raw or un-normalized data
    before applying normalization.

    If ``adata.var_names`` already equals ``train_var_names`` (same length and order),
    the function returns the expression matrix converted to dense ``float32`` without
    rebuilding columns (fast path).

    Parameters
    ----------
    adata
        AnnData with gene expression (cells × genes). var_names define gene names.
    train_var_names
        List of gene names in the order expected by the trained model (e.g. from
        gene_order.json). This order must match the model's feature order.
    source_layer
        Layer to use for expression (e.g. 'counts' for raw counts). If None, uses
        adata.X.

    Returns
    -------
    Matrix of shape (n_cells, len(train_var_names)), float32. Missing genes are 0.
    Column j = model feature j.

    """
    if source_layer is not None and source_layer in adata.layers:
        X = adata.layers[source_layer]
    else:
        X = adata.X

    var_names = list(adata.var_names)
    train_list = list(train_var_names)
    if len(var_names) == len(train_list) and var_names == train_list:
        if hasattr(X, "toarray"):
            X = X.toarray()
        out = np.asarray(X, dtype=np.float32)
        n_cells, n_train = out.shape[0], len(train_list)
        print(
            f"[CloneTrast] Gene alignment: {n_cells} cells × {n_train} training genes | "
            f"matched {n_train}, missing (→0) 0, extra dropped 0 (already aligned)"
        )
        return out

    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    var_index = {g: i for i, g in enumerate(var_names)}

    n_cells = X.shape[0]
    n_train = len(train_var_names)
    var_set = set(var_names)
    train_set = set(train_var_names)
    n_matched = sum(1 for g in train_var_names if g in var_index)
    n_missing = n_train - n_matched
    n_extra = len(var_set - train_set)

    out = np.zeros((n_cells, n_train), dtype=np.float32)
    for j, g in enumerate(train_var_names):
        if g in var_index:
            out[:, j] = X[:, var_index[g]]

    print(
        f"[CloneTrast] Gene alignment: {n_cells} cells × {n_train} training genes | "
        f"matched {n_matched}, missing (→0) {n_missing}, extra dropped {n_extra}"
    )
    return out


def normalize_expression_matrix(
    X: NDArray,
    target_sum: float | None = 10_000.0,
    log1p: bool = True,
) -> NDArray[np.float32]:
    """Normalize an expression matrix the same way as prepare_adata (e.g. for inference).

    Applies in order: (1) row-normalize to target sum per cell, (2) log1p.
    Use after align_genes_to_train when unseen data is provided as raw counts.

    Parameters
    ----------
    X
        Expression matrix (n_cells × n_genes), e.g. raw UMI counts or aligned matrix.
    target_sum
        Target total counts per cell. Use None to skip row-normalization.
    log1p
        Apply log1p after row-normalization.

    Returns
    -------
    Normalized matrix, float32.

    """
    X = np.asarray(X, dtype=np.float32)
    n_cells, n_genes = X.shape
    if target_sum is not None:
        row_sums = np.array(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums[:, np.newaxis] * target_sum
    if log1p:
        np.log1p(X, out=X)
    print(
        f"[CloneTrast] Normalized expression: {n_cells} × {n_genes} "
        f"(target_sum={target_sum}, log1p={log1p})"
    )
    return X
