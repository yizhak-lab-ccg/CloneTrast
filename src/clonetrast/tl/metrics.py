"""Metrics for CloneTrast: KNN-based clone purity on a joint cell embedding."""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def clone_purity(
    embeddings: np.ndarray,
    clone_ids: np.ndarray,
    k: int = 10,
    n_jobs: int = 1,
    min_clone_size: int = 2,
    patient_ids: np.ndarray | None = None,
    *,
    return_per_patient: bool = False,
    weight_by_cells: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Compute clone purity: fraction of *k* nearest neighbors that share the same clone.

    For each cell, finds its *k* nearest neighbors (in cosine space) and computes the
    fraction that share the same ``clone_id``. Only cells from clones with at least
    ``min_clone_size`` cells contribute. The scalar metric is the mean of per-clone
    mean purities (equal weight per clone).

    If ``patient_ids`` is provided, neighbors are restricted to cells from the same
    patient as the query cell (KNN computed separately within each patient slice of the
    **same** joint embedding matrix). Overall purity is then aggregated across patients
    (optionally weighted by cell count per patient).

    Parameters
    ----------
    embeddings
        (n_cells, embedding_dim) array - one joint embedding for all cells.
    clone_ids
        (n_cells,) clone identifiers per cell.
    k
        Number of nearest neighbors (excluding self).
    n_jobs
        Parallel jobs for sklearn NearestNeighbors (-1 = all CPUs).
    min_clone_size
        Minimum clone size for a cell to be included.
    patient_ids
        Optional (n_cells,) patient ids; if set, neighbors are same-patient only.
    return_per_patient
        If True and ``patient_ids`` is set, also return per-patient purity dict.
    weight_by_cells
        If True with ``patient_ids``, overall purity weights patients by cell count.

    Returns
    -------
    Scalar mean purity in ``[0, 1]``, or ``(scalar, per_patient_dict)`` when
    ``return_per_patient`` is True.

    """
    n_cells = embeddings.shape[0]
    if n_cells < k + 1:
        return (0.0, {}) if (patient_ids is not None and return_per_patient) else 0.0

    if patient_ids is not None:
        out = _clone_purity_within_patient(
            embeddings,
            clone_ids,
            patient_ids,
            k,
            n_jobs,
            min_clone_size,
            return_per_patient=return_per_patient,
            weight_by_cells=weight_by_cells,
        )
        return out

    unique_clones, clone_counts = np.unique(clone_ids, return_counts=True)
    clone_size_map = dict(zip(unique_clones, clone_counts, strict=True))

    is_included = np.array([clone_size_map[cid] >= min_clone_size for cid in clone_ids])

    if not is_included.any():
        return 0.0

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    ).fit(embeddings)

    _, indices = nbrs.kneighbors(embeddings)
    neighbor_indices = indices[:, 1:]

    purities_per_clone: dict[Any, list[float]] = {}
    for i in range(n_cells):
        if not is_included[i]:
            continue

        neighbor_clone_ids = clone_ids[neighbor_indices[i]]
        same_clone_mask = neighbor_clone_ids == clone_ids[i]
        purity = same_clone_mask.mean()
        cid = clone_ids[i]
        if cid not in purities_per_clone:
            purities_per_clone[cid] = []
        purities_per_clone[cid].append(purity)

    if len(purities_per_clone) == 0:
        return 0.0

    per_clone_means = [np.mean(purities) for purities in purities_per_clone.values()]
    return float(np.mean(per_clone_means))


def _clone_purity_within_patient(
    embeddings: np.ndarray,
    clone_ids: np.ndarray,
    patient_ids: np.ndarray,
    k: int,
    n_jobs: int,
    min_clone_size: int,
    *,
    return_per_patient: bool = False,
    weight_by_cells: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Clone purity with KNN restricted to the same patient (joint embedding)."""
    unique_patients = np.unique(patient_ids)
    patient_purities: dict[str, float] = {}
    patient_n_cells: dict[str, int] = {}

    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        n_patient = int(mask.sum())
        if n_patient < 2:
            continue
        embed_p = embeddings[mask]
        clone_p = clone_ids[mask]

        unique_clones, clone_counts = np.unique(clone_p, return_counts=True)
        clone_size_map = dict(zip(unique_clones, clone_counts, strict=True))
        is_included = np.array([clone_size_map[cid] >= min_clone_size for cid in clone_p])
        if not is_included.any():
            continue

        n_neighbors = min(k + 1, n_patient)
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=n_jobs,
        ).fit(embed_p)
        _, indices = nbrs.kneighbors(embed_p)
        neighbor_indices = indices[:, 1:]

        cell_purities_by_clone: dict[Any, list[float]] = {}
        for i in range(n_patient):
            if not is_included[i]:
                continue
            neighbor_clone_ids = clone_p[neighbor_indices[i]]
            same_clone_mask = neighbor_clone_ids == clone_p[i]
            purity = float(same_clone_mask.mean())
            cid = clone_p[i]
            if cid not in cell_purities_by_clone:
                cell_purities_by_clone[cid] = []
            cell_purities_by_clone[cid].append(purity)

        if not cell_purities_by_clone:
            continue

        per_clone_means = [np.mean(purities) for purities in cell_purities_by_clone.values()]
        patient_purities[str(patient_id)] = float(np.mean(per_clone_means))
        patient_n_cells[str(patient_id)] = n_patient

    if len(patient_purities) == 0:
        return (0.0, {}) if return_per_patient else 0.0
    if weight_by_cells:
        total_weight = sum(patient_n_cells[p] for p in patient_purities)
        overall = (
            sum(patient_purities[p] * patient_n_cells[p] for p in patient_purities)
            / total_weight
            if total_weight > 0
            else 0.0
        )
    else:
        overall = float(np.mean(list(patient_purities.values())))
    if return_per_patient:
        return (overall, patient_purities)
    return overall


def evaluate_embeddings(
    adata: AnnData,
    obsm_key: str = "X_clonetrast",
    clone_id_key: str = "clone_id",
    patient_key: str | None = None,
    k: int = 10,
    n_jobs: int = 1,
    min_clone_size: int = 2,
    *,
    return_purity_per_patient: bool = False,
    weight_by_cells: bool = False,
) -> dict[str, Any]:
    """Evaluate clone purity on the joint embedding stored in ``adata.obsm[obsm_key]``.

    All cells share one embedding matrix (e.g. after :func:`~clonetrast.tl.embedding.embed`).
    If ``patient_key`` is provided, each cell's *k* neighbors are drawn only from cells
    of the same patient - still using this single joint latent space, not separate
    models per patient.

    Parameters
    ----------
    adata
        AnnData with ``obsm[obsm_key]`` and ``obs[clone_id_key]``.
    obsm_key
        Embedding matrix key (cells × dim).
    clone_id_key
        ``obs`` column for clone ids.
    patient_key
        Optional ``obs`` column for patient ids (same-patient KNN restriction).
    k, n_jobs, min_clone_size
        Passed to :func:`clone_purity`.
    return_purity_per_patient
        If True and ``patient_key`` is set, include ``purity_per_patient`` in the result.
    weight_by_cells
        If True with ``patient_key``, aggregate patient scores by cell count.

    Returns
    -------
    ``{"purity": float}``, optionally with ``"purity_per_patient"`` dict.

    """
    if clone_id_key not in adata.obs:
        raise KeyError(f"adata.obs['{clone_id_key}'] not found.")
    if obsm_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{obsm_key}'] not found. Run embed() first.")

    embeddings = np.asarray(adata.obsm[obsm_key])
    clone_ids = np.asarray(adata.obs[clone_id_key].values)

    patient_ids = None
    if patient_key is not None:
        if patient_key not in adata.obs:
            raise KeyError(f"adata.obs['{patient_key}'] not found.")
        patient_ids = np.asarray(adata.obs[patient_key].astype(str))

    out = clone_purity(
        embeddings,
        clone_ids,
        k=k,
        n_jobs=n_jobs,
        min_clone_size=min_clone_size,
        patient_ids=patient_ids,
        return_per_patient=return_purity_per_patient and patient_ids is not None,
        weight_by_cells=weight_by_cells and patient_ids is not None,
    )
    if return_purity_per_patient and patient_ids is not None:
        purity, purity_per_patient = out
        return {"purity": float(purity), "purity_per_patient": purity_per_patient}
    if return_purity_per_patient:
        return {"purity": float(out), "purity_per_patient": {}}
    return {"purity": float(out)}
