"""PyTorch Dataset for clone contrastive learning."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CloneDataset(Dataset):
    """Dataset of (expression, clone_id) for contrastive training.

    Expression matrix and clone IDs must be aligned by row (same order as adata.obs).
    Optionally includes clone_id_size for parallel size prediction model.
    """

    def __init__(
        self,
        X: np.ndarray,
        clone_ids: np.ndarray,
        gene_order: np.ndarray | None = None,
        clone_sizes: np.ndarray | None = None,
    ) -> None:
        """Build dataset from expression and clone IDs.

        Parameters
        ----------
        X
            Expression matrix (n_cells, n_genes), float32.
        clone_ids
            1D array of clone identifiers (str or int), length n_cells.
        gene_order
            Optional index array to reorder genes (e.g. for test set to match train).
        clone_sizes
            Optional 1D array of clone sizes (``clone_id_size``), length n_cells.
            If provided, ``__getitem__`` returns ``(x, c_int, size)``; otherwise
            ``(x, c_int)``.

        """
        self.X = np.asarray(X, dtype=np.float32)
        self.clone_ids = np.asarray(clone_ids).ravel()
        if self.X.shape[0] != self.clone_ids.shape[0]:
            raise ValueError("X and clone_ids must have same number of rows (cells).")
        if gene_order is not None:
            self.X = self.X[:, np.asarray(gene_order)]
        self._clone_to_int = self._build_clone_to_int()

        self.clone_sizes = None
        if clone_sizes is not None:
            self.clone_sizes = np.asarray(clone_sizes, dtype=np.float32).ravel()
            if self.clone_sizes.shape[0] != self.X.shape[0]:
                raise ValueError("clone_sizes must have same length as X (number of cells).")

    def _build_clone_to_int(self) -> dict[Any, int]:
        unique = np.unique(self.clone_ids)
        return {c: i for i, c in enumerate(unique)}

    def __len__(self) -> int:
        """Return the number of cells."""
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        """Return ``(expression, clone_id)`` or ``(expression, clone_id, size)`` for cell ``idx``."""
        x = torch.from_numpy(self.X[idx])
        c = self.clone_ids[idx]
        c_int = self._clone_to_int.get(c, -1)
        if c_int == -1:
            self._clone_to_int[c] = c_int = len(self._clone_to_int)

        out: list = [x, c_int]
        if self.clone_sizes is not None:
            size = torch.from_numpy(np.array([self.clone_sizes[idx]], dtype=np.float32))[0]
            out.append(size)
        return tuple(out)
