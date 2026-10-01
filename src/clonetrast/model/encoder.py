"""Encoder and supervised contrastive (SupCon-style) loss from clone identities."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ContrastiveLossVariant = Literal["sup_out", "sup_in"]
CONTRASTIVE_LOSS_VARIANTS: tuple[ContrastiveLossVariant, ...] = ("sup_out", "sup_in")


def contrastive_loss_clone(
    z: torch.Tensor,
    clone_ids: torch.Tensor,
    temperature: float = 0.25,
    *,
    variant: ContrastiveLossVariant = "sup_out",
) -> torch.Tensor:
    r"""Supervised contrastive loss from Khosla et al. 2020 (SupCon).

    Positives share the same ``clone_id`` within the batch; negatives are all other
    cells. For anchor *i*, let P(i) be same-clone indices (excluding *i*) and
    A(i) all indices except *i*.

    **sup_out** - Eq. (2), sum over positives outside the log:

    .. math::

        L_{out}^{sup} = \sum_i \frac{-1}{|P(i)|} \sum_{p \in P(i)}
        \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}

    **sup_in** - Eq. (3), sum over positives inside the log:

    .. math::

        L_{in}^{sup} = \sum_i -\log \left\{
        \frac{1}{|P(i)|} \sum_{p \in P(i)}
        \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}
        \right\}

    The returned scalar is the mean over anchors that have at least one positive.

    Parameters
    ----------
    z
        Projections (batch, proj_dim); normalized internally.
    clone_ids
        Per-cell group ids (batch,); same value denotes a positive pair.
    temperature
        Temperature τ scaling dot products.
    variant
        ``"sup_out"`` for Eq. (2) or ``"sup_in"`` for Eq. (3).

    """
    if variant not in CONTRASTIVE_LOSS_VARIANTS:
        raise ValueError(
            f"variant must be one of {CONTRASTIVE_LOSS_VARIANTS}, got {variant!r}"
        )

    batch_size = z.size(0)
    # L2-normalize so z_i · z_j equals cosine similarity (Khosla et al. 2020).
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature  # (B, B), entries in [-1/tau, 1/tau]

    clone_ids = clone_ids.view(-1, 1)
    pos_mask = (clone_ids == clone_ids.t()).float()
    pos_mask.fill_diagonal_(0)

    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    has_pos = pos_mask.sum(dim=1) > 0

    mask_off_diag = 1.0 - torch.eye(batch_size, device=z.device, dtype=sim.dtype)
    exp_sim = torch.exp(sim) * mask_off_diag
    log_denom = torch.log(exp_sim.sum(dim=1).clamp(min=1e-8))

    if variant == "sup_out":
        # Eq. (2): average of per-positive log-softmax terms.
        log_prob = sim - log_denom.unsqueeze(1)
        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / n_pos
    else:
        # Eq. (3): mean of per-positive softmax probs, then -log
        # (equivalent to paper form; shared denom factors out of the sum).
        pos_exp_mean = (exp_sim * pos_mask).sum(dim=1) / n_pos
        loss_per_anchor = -torch.log(pos_exp_mean.clamp(min=1e-8)) + log_denom

    if not has_pos.any():
        return 0.0 * z.sum()
    return loss_per_anchor[has_pos].mean()


class MLP(nn.Module):
    """Simple MLP block: Linear -> BatchNorm -> ReLU -> Dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear, batch-norm, ReLU, and dropout."""
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        return self.dropout(x)


class CloneTrastModel(nn.Module):
    """Encoder + projection head for clone contrastive learning.

    Encoder maps gene expression to embedding; projection head maps embedding
    to space where contrastive loss is applied.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 128,
        projection_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        dims = [n_genes] + hidden_dims + [embedding_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(MLP(dims[i], dims[i + 1], dropout=dropout))
        self.encoder = nn.Sequential(*layers)

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self._embedding_dim = embedding_dim
        self._projection_dim = projection_dim
        self._n_genes = n_genes

    @property
    def embedding_dim(self) -> int:
        """Return the encoder embedding dimension."""
        return self._embedding_dim

    @property
    def n_genes(self) -> int:
        """Return the number of input genes."""
        return self._n_genes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, projection) for input expression (batch, n_genes).

        The projection is L2-normalized to the unit hypersphere, as required by the
        SupCon loss (Khosla et al. 2020); :func:`contrastive_loss_clone` applies the
        same normalization again before computing cosine similarities.
        """
        h = self.encoder(x)
        z = F.normalize(self.projection(h), dim=1)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the embedding (no projection head)."""
        return self.encoder(x)

    def get_state_dict_for_inference(self) -> dict[str, Any]:
        """State dict that can be loaded for inference (encoder only if desired)."""
        return self.state_dict()


class CloneSizeModel(nn.Module):
    """Separate model for predicting clone_id_size (clone size) from gene expression.

    This is a regression model that predicts the size of the clone each cell belongs to.
    It runs in parallel with CloneTrastModel during training but is a completely separate model.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        max_log_size: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        dims = [n_genes] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(MLP(dims[i], dims[i + 1], dropout=dropout))
        self.encoder = nn.Sequential(*layers)

        # Output layer: single value (clone size)
        self.output = nn.Linear(dims[-1], 1)
        self._n_genes = n_genes
        self.max_log_size = None if max_log_size is None else float(max_log_size)

    @property
    def n_genes(self) -> int:
        """Return the number of input genes."""
        return self._n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted clone size for input expression (batch, n_genes).

        Expects targets (obs['clone_id_size']) to be log-scaled; predictions are
        in the same log space. No extra scaling is applied.

        Returns
        -------
        Predicted log clone size: (batch,) tensor. Clipped to a minimum of zero
        and, when ``max_log_size`` is set, to a maximum of ``max_log_size``.

        """
        h = self.encoder(x)
        size_pred = self.output(h).squeeze(-1).clamp(min=0.0)
        if self.max_log_size is not None:
            # Straight-through upper cap: values above the cap still receive gradients,
            # otherwise over-predicting cells could never be pulled back down.
            capped = size_pred.clamp(max=self.max_log_size)
            size_pred = size_pred + (capped - size_pred).detach()
        return size_pred
