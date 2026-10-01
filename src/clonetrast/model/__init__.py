"""CloneTrast encoder and SupCon-style contrastive loss components."""

from .dataset import CloneDataset
from .encoder import (
    CONTRASTIVE_LOSS_VARIANTS,
    CloneSizeModel,
    CloneTrastModel,
    ContrastiveLossVariant,
    contrastive_loss_clone,
)

__all__ = [
    "CONTRASTIVE_LOSS_VARIANTS",
    "CloneDataset",
    "CloneTrastModel",
    "CloneSizeModel",
    "ContrastiveLossVariant",
    "contrastive_loss_clone",
]
