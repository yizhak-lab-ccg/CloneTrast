"""CloneTrast: supervised contrastive (SupCon) learning for clonal-functional architecture from gene expression."""

from importlib.metadata import version

from . import pp, tl

__all__ = ["pp", "tl"]

__version__ = version("clonetrast")
