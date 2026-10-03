<p align="center">
  <a href="https://github.com/yizhak-lab-ccg/CloneTrast/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/yizhak-lab-ccg/CloneTrast/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/yizhak-lab-ccg/CloneTrast"><img alt="codecov" src="https://img.shields.io/codecov/c/github/yizhak-lab-ccg/CloneTrast?logo=codecov&logoColor=white"></a>
  <a href="https://clonetrast.readthedocs.io/en/latest/"><img alt="Docs" src="https://img.shields.io/readthedocs/clonetrast/latest.svg?logo=readthedocs&logoColor=white"></a>
  <img alt="Python 3.10-3.13" src="https://img.shields.io/badge/Python-3.10--3.13-blue.svg?logo=python&logoColor=white">
  <a href="https://github.com/yizhak-lab-ccg/CloneTrast/blob/main/LICENSE"><img alt="License: MIT" src="https://custom-icon-badges.demolab.com/badge/License-MIT-blue.svg?logo=law-24&logoColor=white"></a>
  <a href="https://pepy.tech/project/clonetrast"><img src="https://static.pepy.tech/badge/clonetrast" alt="Downloads" /></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/yizhak-lab-ccg/CloneTrast/main/docs/_static/images/CloneTrast_logo_for_git.png" alt="CloneTrast" width="400">
</p>

<h1 align="center">
  Revealing pan-cancer clonal niches of T cells from single-cell RNA sequencing using contrastive learning
</h1>

CloneTrast maps **gene expression alone** to a latent space where cells from the same T-cell clone cluster together.
At inference you only need scRNA-seq UMI counts, without paired scTCR-seq. Public **pre-trained model** (hosted on [Figshare](https://doi.org/10.6084/m9.figshare.32228991)) is downloaded automatically on first use, with an additional complementary model used to predict clone size from gene expression alone. Following, you can visualize your data and further explore the clonal functional structure for downstream analysis.

## Graphical description

<p align="center">
  <img src="https://raw.githubusercontent.com/yizhak-lab-ccg/CloneTrast/main/docs/_static/images/figure_1_git.jpg" alt="CloneTrast figure 1 overview" width="800">
</p>

## Clone labels vs. TCR sequences

CloneTrast separates **how models are trained** from **how they are applied**. This distinction is central to what the embedding represents.

### During training: categorical clone labels - not TCR sequence identity

Training requires a per-cell **`clone_id`** column: a **categorical label** that groups cells from the same T-cell clone. Those labels are **inferred from scTCR-seq** (please refer to our manuscript for more details). **Importantly**, CloneTrast is **not** trained on TCR sequences. Supervision comes only from the **clone_id category**.

### At inference: scRNA-seq only - no scTCR-seq

When you embed new cells (with the public pre-trained model), CloneTrast reads **gene expression only**. You do **not** need scTCR-seq, CDR3 sequences, or `clone_id` labels. The model maps each cell’s transcriptome to a fixed-dimensional embedding (`X_clonetrast`).

### What this means for the embedding space

During training, every cell with the same `clone_id` is a positive pair: the contrastive loss rewards them for being located next to each other. At inference, the encoder applies that learned mapping - **cells from the same clone are expected to cluster** in `X_clonetrast` when their expression reflects shared clonal identity.

This specific modeling approach results in a distinct embedding space capturing multiple components of clonal organization:

**(1) Different clones, similar functional states:** Two clones defined by different TCR sequences can be positioned near one another if they share similar functional status.

**(2) Same clone, divergent states:** T cells belonging to the same clone are pulled toward one another despite having diverse differentiation states due to intra-clonal transcriptional heterogeneity.

**(3) Same TCR sequence, different functional states:** Two clones with identical TCR sequences but distinct clonal identities, because they originate from different patients, can be positioned far apart if they exhibit different functional states.

## Training and inference description

<p align="center">
  <img src="https://raw.githubusercontent.com/yizhak-lab-ccg/CloneTrast/main/docs/_static/images/figure_2_git.jpg" alt="CloneTrast figure 2 overview" width="800">
</p>

## Installation

For complete installation instructions including prerequisites, package installation, and development setup, please see the [Installation Guide](https://clonetrast.readthedocs.io/en/latest/installation.html).

## Quick start - inference with pre-trained models

You only need **gene expression (scRNA-seq)**. CloneTrast does **not** use scTCR-seq or TCR sequences at inference - see [Clone labels vs. TCR sequences](#clone-labels-vs-tcr-sequences---what-clonetrast-actually-uses) above. Optional `clone_id` labels are for coloring plots or computing metrics only.

### Pre-trained models

The pre-trained models expect **Ensembl gene IDs** in `adata.var_names` (e.g. `ENSG00000156234`). Input genes are automatically aligned to the training gene list (11,950 genes): missing genes are set to zero (thus mimicking sequencing drop-outs), extra genes are dropped.

### Data format (inference)

- AnnData with gene expression (cells × genes), e.g. raw UMI counts in `adata.X`. Raw counts require subsequent row-normalization and log-transformation, as described below.
- `adata.var_names`: **Ensembl IDs** for compatibility with the public checkpoints.
- **Optional:** `adata.obs['clone_id']` for visualization or evaluation (not required to run the model).
- **Importantly:** the model was trained using T cells alone. It is therefore required to ensure your data includes only T cells as input (for the results to be meaningful).

### Python API

```python
import scanpy as sc
import clonetrast as ct

# These are raw UMI counts
adata = sc.read_h5ad("your_data.h5ad")

# Raw counts should be subsequently normalized
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)

# Applying the contrastive model and creating a two-dimensional representation with UMAP
ct.tl.embed(
    adata,
    use_pretrained=True          # default; downloads contrastive model on first run
)
ct.tl.compute_umap(adata, obsm_key="X_clonetrast", umap_key="X_umap_clonetrast")

# Optional complementary model to predict log clone size per cell
ct.tl.predict_clone_size(adata, use_pretrained=True)

# Visualize
sc.pl.embedding(adata, basis="umap_clonetrast", color=["predicted_clone_size"])
```

Embeddings are stored in `adata.obsm['X_clonetrast']`; UMAP coordinates in `adata.obsm['X_umap_clonetrast']`.

Predicted clone size is stored in `adata.obs['predicted_clone_size']`.

## Tutorial notebooks

Two step-by-step tutorials:

1. **[Data preparation](https://clonetrast.readthedocs.io/en/latest/notebooks/data_preparation_tutorial.html)** - prepare paired scRNA/TCR-seq data for training (QC, T-cell subtyping, clone labeling, export)
2. **[Applying CloneTrast](https://clonetrast.readthedocs.io/en/latest/notebooks/applying_clonetrast_tutorial.html)** - apply CloneTrast to gene expression alone (embed, predict clone size, visualize and compare to a standard UMAP)

## Documentation

[Full documentation](https://clonetrast.readthedocs.io/en/latest/) (includes the [contrastive loss reference](https://clonetrast.readthedocs.io/en/latest/contrastive_loss.html) and [tutorials](https://clonetrast.readthedocs.io/en/latest/tutorials.html)).

## License

This project is licensed under the MIT License - see the LICENSE file for more details.

## Citation

Reference will be added here when available.

## Acknowledgements

CloneTrast's logo, the project's graphical description, and the graphical description of the training and inference, were created with [BioRender.com](https://www.biorender.com/) using a paid license.

</details>

---

<div align="center">
<p><em>This project was created in favor of the scientific community worldwide, with a special dedication to the cancer research community.</em></p>
<p><em>We hope you'll find this repository helpful, and we warmly welcome any requests or suggestions - please don't hesitate to reach out!</em></p>

<a href="https://mapmyvisitors.com/web/1c8l2">
<img src="https://mapmyvisitors.com/map.png?d=dwfyT67_zJfn-BQ-6x-NAaKvey45Vl66GhWHhFcDZHw&cl=ffffff" alt="Visitor Map">
</a>
</div>