Inference with pre-trained models
==================================

For inference, you only need **gene expression (scRNA-seq)**. CloneTrast does **not** use scTCR-seq at inference - see :ref:`clone-labels-vs-tcr` on the home page. Optional ``clone_id`` labels are for coloring plots or computing metrics only.

Pre-trained models
------------------

Two checkpoints are available on `Figshare <https://doi.org/10.6084/m9.figshare.32228991>`__ and cached locally after the first use of the Python API:

.. list-table::
   :header-rows: 1
   :widths: 18 42 22

   * - Model
     - Purpose
     - Cache subdirectory
   * - Contrastive encoder
     - Cell embeddings (``adata.obsm['X_clonetrast']``)
     - ``contrastive_model/``
   * - Clone-size model
     - Per-cell clone size predictions (``adata.obs['predicted_clone_size']``)
     - ``clone_size_model/``

Default cache: ``~/.cache/clonetrast/pretrained_models/``.

Override with the ``CLONETRAST_PRETRAINED_CACHE`` environment variable or ``pretrained_cache_dir=`` in the Python API.

The pre-trained models expect **Ensembl gene IDs** in ``adata.var_names`` (e.g. ``ENSG00000156234``). Input genes are automatically aligned to the training gene list (11,950 genes): missing genes are set to zero (thus mimicking sequencing drop-outs), extra genes are dropped.

Data format (inference)
-----------------------

* AnnData with gene expression (cells × genes), e.g. raw UMI counts in ``adata.X``. Raw counts require subsequent row-normalization and log-transformation, as described below.
* ``adata.var_names``: **Ensembl IDs** for compatibility with the public checkpoints.
* **Optional:** ``adata.obs['clone_id']`` for visualization or evaluation (not required to run the model).
* **Importantly:** the model was trained using T cells alone. It is therefore required to ensure your data includes only T cells as input (for the results to be meaningful).

Python API
----------

.. code-block:: python

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

Embeddings are stored in ``adata.obsm['X_clonetrast']``; UMAP coordinates in ``adata.obsm['X_umap_clonetrast']``.

Predicted clone size is stored in `adata.obs['predicted_clone_size']`.

Clonal purity score evaluation
------------------------------

This evaluation requires the ``clone_id`` column to be present in the AnnData object.

.. code-block:: python

   # Joint embedding in adata.obsm["X_clonetrast"]; optional same-patient k-NN via patient_key
   out = ct.tl.evaluate_embeddings(
       adata,
       obsm_key="X_clonetrast",
       clone_id_key="clone_id",
       patient_key="patient",
       k=10,
       min_clone_size=20,
       return_purity_per_patient=True,
       weight_by_cells=True,
   )
   overall_clonal_purity_score = out["purity"]
   clonal_purity_score_per_patient = out["purity_per_patient"]

Labels are not required to run the model. See :doc:`evaluation_metrics` for metric definitions.

How inference works
-------------------

Inference uses **scRNA-seq only** - no scTCR-seq and no ``clone_id`` (see :ref:`clone-labels-vs-tcr` on the home page).

1. **Gene alignment:** Expression is automatically re-ordered to match the checkpoint's ``gene_order.json``; missing genes are set to zero (thus mimicking sequencing drop-outs), extra genes dropped.
2. **Normalization:** The model expects row-normalized data (to a target sum of 10,000 counts) and log1p-transformed (same as training).
3. **Encoding:** The contrastive encoder maps expression to a fixed-dimensional embedding (``X_clonetrast``).
4. **Clone size (optional):** A separate complementary model predicts log clone size per cell.
5. **Visualization:** UMAP on the embedding reveals clonal niches. Cells from the same clone should cluster.

The encoder was trained with **supervised contrastive (SupCon) loss** from Khosla et al. (2020) on **categorical 'clone_id' labels** (derived from scTCR-seq). At inference only the encoder embedding is used (the projection head is discarded). See :doc:`contrastive_loss` for the objective and formulas.
