Data format
============

AnnData (``scanpy`` / ``anndata``) is the primary input. Expression should reflect **Raw UMI counts** without any normalization (e.g. ``adata.X``).

Inference (pre-trained models)
-------------------------------

**Required**

Gene expression matrix
   Cells × genes, e.g. raw UMI counts in ``adata.X``. **scTCR-seq is not used at inference**, only scRNA-seq data. Normalize with ``sc.pp.normalize_total(..., target_sum=1e4)`` and ``sc.pp.log1p`` before ``embed`` (see :doc:`workflow_inference`).

``var_names`` - Ensembl IDs
   The public Figshare checkpoints were trained on Ensembl gene IDs (e.g. ``ENSG00000156234``). Input genes are aligned to the training gene list (11,950 genes). If your ``var_names`` use symbols only, map them to Ensembl before embedding.

T cells only
   The model was trained on T cells alone. Ensure your input includes only T cells for the results to be meaningful.

**Optional**

``obs['clone_id']``
   For coloring UMAP plots or computing metrics only (e.g. clonal purity score).
   **This input is not required to run the model** and never passed to the model. If present, these are typically the same categorical labels used during training (derived from scTCR-seq clonotype calling).

``obs['patient']``
   For patient-restricted evaluation metrics (see :doc:`evaluation_metrics`).

Training (custom models)
------------------------

**Required**

``obs['clone_id']`` (or another column name you pass as ``clone_id_key``)
   Per-cell **categorical** clone identifier. Cells with the same value are treated as positives in the supervised contrastive loss (see :doc:`contrastive_loss`). These labels are **inferred from scTCR-seq** (e.g. shared clonotype is reflected in a shared ``clone_id``) **before** training.
   In our manuscript, each clone was further restricted to a single T-cell subtype (CD8\ :sup:`+`, CD4\ :sup:`+`, Treg, double-negative, or double positive). CloneTrast does **not** use TCR sequences, CDR3 amino acids, CDR3 nucleic acids, or any other TCR information as model input. Only the categorical label and the expression matrix.

Gene expression matrix
   Used for training and inference.

Train / test split
~~~~~~~~~~~~~~~~~~

``clonetrast.pp.split_train_test`` assigns **whole patients** to train or test so the same patient never appears in both splits.

* **Required for the default split:** ``obs['patient']`` (or set ``patient_key`` to your column name). Every cell must have a patient id.

Optional metadata
~~~~~~~~~~~~~~~~~

``obs['clone_id_size']``
   If present, can support a complementary clone-size prediction model.

Preprocessing
-------------

``prepare_adata`` applies, in order:

#. Row-normalize each cell to ``target_sum`` total counts (default ``10000``); use ``target_sum=None`` to skip.
#. ``log1p`` on the result (default ``True``).

Sparse matrices are supported and densified internally for this step.

At inference, normalize with Scanpy (``normalize_total`` to 10,000 counts/cell + ``log1p``) before calling ``embed``, matching the README quick start. Alternatively, call ``prepare_adata`` first, or pass raw counts to ``embed`` with ``normalize_raw=True``.

Gene alignment at inference
---------------------------

Checkpoints include ``gene_order.json``. New data are aligned to training genes (missing genes filled with zeros, thus mimicking sequencing drop-outs; extra genes dropped) before embedding.
