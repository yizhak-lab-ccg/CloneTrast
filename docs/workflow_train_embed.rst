Training your own model (advanced)
===================================

**Pay attention**: We highly recommend using the provided pre-trained models rather than training your own models.

This workflow is provided for advanced users who want to train on their own labeled data instead of using the public Figshare checkpoints.

For most users, start with :doc:`workflow_inference`.

Overview
--------

1. **Prepare** expression (normalize + log1p).
2. **Split** train vs test by patient (``split_train_test``).
3. **Train** the encoder with supervised contrastive loss (``CloneTrastTrainer`` / ``clonetrast.tl.train``).
4. **Embed** held-out or new data with the saved checkpoint (``clonetrast.tl.embed`` with ``use_pretrained=False``).
5. **Visualize** with UMAP or any equivalent method on ``obsm['X_clonetrast']``.

Requirements
------------

* ``obs['clone_id']`` - per-cell **categorical** clone identifier used as supervision for the contrastive loss. These labels are **inferred from scTCR-seq** before training; CloneTrast does **not** take TCR sequences as input, only expression vectors and the ``clone_id`` category.
  
  In our manuscript, each clone was further restricted to a single T-cell subtype (CD8\ :sup:`+`, CD4\ :sup:`+`, Treg, double-negative, or double positive).
* ``obs['patient']`` - required for patient-aware train/test split (see :doc:`data_format`).

Python API
----------

This is a simplified example. In practice, we used Optuna for hyperparameter tuning and training for 100 different trials.

.. code-block:: python

   import clonetrast as ct

   ct.pp.prepare_adata(adata, use_raw=True, target_sum=10_000.0, log1p=True) # insert use_raw = False if using adata.X instead of adata.raw.X
   adata_train, adata_test = ct.pp.split_train_test(
       adata, clone_id_key="clone_id", patient_key="patient", test_fraction=0.2
   )
   ct.tl.train(adata_train, clone_id_key="clone_id", out_dir="clonetrast_out", epochs=100)
   ct.tl.embed(adata_test, "clonetrast_out", use_pretrained=False)
   ct.tl.compute_umap(adata_test, obsm_key="X_clonetrast", umap_key="X_umap_clonetrast")

Supervised contrastive loss
---------------------------

Training optimizes the **SupCon** objective from Khosla et al. (2020) on the projection head. Within each batch, cells with the same **categorical 'clone_id'** are positives (pulled together); all other cells are negatives (pushed apart). **Same-clone cells are trained to sit together in embedding space** - that is the primary objective. Labels are assigned from scTCR-seq clonotype inference **before** training; the network always learns from **expression**, not from TCR sequence identity. Projections are L2-normalized so dot products equal cosine similarity.

Two variants are supported (see :doc:`contrastive_loss` for full notation and formulas):

* ``sup_out`` (API default) - Eq. (2): sum over positives outside the log.
* ``sup_in`` - Eq. (3): sum over positives inside the log.

The public pre-trained contrastive model uses ``sup_in``. That choice came from Optuna hyperparameter tuning that searched over both variants along with temperature and architecture settings (see :doc:`contrastive_loss`).

.. code-block:: python

   # Example of training with sup_out and temperature of 0.25
   ct.tl.train(
       adata_train,
       clone_id_key="clone_id",
       out_dir="clonetrast_out",
       temperature=0.25,
       contrastive_loss_variant="sup_out",
   )

At inference only the **encoder** embedding is used; the projection head is discarded.

Validation and checkpointing
----------------------------

When a validation set is provided (``adata_val`` in the Python API), validation contrastive loss is logged each epoch. Checkpoint selection can use **clonal purity** on encoder embeddings (higher is better) rather than loss alone - see :doc:`evaluation_metrics`. Per-epoch metrics are written to ``loss_history.csv`` in the output directory (see :doc:`output_format`).
