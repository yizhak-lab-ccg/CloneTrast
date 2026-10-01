CLI reference
=============

The console script is ``clonetrast`` (``clonetrast.cli:main``).

.. code-block:: text

   clonetrast --help

embed
-----

Load a checkpoint directory and embed an ``AnnData`` file; writes UMAP under ``X_umap_clonetrast`` by default.

.. code-block:: text

   clonetrast embed INPUT.h5ad CHECKPOINT_DIR [--out PATH] [--clone-id COL] [--device DEVICE]

* ``CHECKPOINT_DIR`` - directory with ``model.pt``, ``config.json``, and ``gene_order.json``.
* For the **pre-trained contrastive model**, use the cached path after first download via the Python API, e.g. ``~/.cache/clonetrast/pretrained_models/contrastive_model`` (see :doc:`workflow_inference`).

train
-----

Prepare data, split train/test by patient, and train. Requires ``obs['clone_id']``.

.. code-block:: text

   clonetrast train INPUT.h5ad [--clone-id COL] [--out-dir DIR] [--test-fraction F]
       [--epochs N] [--batch-size N] [--lr LR] [--temperature T]
       [--contrastive-loss-variant {sup_out,sup_in}] [--seed S] [--device DEVICE]

* ``INPUT`` - path to ``.h5ad``.
* ``--clone-id`` - ``obs`` column for clones (default: ``clone_id``).
* ``--out-dir`` - output directory (default: ``clonetrast_out``).
* ``--temperature`` - SupCon temperature :math:`\tau` (default ``0.25``); lower values sharpen clone separation.
* ``--contrastive-loss-variant`` - ``sup_out`` for Khosla et al. Eq. (2) (default) or ``sup_in`` for Eq. (3); see :doc:`contrastive_loss`.
* ``--device`` - ``cuda``, ``cpu``, or default auto.

run
---

Full training pipeline: prepare, split, train, embed train and test, write embedded objects.

.. code-block:: text

   clonetrast run INPUT.h5ad [--clone-id COL] [--out-dir DIR] [--test-fraction F]
       [--epochs N] [--batch-size N] [--temperature T]
       [--contrastive-loss-variant {sup_out,sup_in}] [--seed S]

See :doc:`workflow_train_embed` and :doc:`contrastive_loss` for the training workflow and loss definitions.
