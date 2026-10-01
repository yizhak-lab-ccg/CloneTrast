Output and cache layout
========================

Pre-trained model cache
-----------------------

After the first call to ``embed(use_pretrained=True)`` or ``predict_clone_size(use_pretrained=True)``, models are stored under:

``~/.cache/clonetrast/pretrained_models/`` *(or ``CLONETRAST_PRETRAINED_CACHE``)*

Each subdirectory contains:

``contrastive_model/``
   ``model.pt``, ``config.json``, ``gene_order.json`` - contrastive encoder for embeddings.

``clone_size_model/``
   ``size_model.pt``, ``config.json``, ``gene_order.json`` - clone-size prediction model.

Training output directory
-------------------------

After ``clonetrast train`` or ``clonetrast run``, the output directory (e.g. ``clonetrast_out``) typically contains:

``model.pt``
   Trained encoder / contrastive checkpoint.

``config.json``
   Hyperparameters and run configuration (including ``temperature``, ``contrastive_loss_variant``, architecture sizes, and training options).

``loss_history.csv``
   Per-epoch contrastive training loss; when validation data are used, also validation loss and clonal-purity metrics.

``gene_order.json``
   Gene names in model input order (for aligning new data).

``size_model.pt``
   Present if a clone-size head was trained.

In-memory keys
--------------

* ``obsm['X_clonetrast']`` - learned embedding.
* ``obsm['X_umap_clonetrast']`` - optional UMAP (or choose another key via ``compute_umap``).
* ``obs['predicted_clone_size']`` - optional clone-size predictions from ``predict_clone_size``.
