Evaluation metrics
===================

These metrics are **optional** at inference. They require ``clone_id`` labels in ``adata.obs``. Use them to assess embedding quality after running :doc:`workflow_inference`, or during custom training validation.

Clonal purity score
-------------------

``clonetrast.tl.clone_purity`` measures how often a cell's *k* nearest neighbors in embedding space share the same ``clone_id`` (clonal purity score). Neighbors are found with **cosine distance** (sklearn ``NearestNeighbors`` with ``metric="cosine"``), which matches the cosine-similarity objective used during contrastive training. You do **not** need to L2-normalize embeddings before evaluation (see `Cosine distance and normalization`_ below). This also applies to ``adata.obsm["X_clonetrast"]``; inference stores the **encoder** output, which is not L2-normalized (only the training-time projection head is), and to other embeddings you pass via ``obsm_key`` (e.g. PCA). When ``patient_ids`` are provided, neighbors are restricted to the same patient so purity is not inflated by mixing patients.

Cosine distance and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per scikit-learn, ``metric="cosine"`` uses **cosine distance** (``1 - cosine similarity``), where similarity is the normalized dot product:

.. math::

   \frac{\langle x, y \rangle}{\|x\|\,\|y\|}

i.e. the angle between vectors, not their magnitudes. Neighbor rankings are therefore unchanged by per-vector scaling. See the sklearn `cosine_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html>`_ and `cosine similarity <https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity>`_ documentation.

``clonetrast.tl.evaluate_embeddings`` wraps purity (and related outputs) for a full ``AnnData`` object given ``obsm_key`` and optional ``patient_key``.

Typical usage after embedding:

.. code-block:: python

   import clonetrast as ct

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

From per-patient to overall score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``patient_key`` is set, purity is computed **within each patient** on the shared joint embedding (k-NN never crosses patients), then aggregated:

1. **Per cell:** for each cell from a clone with at least ``min_clone_size`` cells in that patient, take the fraction of its *k* same-patient neighbors that share its ``clone_id``.
2. **Per clone (within patient):** average those cell-level fractions so each clone contributes equally.
3. **Per patient:** average the per-clone means to get one purity value in ``purity_per_patient``.
4. **Overall (``out["purity"]``):** average the per-patient scores. With ``weight_by_cells=True`` (as in the example above), patients are weighted by cell count, so low-cell-count patients contribute less to the overall score and their (often higher) impurity does not dominate the aggregate; otherwise each patient has equal weight.

Without ``patient_key``, steps 1-2 run once over all cells (global k-NN), and the overall score is the mean of per-clone means.

Use these metrics to compare embeddings or to monitor validation during training (see source in ``tl/train.py`` for how validation uses purity).
