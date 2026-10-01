Supervised contrastive loss (Khosla et al. 2020)
=================================================

CloneTrast trains an expression encoder with the **Supervised Contrastive (SupCon)** objective from Khosla et al. (2020), *Supervised Contrastive Learning* (`NeurIPS <https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html>`__).

Within each mini-batch, cells that share the same **categorical 'clone_id'** are treated as **positives** (the loss pulls them together in embedding space); all other cells in the batch are **negatives** (pushed apart). **Grouping same-clone cells together is the core training objective.** Labels are assigned from scTCR-seq clonotype inference upstream; the model learns from **expression**, not TCR sequence identity. The loss is computed on the **projection head** outputs (not on the encoder embedding used at inference).

Notation
--------

For a batch of :math:`N` cells, let :math:`\mathbf{z}_i \in \mathbb{R}^d` be the L2-normalized projection of cell :math:`i` (CloneTrast normalizes internally so :math:`\mathbf{z}_i \cdot \mathbf{z}_j` equals cosine similarity).

For each anchor index :math:`i`:

* :math:`P(i) = \{ p \neq i \mid \text{clone_id}_p = \text{clone_id}_i \}` - same-clone positives in the batch (excluding the anchor itself).
* :math:`A(i) = \{ a \neq i \}` - all other cells in the batch (negatives and positives).

Temperature :math:`\tau > 0` scales similarities (dot products) before the softmax. Lower :math:`\tau` sharpens the similarity distribution and tends to increase separation between clones. The temperature used by the public pre-trained model was selected together with the loss variant during Optuna hyperparameter search (see `Hyperparameter selection`_ below).

Loss variants
-------------

Khosla et al. define two supervised contrastive variants. CloneTrast exposes them as ``contrastive_loss_variant``:

``sup_out`` - Eq. (2), API default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sum over positives **outside** the logarithm:

.. math::

   \mathcal{L}_{\mathrm{out}}^{\mathrm{sup}}
   = \sum_i \frac{-1}{|P(i)|} \sum_{p \in P(i)}
   \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}
   {\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}

Each positive contributes its own log-softmax term; the anchor loss is the average over :math:`P(i)`.

``sup_in`` - Eq. (3), used by the pre-trained models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sum over positives **inside** the logarithm:

.. math::

   \mathcal{L}_{\mathrm{in}}^{\mathrm{sup}}
   = \sum_i -\log \left\{
   \frac{1}{|P(i)|} \sum_{p \in P(i)}
   \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}
   {\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}
   \right\}

This variant averages the per-positive softmax probabilities before taking the log. In practice :math:`\mathcal{L}_{\mathrm{out}}^{\mathrm{sup}} \geq \mathcal{L}_{\mathrm{in}}^{\mathrm{sup}}` for the same batch (Jensen's inequality).

Hyperparameter selection
------------------------

Both ``sup_out`` and ``sup_in`` are valid SupCon formulations and are exposed in the API.
The loss variant chosen for the final (public pre-trained) model was determined following hyperparameter optimization using `Optuna <https://optuna.org/>`__.
It was treated as a searchable hyperparameter together with temperature, encoder depth/width, and learning-rate schedule settings.
Please refer to our manuscript (Table S2) for the full list of hyperparameters and their ranges.

The Optuna study maximized a validation embedding metric (clonal purity) across trials.
The winning configuration used ``contrastive_loss_variant="sup_in"`` (with the corresponding temperature and architecture settings), and that is what the public pre-trained contrastive model was trained with.

When calling ``ct.tl.train`` / ``clonetrast train`` without specifying a variant, the API still defaults to ``sup_out``.
To match the optimized manuscript model, set ``contrastive_loss_variant="sup_in"``.

Batch reduction
---------------

The formulas above sum over all anchors :math:`i`. CloneTrast returns the **mean over anchors that have at least one positive** in the batch (:math:`|P(i)| > 0`). Anchors whose clone appears only once in the batch contribute nothing (no within-batch positive pairs).

Model architecture
------------------

1. **Encoder** - MLP on log-normalized expression → embedding :math:`\mathbf{h}_i` (stored as ``X_clonetrast`` at inference).
2. **Projection head** - small MLP on :math:`\mathbf{h}_i` → :math:`\mathbf{z}_i`, L2-normalized to the unit hypersphere.
3. **Loss** - one of the variants above on :math:`\{\mathbf{z}_i\}`; the projection head is discarded after training.

Reference
---------

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., and Krishnan, D. (2020). Supervised Contrastive Learning. Advances in Neural Information Processing Systems 33, 18661–18673.