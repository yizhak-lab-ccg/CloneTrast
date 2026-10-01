Tutorials and examples
======================

Step-by-step notebooks to help you get started with CloneTrast.

:doc:`notebooks/data_preparation_tutorial`
   Prepare paired scRNA/TCR-seq data for training: quality control, clonotype
   definition, T-cell subtype classification, clone-size annotation, and export.

:doc:`notebooks/applying_clonetrast_tutorial`
   Load pre-processed scRNA-seq data, embed it with CloneTrast, predict clone
   size, visualize the results (including a comparison to a standard Harmony
   UMAP), and evaluate batch mixing with iLISI.

Running the notebooks locally
-----------------------------

The notebooks live in ``docs/notebooks/`` in the
`GitHub repository <https://github.com/yizhak-lab-ccg/CloneTrast>`_.
To run them, install CloneTrast together with Jupyter in the same environment
and select that environment as the notebook kernel:

.. code-block:: bash

   pip install clonetrast jupyterlab
   jupyter lab docs/notebooks/

.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/data_preparation_tutorial
   notebooks/applying_clonetrast_tutorial
