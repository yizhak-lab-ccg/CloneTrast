Tutorials and examples
=======================

We provide step-by-step notebooks to help you get started with CloneTrast.
Install ``.[notebook]`` or ``ipykernel`` and select the project venv as the Jupyter kernel if you want to run them locally from ``notebooks/``.

- :doc:`notebooks/data_preparation_tutorial` - prepare paired scRNA/TCR-seq data for training: quality control, clonotype definition, T-cell subtype classification, clone-size annotation, and export.
- :doc:`notebooks/applying_clonetrast_tutorial` - load pre-processed scRNA-seq data, embed with CloneTrast, predict clone size, visualize results (including comparison to a standard Harmony UMAP), and evaluate batch mixing with iLISI.

.. toctree::
   :maxdepth: 1

   notebooks/data_preparation_tutorial
   notebooks/applying_clonetrast_tutorial