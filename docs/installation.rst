Installation Guide
==================

.. note::

   This guide covers two installation methods:

   - **Installing the Published Package** (recommended for most users):
     Use this if you want to use CloneTrast for analysis or inference with the pre-trained models.
   - **Local Development Setup** (for contributors/developers):
     Use this if you want to contribute to CloneTrast or work with the latest source code from GitHub.

CloneTrast requires **Python 3.10-3.13**. GPU support is optional but recommended for large datasets and for training.
The default PyTorch wheel from PyPI is typically CPU-oriented. For NVIDIA GPU acceleration, install a CUDA-enabled
PyTorch build after installing CloneTrast (see each option below).

Installing the Published Package
--------------------------------

Option 1: Using a virtual environment with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Create and activate a virtual environment**

.. code-block:: bash

   # Create virtual environment (use python3.13, python3.12, python3.11, or python3.10)
   python3.12 -m venv clonetrast-env

   # Activate it:
   # On macOS/Linux:
   source clonetrast-env/bin/activate

   # On Windows (Command Prompt):
   # clonetrast-env\Scripts\activate.bat

   # On Windows (PowerShell):
   # clonetrast-env\Scripts\Activate.ps1

**2. Install CloneTrast**

.. code-block:: bash

   pip install clonetrast

**3. Optional: NVIDIA GPU with CUDA**

Install a CUDA-enabled PyTorch build that matches your driver.
CUDA 12.8 is shown as an example; pick the index URL for your CUDA version from the
`PyTorch install page <https://pytorch.org/get-started/locally/>`__.

**Pay attention**: This installtion may require deleteing the existing CPU-only PyTorch installation in advance.

**Important note:** Practically, for inference alone this step is not necessary, and is more of a concern for training and development.

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu128

Option 2: Using Conda with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Create and activate a conda environment**

.. code-block:: bash

   # Create conda environment (use python=3.13, python=3.12, python=3.11, or python=3.10)
   conda create -n clonetrast-env python=3.12

   # Activate it:
   conda activate clonetrast-env

**2. Install CloneTrast**

.. code-block:: bash

   pip install clonetrast

**3. Optional: NVIDIA GPU with CUDA**

Pay attention that this installtion may require deleteing the existing CPU-only PyTorch installation in advance.

**Important note:** Practically, for inference alone this step is not necessary, and is more of a concern for training and development.

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu128

Option 3: Using the uv package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Install uv**

`uv <https://docs.astral.sh/uv/>`__ is a fast Python package manager.

→ `Install uv <https://docs.astral.sh/uv/getting-started/installation/>`__

**2. Create and activate a virtual environment**

.. code-block:: bash

   # Create virtual environment (use --python 3.13, 3.12, 3.11, or 3.10)
   uv venv --python 3.12

   # Activate it:
   # On macOS/Linux:
   source .venv/bin/activate

   # On Windows (Command Prompt):
   # .venv\Scripts\activate.bat

   # On Windows (PowerShell):
   # .venv\Scripts\Activate.ps1

**3. Install CloneTrast**

.. code-block:: bash

   uv pip install clonetrast

**4. Optional: NVIDIA GPU with CUDA**

Pay attention that this installtion may require deleteing the existing CPU-only PyTorch installation in advance.

**Important note:** Practically, for inference alone this step is not necessary, and is more of a concern for training and development.

.. code-block:: bash

   uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

Verify the installation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import clonetrast; print(clonetrast.__version__)"
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"

Development Setup (from Source)
-------------------------------

To work with the latest version on GitHub (for development or contributions):

.. code-block:: bash

   git clone https://github.com/yizhak-lab-ccg/CloneTrast.git
   cd CloneTrast

CloneTrast uses `uv <https://docs.astral.sh/uv/>`__ for fast, reliable dependency management.
Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__ if you do not already have it.

Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root:

.. code-block:: bash

   uv venv --python 3.12

Activate the environment:

.. code-block:: bash

   # macOS / Linux:
   source .venv/bin/activate

   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1

   # Windows Command Prompt:
   .\.venv\Scripts\activate.bat

Install CloneTrast in editable mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the package and common developer dependency groups from the lock file:

.. code-block:: bash

   uv sync --extra docs --group test --group dev

**GPU (CUDA 12.8 example):** after the sync above, install a CUDA-enabled PyTorch build.

Pay attention that this installtion may require deleteing the existing CPU-only PyTorch installation in advance.

**Important note:** Practically, for inference alone this step is not necessary, and is more of a concern for training and development.

.. code-block:: bash

   uv pip install torch --index-url https://download.pytorch.org/whl/cu128

Adjust the PyTorch index URL for your CUDA version (`PyTorch install page <https://pytorch.org/get-started/locally/>`__).

What the extras and groups provide:

* ``docs`` (extra): Sphinx stack for building this documentation
* ``test`` (group): pytest and coverage
* ``dev`` (group): pre-commit and packaging tools

For a minimal editable install without docs/tests:

.. code-block:: bash

   uv sync

Verify
~~~~~~

.. code-block:: bash

   python -c "import clonetrast; print(clonetrast.__version__)"
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"

Or from the repo root without activating the venv:

.. code-block:: bash

   uv run python -c "import clonetrast; print(clonetrast.__version__)"

Next steps
----------

* :doc:`workflow_inference` for embedding cells with the pre-trained models
* :doc:`tutorials` for end-to-end notebooks
* :doc:`api/index` for the Python API reference
* :doc:`contributing` for reporting issues and submitting pull requests
