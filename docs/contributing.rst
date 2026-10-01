Contributing to CloneTrast
==========================

We welcome contributions to CloneTrast. This page covers how to report issues and submit pull requests.

Reporting Issues
----------------

When reporting issues, please include:

- Python version
- CloneTrast version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

Contributing Process
--------------------

**Quick start:**

1. **Fork and clone** the repository
2. **Set up the development environment** following the `Development Setup (from Source) <https://clonetrast.readthedocs.io/en/latest/installation.html#development-setup-from-source>`_ instructions
3. **Create a feature branch** and make your changes

**Before submitting your pull request:**

4. **Ensure all tests pass:**

   .. code-block:: bash

      pytest --cov=clonetrast

5. **Add tests** for new functionality
6. **Update documentation** if your changes affect the API or user-facing functionality

**Submit your pull request:**

7. **Create a pull request** with:

   - A clear, descriptive title
   - A short description of the changes
   - A reference to any related issues

8. **Request review** from maintainers

Your pull request will be reviewed and may require changes before being merged.
