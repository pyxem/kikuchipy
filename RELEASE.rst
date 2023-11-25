How to make a new release of ``kikuchipy``
==========================================

kikuchipy's branching model is similar to the Gitflow Workflow (`original blog post
<https://nvie.com/posts/a-successful-git-branching-model/>`__).

kikuchipy versioning tries its best to adhere to `Semantic Versioning
<https://semver.org/spec/v2.0.0.html>`__.
See the `Python Enhancement Proposal (PEP) 440 <https://peps.python.org/pep-0440/>`__
for supported version identifiers.

Preparation
-----------
- Locally, create a minor release branch from the ``develop`` branch when making a minor
  release, or create a patch release branch from the ``main`` branch when making a patch
  release. Ideally, a patch release is published immediately after a bug fix is merged
  on ``main``. Therefore, it might be best to do the release updates directly on the bug
  fix branch. This way, we do not have to create a separate patch release branch.

- Run tutorial notebooks and examples in the documentation locally.
  Confirm that they produce the expected results.
  From time to time, check the documentation links (``make linkcheck``).
  Fix broken links.

- Ensure all contributors are included and sorted correctly by reviewing the contributor
  list ``__credits__`` in ``kikuchipy/release.py``.
  Do the same for the Zenodo contributors file ``.zenodo.json``.
  Review ``.all-contributorsrc`` and regenerate the table if necessary.

- Increment the version number in ``kikuchipy/release.py``.
  Review and clean up ``CHANGELOG.rst`` as per Keep a Changelog.

- Make a PR of the release branch to ``main``.
  Discuss the changelog with others and make any changes *directly* to the release
  branch.
  Merge the branch onto ``main``.

Tag and release
---------------
- If the ``__version__`` in ``kikuchipy/release.py`` on ``main`` has changed in a new
  commit, a tagged, annotated release *draft* is automatically created.
  If ``__version__`` is now "0.42.0", the release name is "kikuchipy 0.42.0" and the
  tag name is "v0.42.0".
  The tag target will be the ``main`` branch.
  The release body contains a static description and a link to the changelog.
  This release draft can be published as is.
  Or, changes to the release body can be made before publishing.

- Monitor the publish workflow to ensure the release is successfully published to PyPI.

Post-release action
-------------------
- Monitor the `documentation build <https://readthedocs.org/projects/kikuchipy/builds>`__
  to ensure that the new stable documentation is successfully built from the release.

- Ensure that `Zenodo <https://doi.org/10.5281/zenodo.3597646>`__ displays the new
  release.

- Ensure that Binder can run the tutorial notebooks by clicking the Binder badge in the
  top banner of one of the tutorials in the `online documentation
  <https://kikuchipy.org/en/stable>`__.

- Bring changes on ``main`` onto ``develop`` by branching from ``main``, merging
  ``develop`` onto the new branch, and fixing potential conflicts.
  After any conflicts are fixed, update or revert ``__version__`` and make any updates
  to this guide if necessary.
  Make a PR to ``develop`` and merge.

- Tidy up GitHub issues and close the corresponding milestone.

- A PR to the conda-forge feedstock will automatically be created by the conda-forge
  bot.
  Follow the instructions in the PR and any relevant instructions in the conda-forge
  documentation on updating packages.
  Merge the PR after checks pass.
  Monitor the Azure pipeline CI to ensure that the release is successfully published to
  conda-forge.
