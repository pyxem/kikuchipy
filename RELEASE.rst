How to make a new release of ``kikuchipy``
==========================================

kikuchipy's branching model is similar to the Gitflow Workflow (`original blog post
<https://nvie.com/posts/a-successful-git-branching-model/>`_).

Preparation
-----------
- If a minor release is to be made, create a release branch v<major>.<minor>.0 off of
  the ``develop`` branch locally. If a patch release is to be made, create a release
  branch v<major>.<minor>.<patch> off of the ``main`` branch locally instead. Ideally, a
  patch release should be made immediately after a bug fix has been made. Therefore, it
  might be best to do the release updates, listed in the following steps, directly on
  the bug fix branch, so that no separate patch release branch has to be made. The bug
  fix branch should of course be branched off of ``main``.
- Run all user guide notebooks locally and confirm that they produce the expected
  results.
- Review the contributor list ``__credits__`` in ``release.py`` to ensure all
  contributors are included and sorted correctly. Review ``.all-contributorsrc`` and
  regenerate the table if necessary.
- Increment the version number in ``release.py``. Review and clean up ``CHANGELOG.rst``
  as per Keep a Changelog. Make sure all contributors and reviewers are acknowledged.
- Make a PR of the release branch to ``main``. Discuss the changelog with others, and
  make any changes *directly* to the release branch. Merge the branch onto ``main``.

Release (and tag)
-----------------
- If the ``__version__`` in ``release.py`` on ``main`` has changed in a new commit, a
  tagged, annotated release *draft* is automatically created. If ``__version__`` is now
  "0.42.0", the release name is "kikuchipy 0.42.0", and the tag name is "v0.42.0". The
  tag target will be the ``main`` branch. The release body contains a static description
  and a link to the changelog. This release draft can be published as is, or changes to
  the release body can be made before publishing.
- Monitor the publish GitHub Action to ensure the release is successfully published to
  PyPI.

Post-release action
-------------------
- Monitor the `documentation build <https://readthedocs.org/projects/kikuchipy/builds>`_
  to make sure the new stable documentation is successfully built from the release.
- Ensure that `Zenodo <https://doi.org/10.5281/zenodo.3597646>`_ received the new
  release.
- Ensure that Binder can run the user guide notebooks by clicking the Binder badges in
  the top banner of one of the user guide notebooks via `Read The Docs
  <https://kikuchipy.org/en/stable>`_.
- Bring changes on ``main`` onto ``develop`` by branching off of ``main``, merge
  ``develop`` onto the new branch, fix conflicts, and make a PR to ``develop``.
- Make a post-release PR to ``develop`` with ``__version__`` updated (or reverted), e.g.
  from "0.42.0" to "0.43.dev0", and any updates to this guide if necessary.
- Tidy up GitHub issues and close the corresponding milestone.
- A PR to the conda-forge feedstock will be created by the conda-forge bot. Follow the
  relevant instructions from the conda-forge documentation on updating packages, as well
  as the instructions in the PR. Merge after checks pass. Monitor the Azure pipeline CI
  to ensure the release is successfully published to conda-forge.
