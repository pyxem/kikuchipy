How to make a new release of ``kikuchipy``
==========================================

kikuchipy's branching model is similar to the Gitflow Workflow (`original blog post
<https://nvie.com/posts/a-successful-git-branching-model/>`_).

- If a minor release is to be made, create a release branch v<major>.<minor>.0 off of
  the `develop` branch locally. If a patch release is to be made, create a release
  branch v<major>.<minor>.<patch> off of the `main` branch locally instead. Ideally, a
  patch release should be made immediately after a bug fix has been made. Therefore, it
  might be best to do the release updates, listed in the following steps, directly on
  the bug fix branch, so that no separate patch release branch has to be made. The bug
  fix branch should of course be branched off of `main`.
- Increment the version number in `release.py`. Review and clean up `doc/changelog.rst`
  as per Keep a Changelog. Make sure all contributors and reviewers are acknowledged.
- Make a PR of the release branch to `main`. Discuss the changelog with others, and
  make any changes *directly* to the release branch. Merge the branch into `main`. Then
  make a PR of `main` to `develop`, and merge this.
- Create a release draft (tag) via the GitHub repo from main with the correct tag
  version name, e.g. v0.42.0, and release title "kikuchipy 0.42.0". Add the new release
  notes from the changelog, and convert any reStructuredText formatting to Markdown by
  hand. Publish the release.
- Monitor the publish GitHub Action to ensure the release is successfully published to
  PyPI.
- Download the new version from PyPI with the `dev` dependencies with
  `pip install kikuchipy[dev]==0.42.0` locally and run the tests with
  `pytest --pyargs kikuchipy` to make sure everything is as it should be.
- Make a PR to `develop` with any updates to this guide if necessary.

conda-forge
-----------
A kikuchipy build recipe is available at
https://github.com/conda-forge/kikuchipy-feedstock. conda-forge documentation is
available at https://conda-forge.org/docs/index.html.

- Normally, a conda-forge bot will make a PR to the feedstock with the necessary
  changes within a day of a new PyPI release, to our great convenience!
- Proceed with the last steps below.

If the bot for some reason does not make this PR:

- Fork the feedstock.
- Create a *local* release branch named v<major>.<minor>.x off of master.
- Increment the version number in `recipe/meta.yml`. Get the SHA256 for the
  package distribution (`.tar.gz`) from PyPI
  https://pypi.org/project/kikuchipy/.
- Make a PR to master from your release branch.
- Proceed with the last steps below.

Last steps:

- Follow the relevant instructions from the conda-forge documentation on
  updating packages, as well as the instructions in the PR, and merge it after
  all checks are green.
- Monitor the Azure pipeline CI to ensure the release is successfully published
  to conda-forge.
