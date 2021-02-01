How to make a new release of ``kikuchipy``
==========================================

- Create a release branch v<major>.<minor>.x. If a new minor release is to be
  made, branch off of master via the GitHub repo, and pull this branch locally.
  If a new patch release is to be made, pull the existing minor branch locally
  and branch off of it.
- Review and clean up `doc/changelog.rst` as per Keep a Changelog. Make sure all
  contributors and reviewers are acknowledged. Increment the version number in
  `release.py`. Make a PR to the release branch.
- Create a PR from the release branch to master. Discuss the changelog with
  others, and make any changes *directly* to the release branch.
- On the master branch, increment the version number in `release.py` to the next
  v<major>.<minor>.dev0.
- If a release candidate (RC) is to be made, make a PR to the release branch
  with `-rc.1` added to the version name, like v0.42.0-rc.1, merge the PR after
  all checks pass, and publish an RC release off of the release branch. See that
  it is made available via PyPI, download it with the `dev` dependencies, and
  run the tests to make sure everything is as it should be. Also check that
  Binder can build the repository successfully by going to the Read the Docs PR
  check in the RC PR above after the RC release.
- Create a release draft (tag) via the GitHub repo from the release branch with
  the correct tag version name, e.g. v0.42.0, and release title
  "kikuchipy 0.42.0". Add the new release notes from the changelog, and convert
  any reStructuredText formatting to Markdown by hand. Publish the release.
- Monitor the publish GitHub Action to ensure the release is successfully
  published to PyPI.
- Download the new version from PyPI with the `dev` dependencies with
  `pip install kikuchipy[dev]==0.42.0` locally and run the tests with
  `pytest --pyargs kikuchipy` to make sure everything is as it should be.
- Make a PR to master with any updates to this guide if necessary.

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
