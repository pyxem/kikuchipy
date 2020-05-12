How to make a new release of ``kikuchipy``
==========================================

This guide should be updated after every new release!

- Create a release branch v<major>.<minor>.x. If a new minor release is to be
  made, branch off of master via the GitHub repo, and pull this branch locally.
  If a new patch release is to be made, pull the existing minor branch locally
  and branch off of it.

- Review and clean up doc/changelog.rst as per Keep a Changelog. Make sure all
  contributors and reviewers are acknowledged. Increment the version number in
  `release.py`. Make a PR to the release branch.

- Create a PR from the release branch to master. Discuss the changelog with
  others, and make the changes directly to the release branch. After all checks
  are green and the PR is merged, cherry-pick any resulting changes to the
  release branch.

- Make sure that the documentation, with the changelog updates, can be
  successfully built from the release branch by making Read the Docs build the
  release branch: https://readthedocs.org/projects/kikuchipy/.

- On the master branch, increment the version number in `release.py` to the next
  ``.dev0``.

- Create a release draft (tag) via the GitHub repo from the release branch with
  the correct tag version name, e.g. v0.42.x and release title (see previous
  releases). Add the new release notes from the changelog. Publish the release.

- Monitor Travis CI to ensure the release is successfully published to PyPI.

conda-forge
-----------

A kikuchipy build recipe is available at
https://github.com/conda-forge/kikuchipy-feedstock. conda-forge documentation is
available at https://conda-forge.org/docs/index.html.

- Fork the feedstock.

- Create a new release branch named v<major>.<minor>.x off of master.

- Increment the version number in `recipe/meta.yml`. Get the SHA256 for the
  package distribution (`.tar.gz`) from PyPI
  https://pypi.org/project/kikuchipy/.

- Make a PR to master from the release branch. Follow the relevant instructions
  from the conda-forge documentation on updating packages. Merge the PR after
  all checks are green.

- Monitor the Azure pipeline CI to ensure the release is successfully published
  to conda-forge.
