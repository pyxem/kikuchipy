Using Git
=========

Making changes
--------------

If you want to add a new feature, branch off of the ``develop`` branch, and when you
want to fix a bug, branch off of ``main`` instead.

To create a new feature branch that tracks the upstream development branch::

    git checkout develop -b your-awesome-feature-name upstream/develop

When you've made some changes you can view them with::

    git status

Add and commit your created, modified or deleted files::

    git add my-file-or-directory
    git commit -s -m "An explanatory commit message"

The ``-s`` makes sure that you sign your commit with your `GitHub-registered email
<https://github.com/settings/emails>`__ as the author. You can set this up following
`this GitHub guide
<https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address>`__.

Keeping your branch up-to-date
------------------------------

If you are adding a new feature, make sure to merge ``develop`` into your feature
branch. If you are fixing a bug, merge ``main`` into your bug fix branch instead.

To update a feature branch, switch to the ``develop`` branch::

    git checkout develop

Fetch changes from the upstream branch and update ``develop``::

    git pull upstream develop --tags

Update your feature branch::

    git checkout your-awesome-feature-name
    git merge develop

Sharing your changes
--------------------

Update your remote branch::

    git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects#making-a-pull-request>`__
to kikuchipy's ``develop`` branch for new features and ``main`` branch for bug fixes.
Good job!