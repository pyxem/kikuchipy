Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/kikuchipy/actions>`__ to ensure that
kikuchipy can be installed on Windows, macOS, and Linux (Ubuntu).
After a successful installation of the package, the CI server runs the tests.
After the tests return no errors, code coverage is reported to `Codecov
<https://app.codecov.io/github/pyxem/kikuchipy>`__.
Add ``"[skip ci]"`` to a commit message to skip this workflow on any commit to a pull
request.