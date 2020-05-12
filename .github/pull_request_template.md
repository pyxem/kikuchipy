## Description

<!-- If this is a bug-fix or enhancement, state the issue # it closes -->
<!-- If this is a new feature, reference what paper it implements. -->

## Checklist

<!-- It's fine to submit PRs which are a work in progress! -->
<!-- But before they are merged, all PRs should provide: -->
- [ ] [Docstrings for all functions](https://github.com/numpy/numpy/blob/master/doc/example.py)
- [ ] Unit tests with pytest for all lines
- [ ] Clean style in [as per black](https://black.readthedocs.io/en/stable/the_black_code_style.html)

<!-- For detailed information on these and other aspects see -->
<!-- the kikuchipy contribution guidelines. -->
<!-- https://kikuchipy.readthedocs.io/en/latest/contributing.html -->

#### Minimal example of the bug fix or new feature
<!-- Note that this example can be useful to update the user guide with! -->

```python
>>> import kikuchipy as kp
>>> import numpy as np
>>> s = kp.signals.EBSD(np.zeros((10, 10, 10, 10)))
>>> # Your new feature...
```

## For reviewers

<!-- Don't remove the checklist below. -->
- [ ] Check that the PR title is short, concise, and will make sense 1 year
  later.
- [ ] Check that new functions are imported in corresponding `__init__.py`.
- [ ] Check that new features, API changes, and deprecations are mentioned in
      the unreleased section in `doc/changelog.rst`.
