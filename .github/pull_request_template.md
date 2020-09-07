#### Description of the change

#### Progress of the PR

- [ ] [Docstrings for all functions](https://github.com/numpy/numpy/blob/master/doc/example.py)
- [ ] Unit tests with pytest for all lines
- [ ] Clean style in [as per black](https://black.readthedocs.io/en/stable/the_black_code_style.html)

#### Minimal example of the bug fix or new feature

```python
>>> import kikuchipy as kp
>>> import numpy as np
>>> s = kp.signals.EBSD(np.zeros((10, 10, 10, 10)))
>>> # Your new feature...
```

#### For reviewers

<!-- Don't remove the checklist below. -->
- [ ] Check that the PR title is short, concise, and will make sense 1 year
  later.
- [ ] Check that new functions are imported in corresponding `__init__.py`.
- [ ] Check that new features, API changes, and deprecations are mentioned in
      the unreleased section in `doc/changelog.rst`.
