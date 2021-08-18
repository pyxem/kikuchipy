#### Description of the change
<!-- Remember to branch off develop for new features and main for patches. -->

#### Progress of the PR
- [ ] [Docstrings for all functions](https://github.com/numpy/numpy/blob/master/doc/example.py)
- [ ] Unit tests with pytest for all lines
- [ ] Clean code style by [running black via pre-commit](https://kikuchipy.org/en/latest/contributing.html#code-style)

#### Minimal example of the bug fix or new feature
```python
>>> import kikuchipy as kp
>>> s = kp.data.nickel_ebsd_small()
>>> s
<EBSD, title: patterns My awes0m4 ..., dimensions: (3, 3|60, 60)>
>>> # Your new feature...
```

#### For reviewers
<!-- Don't remove the checklist below. -->
- [ ] The PR title is short, concise, and will make sense 1 year later.
- [ ] New functions are imported in corresponding `__init__.py`.
- [ ] New features, API changes, and deprecations are mentioned in the unreleased
      section in `doc/changelog.rst`.
- [ ] New contributors are added to .all-contributorsrc and the table is regenerated.
