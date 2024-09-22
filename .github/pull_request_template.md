#### Description of the change
<!-- Remember to branch off the develop branch for new features and the main branch for patches. -->


#### Progress of the PR
- [ ] [Docstrings for all functions](https://numpydoc.readthedocs.io/en/latest/example.html)
- [ ] Unit tests with pytest for all lines
- [ ] Clean code style by [running black via pre-commit](https://kikuchipy.org/en/latest/dev/code_style.html)

#### Minimal example of the bug fix or new feature
```python
>>> import kikuchipy as kp
>>> s = kp.data.nickel_ebsd_small()
>>> s
<EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>
>>> # Your new feature...
```

#### For reviewers
<!-- Don't remove the checklist below. -->
- [ ] The PR title is short, concise, and will make sense 1 year later.
- [ ] New functions are imported in corresponding `__init__.py`.
- [ ] New features, API changes, and deprecations are mentioned in the unreleased
      section in `CHANGELOG.rst`.
- [ ] New contributors are added to `kikuchipy/__init__.py` and `.zenodo.json`.
