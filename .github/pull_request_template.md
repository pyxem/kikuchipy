<!-- Requirements -->
<!-- * Read the contributor guide: https://kikuchipy.readthedocs.io/en/latest/contributing.html -->
<!-- * Fill out the template. It helps the review process and is useful to summarise the PR. -->
<!-- * This template can be updated during the progression of the PR to summarise its status -->

#### Description
<!-- What does this pull request (PR) do? Why is it necessary? -->
<!-- A few sentences and/or a bullet list. -->

#### Type of change
<!-- Please delete options that are not relevant. -->
- [ ] Bug-fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

#### Progress
- [ ] Change(s) implemented (can be split into several points)
- [ ] Docstrings updated (if appropriate)
- [ ] Code is commented, particularly in hard-to-understand areas (if appropriate)
- [ ] Documentation and/or user guide updated (if appropriate)
- [ ] Tests have been written
- [ ] Ready for review!

#### How has this been tested?
<!-- Please describe the tests that you ran to verify your changes. -->
- [ ] example: the test suite for my feature covers cases x, y, and z
- [ ] example: all tests pass with my change

#### Minimal example of the bug fix or new feature
<!-- Note that this example can be useful to update the user guide with! -->

```python
>>> import kikuchipy as kp
>>> import numpy as np
>>> s = kp.signals.EBSD(np.zeros((10, 10, 10, 10)))
>>> # Your new feature...
```

#### References
<!-- What resources, documentation, and guides were used in the creation of this PR? -->
<!-- If this is a bug-fix or otherwise resolves an issue, reference it here with "closes #(issue)" -->
