---
name: Bug report
about: Create a report to help us improve 
---

#### Description
<!-- A clear and concise description of what the bug is. -->

#### Way to reproduce

```python
>>> import kikuchipy as kp
>>> import numpy as np
>>> s = kp.signals.EBSD(np.zeros((10, 10, 10, 10)))
>>> # The bug here...
```

#### Version information

```python
# Paste the output of the following python commands
import sys; print(sys.version)
import platform; print(platform.platform())
import kikuchipy; print("kikuchipy version: {}".format(kikuchipy.__version__))
```

```python
# Your output here
```

#### Expected behaviour
<!-- A clear and concise description of what you expected to happen. -->
