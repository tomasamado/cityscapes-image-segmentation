 ## Controlling sources of randomnes
 
Include the following code before executing an experiment:

```python
import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)
```

Refer to the [PyTorch documentation](https://pytorch.org/docs/stable/notes/randomness.html) for more information.
