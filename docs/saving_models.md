# Saving and loading model's checkpoints


### Saving a model's checkpoint

The elements that should be included when saving a model are:

- **epoch**: the number of the epoch util which the model was trained.
- **model_state_dict**: model's state dictionary (weights)
- **optimizer_state_dict**: optimizer's state dictionary 
- **loss**: training loss of the last epoch
- others (validation F1-score, training F1-score, ...)

In order to save the model:

```python
import copy
import torch

model_path = 'models/task2/with_weights/best_model.pt'

checkpoint = {
    'epoch': 10,
    'model_state_dict': copy.deepcopy(model.state_dict()),
    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
    'loss': loss,
    ...
    }
    
torch.save(checkpoint, model_path)
```

### Loading a model's checkpoint 

```python
import torch

model_path = 'models/task2/with_weights/best_model.pt'

model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

Refer to the [Pytorch tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for more information.
