# vit-pytorch

This repository is intended as a personal experiment to try and implement existing model of neural network based on 
transformers.

This repository **does not** include any resource for training the models. 

For more details on the models themselves, please refer to the corresponding papers.


# Current implemented models

## Vision Transformer

### Usage

```python
from models.vit import VisionTransformer
import torch


vit = VisionTransformer(num_classes=1000,
                        input_size=(224, 224),
                        patch_size=(16, 16),
                        depth=12,
                        hidden_dim=768,
                        num_heads=12)
x = torch.rand((1, 3, 224, 224)) # input
y = vit(x) # output
print(y.shape) # (1, 1000)
```

### Paper

```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
